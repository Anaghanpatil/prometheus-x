"""Evaluate PatchCore on Bottle data with explicit training/calibration splits."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from backend.ai.patchcore_service import CHECKPOINT_PATH, OUTPUT_FOLDER, inspect_image
from src.calibration import Calibrator
from src.decision import combine_decision
from src.severity import estimate_severity
from src.shift import ShiftDetector


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIXEL_REFERENCE_QUANTILE = 0.999
PIXEL_REFERENCE_SAMPLES_PER_IMAGE = 25000
PIXEL_METRIC_SIDE = 256
DEFAULT_PIXEL_THRESHOLD = 0.5
REFERENCE_IMAGE_COUNT = 20


def collect_images(split):
    """Return sorted (ground-truth label, path) pairs for a data split."""
    bottle_root = PROJECT_ROOT / "dataset" / "bottle"
    if split == "train":
        folders = [("good", bottle_root / "train" / "good")]
    elif split == "test":
        test_root = bottle_root / "test"
        folders = [
            (folder.name, folder)
            for folder in sorted(test_root.iterdir())
            if folder.is_dir()
        ]
    else:
        raise ValueError(f"Unsupported split: {split}")

    images = []
    for label, folder in folders:
        if not folder.is_dir():
            continue
        images.extend(
            (label, path)
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return images


def _read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _load_ground_truth(label, image_path):
    if label == "good":
        return np.zeros(_read_rgb(image_path).shape[:2], dtype=bool), None

    mask_path = (
        PROJECT_ROOT
        / "dataset"
        / "bottle"
        / "ground_truth"
        / label
        / f"{image_path.stem}_mask.png"
    )
    if not mask_path.is_file():
        raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) > 0
    return mask, mask_path


def _predict(image_path, output_dir):
    started = time.perf_counter()
    result = inspect_image(image_path, output_folder=output_dir)
    elapsed = time.perf_counter() - started
    anomaly_map = np.asarray(result.pop("anomaly_map"), dtype=np.float32)
    if anomaly_map.ndim != 2 or not np.isfinite(anomaly_map).all():
        raise ValueError(f"Invalid anomaly map for {image_path}: {anomaly_map.shape}")
    return result, anomaly_map, elapsed


def _sample_reference_pixels(anomaly_map, seed):
    flat = anomaly_map.ravel()
    sample_count = min(PIXEL_REFERENCE_SAMPLES_PER_IMAGE, flat.size)
    indices = np.random.default_rng(seed).choice(flat.size, sample_count, replace=False)
    return flat[indices]


def _pixel_threshold(samples, exclude_index=None):
    reference = [sample for index, sample in enumerate(samples) if index != exclude_index]
    if not reference:
        raise ValueError("At least one normal pixel-map reference is required.")
    threshold = float(np.quantile(np.concatenate(reference), PIXEL_REFERENCE_QUANTILE))
    return max(threshold, 1e-6)


def _load_shift_detector():
    images = collect_images("train")
    if len(images) < 5:
        raise RuntimeError("At least 5 normal training images are required for shift references.")
    positions = np.linspace(0, len(images) - 1, min(REFERENCE_IMAGE_COUNT, len(images)), dtype=int)
    reference_images = [_read_rgb(images[index][1]) for index in positions]
    return ShiftDetector().fit(reference_images), len(reference_images)


def _save_localization(image_path, output_dir, anomaly_map, pixel_threshold):
    severity = estimate_severity(anomaly_map, pixel_threshold)
    mask = severity.mask.astype(np.uint8) * 255
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    map_path = output_dir / f"{image_path.stem}_map.npz"
    Image.fromarray(mask).save(mask_path)
    np.savez_compressed(map_path, anomaly_map=anomaly_map)
    return severity, mask_path, map_path, mask.astype(bool)


def _make_record(
    split,
    label,
    image_path,
    result,
    anomaly_map,
    inference_seconds,
    pixel_threshold,
    calibrator,
    shift_detector,
    output_dir,
    calibration_sample_count,
):
    gt_mask, gt_path = _load_ground_truth(label, image_path)
    severity, mask_path, map_path, predicted_mask = _save_localization(
        image_path,
        output_dir,
        anomaly_map,
        pixel_threshold,
    )
    score = float(result["anomaly_score"])
    shift = shift_detector.assess(_read_rgb(image_path)) if shift_detector else None
    calibration = calibrator.predict(score) if calibrator else None

    if split == "test":
        decision = combine_decision(
            calibration.verdict,
            shift.shifted,
            shift.reliability,
            getattr(shift, "quality_issues", []),
        )
    else:
        decision = None

    return {
        "split": split,
        "label": label,
        "image_name": image_path.name,
        "image_path": str(image_path),
        "status": "success",
        "anomaly_score": score,
        "model_prediction": "Anomaly Detected" if result["is_anomaly"] else "Normal",
        "calibration_verdict": calibration.verdict if calibration else "Not evaluated",
        "normality_percentile": calibration.normality_percentile if calibration else None,
        "calibration_sample_count": calibration_sample_count,
        "lower_threshold": calibration.lower_threshold if calibration else None,
        "upper_threshold": calibration.upper_threshold if calibration else None,
        "pixel_threshold": pixel_threshold,
        "decision": decision.action if decision else "TRAINING SET ONLY",
        "decision_rationale": decision.rationale if decision else ["Training images are not independent evaluation data"],
        "manual_review_required": decision.manual_review_required if decision else False,
        "severity": severity.level,
        "severity_index": severity.index,
        "affected_area_pct": severity.area_pct,
        "peak_anomaly": severity.peak_score,
        "mean_defect_anomaly": severity.mean_defect_score,
        "defect_region_count": severity.num_regions,
        "bounding_boxes_xyxy": severity.bboxes,
        "ground_truth_mask_path": str(gt_path) if gt_path else "",
        "mask_path": str(mask_path),
        "anomaly_map_path": str(map_path),
        "heatmap_path": result["heatmap_path"],
        "overlay_path": result["overlay_path"],
        "shift_detected": shift.shifted if shift else False,
        "shift_reasons": shift.reasons if shift else [],
        "shift_score": shift.max_z_score if shift else None,
        "reliability": shift.reliability if shift else "Unavailable",
        "reliability_score_heuristic": shift.reliability_score if shift else None,
        "quality_issues": getattr(shift, "quality_issues", []) if shift else [],
        "inference_seconds": round(float(inference_seconds), 6),
        "error": "",
        "_gt_mask": gt_mask,
        "_predicted_mask": predicted_mask,
        "_anomaly_map": anomaly_map,
    }


def _failure_record(split, label, image_path, error):
    return {
        "split": split,
        "label": label,
        "image_name": image_path.name,
        "image_path": str(image_path),
        "status": "failure",
        "anomaly_score": None,
        "model_prediction": "",
        "calibration_verdict": "",
        "normality_percentile": None,
        "calibration_sample_count": 0,
        "lower_threshold": None,
        "upper_threshold": None,
        "pixel_threshold": None,
        "decision": "",
        "decision_rationale": [],
        "manual_review_required": False,
        "severity": "",
        "severity_index": None,
        "affected_area_pct": None,
        "peak_anomaly": None,
        "mean_defect_anomaly": None,
        "defect_region_count": 0,
        "bounding_boxes_xyxy": [],
        "ground_truth_mask_path": "",
        "mask_path": "",
        "anomaly_map_path": "",
        "heatmap_path": "",
        "overlay_path": "",
        "shift_detected": False,
        "shift_reasons": [],
        "shift_score": None,
        "reliability": "Unavailable",
        "reliability_score_heuristic": None,
        "quality_issues": [],
        "inference_seconds": None,
        "error": f"{type(error).__name__}: {error}",
    }


def _safe_auc(labels, scores, metric):
    if len(set(labels)) < 2:
        return None
    return float(metric(labels, scores))


def _binary_summary(records, labels, scores, predictions):
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    fpr_values, tpr_values, _ = roc_curve(labels, scores)
    precisions, recalls, _ = precision_recall_curve(labels, scores)
    return {
        "images": len(records),
        "image_auroc": _safe_auc(labels, scores, roc_auc_score),
        "image_average_precision": _safe_auc(labels, scores, average_precision_score),
        "precision_defect_decision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_defect_decision": float(recall_score(labels, predictions, zero_division=0)),
        "f1_defect_decision": float(f1_score(labels, predictions, zero_division=0)),
        "false_positive_rate_defect_decision": float(fp / (fp + tn)) if fp + tn else None,
        "confusion_matrix_labels": ["normal", "defect"],
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "uncertain_count": sum(row["decision"] == "UNCERTAIN" for row in records),
        "manual_review_count": sum(row["decision"] == "MANUAL REVIEW REQUIRED" for row in records),
        "decision_metric_semantics": "Only DEFECT counts as an automatic positive; UNCERTAIN and MANUAL REVIEW REQUIRED are non-positive outcomes and are counted in the confusion matrix false-negative cell for defective ground truth.",
        "normal_flag_rate_including_review": float(
            sum(row["label"] == "good" and row["decision"] != "NORMAL" for row in records)
            / max(sum(row["label"] == "good" for row in records), 1)
        ),
        "roc_curve": {"fpr": fpr_values.tolist(), "tpr": tpr_values.tolist()},
        "precision_recall_curve": {"precision": precisions.tolist(), "recall": recalls.tolist()},
    }


def _pixel_metrics(records):
    true_pixels = []
    score_pixels = []
    intersection = 0
    predicted_count = 0
    ground_truth_count = 0
    for row in records:
        truth = row["_gt_mask"]
        prediction = row["_predicted_mask"]
        amap = row["_anomaly_map"]
        intersection += int(np.logical_and(truth, prediction).sum())
        predicted_count += int(prediction.sum())
        ground_truth_count += int(truth.sum())

        resized_scores = np.asarray(
            Image.fromarray(amap, mode="F").resize(
                (PIXEL_METRIC_SIDE, PIXEL_METRIC_SIDE),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
        resized_truth = np.asarray(
            Image.fromarray(truth.astype(np.uint8) * 255).resize(
                (PIXEL_METRIC_SIDE, PIXEL_METRIC_SIDE),
                Image.Resampling.NEAREST,
            )
        ) > 0
        true_pixels.append(resized_truth.ravel())
        score_pixels.append(resized_scores.ravel())

    labels = np.concatenate(true_pixels).astype(np.uint8)
    scores = np.concatenate(score_pixels)
    union = predicted_count + ground_truth_count - intersection
    denominator = predicted_count + ground_truth_count
    return {
        "pixel_metric_resolution": [PIXEL_METRIC_SIDE, PIXEL_METRIC_SIDE],
        "pixel_auroc": _safe_auc(labels, scores, roc_auc_score),
        "pixel_average_precision": _safe_auc(labels, scores, average_precision_score),
        "dice_micro_full_resolution": float(2 * intersection / denominator) if denominator else 1.0,
        "iou_micro_full_resolution": float(intersection / union) if union else 1.0,
        "pro_aupro": None,
        "pro_aupro_note": "Not calculated; no validated PRO implementation is included.",
    }


def _summarize_test(records, total_seconds, calibration_sample_count, pixel_threshold):
    successful = [row for row in records if row["status"] == "success"]
    labels = [int(row["label"] != "good") for row in successful]
    scores = [float(row["anomaly_score"]) for row in successful]
    predictions = [int(row["decision"] == "DEFECT") for row in successful]
    if len(set(labels)) < 2:
        raise RuntimeError("Image-level metrics require both normal and defective test samples.")

    summary = _binary_summary(successful, labels, scores, predictions)
    summary["pixel_metrics"] = _pixel_metrics(successful)
    summary["category_metrics"] = {}
    for category in sorted({row["label"] for row in successful if row["label"] != "good"}):
        selected = [row for row in successful if row["label"] in {"good", category}]
        category_labels = [int(row["label"] == category) for row in selected]
        category_scores = [float(row["anomaly_score"]) for row in selected]
        category_predictions = [int(row["decision"] == "DEFECT") for row in selected]
        cm = confusion_matrix(category_labels, category_predictions, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in cm.ravel())
        summary["category_metrics"][category] = {
            "samples": len(selected),
            "defect_samples": sum(category_labels),
            "image_auroc": _safe_auc(category_labels, category_scores, roc_auc_score),
            "image_average_precision": _safe_auc(category_labels, category_scores, average_precision_score),
            "precision_defect_decision": float(precision_score(category_labels, category_predictions, zero_division=0)),
            "recall_defect_decision": float(recall_score(category_labels, category_predictions, zero_division=0)),
            "false_positive_rate_defect_decision": float(fp / (fp + tn)) if fp + tn else None,
            "confusion_matrix": [[tn, fp], [fn, tp]],
        }

    summary.update({
        "evaluation_protocol": "PatchCore trained on train/good; 20 test/good images calibrate defects; each normal test image uses leave-one-out calibration.",
        "training_images_excluded_from_test_metrics": 209,
        "normal_calibration_sample_count_for_defects": calibration_sample_count,
        "pixel_threshold_reference_quantile": PIXEL_REFERENCE_QUANTILE,
        "pixel_threshold_for_defects": pixel_threshold,
        "successful_predictions": len(successful),
        "failed_predictions": len(records) - len(successful),
        "total_execution_seconds": round(total_seconds, 3),
        "average_inference_seconds": float(np.mean([row["inference_seconds"] for row in successful])),
        "defect_categories_absent_from_training": ["broken_large", "broken_small", "contamination"],
        "external_image_results": "No external image set is included; uploaded examples are single-case demonstrations, not generalization evidence.",
        "false_positive_examples": [
            row["image_path"] for row in successful
            if row["label"] == "good" and row["decision"] == "DEFECT"
        ],
        "false_negative_examples": [
            row["image_path"] for row in successful
            if row["label"] != "good" and row["decision"] != "DEFECT"
        ],
        "defective_images_routed_to_review": [
            {"image_path": row["image_path"], "decision": row["decision"]}
            for row in successful
            if row["label"] != "good" and row["decision"] in {"UNCERTAIN", "MANUAL REVIEW REQUIRED"}
        ],
        "training_results_are_independent_test": False,
    })
    return summary


def _write_reports(report_dir, records, summary):
    report_path = report_dir / "report.csv"
    export_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in records]
    fieldnames = list(export_rows[0])
    with report_path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in export_rows:
            writer.writerow({
                key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })

    json_report = {
        "summary": summary,
        "images": export_rows,
    }
    with (report_dir / "report.json").open("w", encoding="utf-8") as report_file:
        json.dump(json_report, report_file, indent=2, allow_nan=False)
    return report_path


def _load_existing_calibration():
    path = OUTPUT_FOLDER / "batch_test" / "calibration.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("checkpoint_mtime_ns") != CHECKPOINT_PATH.stat().st_mtime_ns:
        return None
    return data


def run_batch(split, output_dir=None, progress_callback=None):
    """Run the separate training inspection or leakage-aware 83-image test."""
    images = collect_images(split)
    if not images:
        raise FileNotFoundError(f"No Bottle {split} images were found.")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"PatchCore checkpoint not found: {CHECKPOINT_PATH}")

    report_dir = Path(output_dir) if output_dir else OUTPUT_FOLDER / f"batch_{split}"
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stamp = CHECKPOINT_PATH.stat().st_mtime_ns
    started = time.perf_counter()
    shift_detector, shift_reference_count = _load_shift_detector()
    records = []

    print(f"Running {len(images)} {split} images with one cached CPU model.")
    if split == "train":
        print("Training-set inspection only: these images are not independent evaluation data.")

    if split == "train":
        calibration_data = _load_existing_calibration()
        calibrator = Calibrator().fit(calibration_data["normal_scores"]) if calibration_data else None
        pixel_threshold = calibration_data["pixel_threshold"] if calibration_data else DEFAULT_PIXEL_THRESHOLD
        for index, (label, image_path) in enumerate(images, start=1):
            try:
                result, anomaly_map, elapsed = _predict(image_path, report_dir / label)
                record = _make_record(
                    split, label, image_path, result, anomaly_map, elapsed,
                    pixel_threshold, calibrator, shift_detector,
                    report_dir / label,
                    len(calibration_data["normal_scores"]) if calibration_data else 0,
                )
                for internal_key in ("_gt_mask", "_predicted_mask", "_anomaly_map"):
                    record.pop(internal_key, None)
            except Exception as exc:
                record = _failure_record(split, label, image_path, exc)
                print(f"FAILED [{index}/{len(images)}] {image_path}: {record['error']}")
            records.append(record)
            if progress_callback:
                progress_callback(index, len(images), record)
            if index % 20 == 0 or index == len(images):
                print(f"Processed {index}/{len(images)} images.")

        summary = {
            "split": "train/good",
            "training_set_only": True,
            "independent_test_metrics": None,
            "total_images": len(records),
            "successful_predictions": sum(row["status"] == "success" for row in records),
            "failed_predictions": sum(row["status"] != "success" for row in records),
            "average_inference_seconds": float(np.mean([
                row["inference_seconds"] for row in records
                if row["status"] == "success"
            ])),
            "note": "Results describe training-set behavior and must not be reported as independent test performance.",
            "total_execution_seconds": round(time.perf_counter() - started, 3),
            "shift_reference_images": shift_reference_count,
        }
    else:
        normal_images = [(label, path) for label, path in images if label == "good"]
        if len(normal_images) < 5:
            raise RuntimeError("At least five held-out normal test images are required for calibration.")

        normal_predictions = []
        for index, (label, image_path) in enumerate(normal_images, start=1):
            try:
                result, anomaly_map, elapsed = _predict(image_path, report_dir / label)
                normal_predictions.append((label, image_path, result, anomaly_map, elapsed))
            except Exception as exc:
                records.append(_failure_record(split, label, image_path, exc))
                print(f"FAILED calibration normal [{index}/{len(normal_images)}] {image_path}: {type(exc).__name__}: {exc}")
        if len(normal_predictions) < 5:
            raise RuntimeError("Fewer than five held-out normal test predictions succeeded; cannot calibrate.")

        normal_scores = [float(item[2]["anomaly_score"]) for item in normal_predictions]
        pixel_samples = [
            _sample_reference_pixels(item[3], 4200 + index)
            for index, item in enumerate(normal_predictions)
        ]
        full_calibrator = Calibrator().fit(normal_scores)
        full_pixel_threshold = _pixel_threshold(pixel_samples)
        calibration_data = {
            "normal_scores": normal_scores,
            "sample_count": len(normal_scores),
            "reference_paths": [str(item[1]) for item in normal_predictions],
            "source": "dataset/bottle/test/good; held out from PatchCore training",
            "checkpoint_mtime_ns": checkpoint_stamp,
            "pixel_threshold": full_pixel_threshold,
            "pixel_reference_quantile": PIXEL_REFERENCE_QUANTILE,
            "pixel_reference_samples_per_image": PIXEL_REFERENCE_SAMPLES_PER_IMAGE,
        }
        with (report_dir / "calibration.json").open("w", encoding="utf-8") as file:
            json.dump(calibration_data, file, indent=2, allow_nan=False)

        completed = len(records)
        for index, item in enumerate(normal_predictions):
            label, image_path, result, anomaly_map, elapsed = item
            loo_scores = [score for score_index, score in enumerate(normal_scores) if score_index != index]
            loo_calibrator = Calibrator().fit(loo_scores)
            loo_pixel_threshold = _pixel_threshold(pixel_samples, exclude_index=index)
            try:
                record = _make_record(
                    split, label, image_path, result, anomaly_map, elapsed,
                    loo_pixel_threshold, loo_calibrator, shift_detector,
                    report_dir / label, len(loo_scores),
                )
            except Exception as exc:
                record = _failure_record(split, label, image_path, exc)
            records.append(record)
            completed += 1
            if progress_callback:
                progress_callback(completed, len(images), record)

        for label, image_path in [(label, path) for label, path in images if label != "good"]:
            index = len(records) + 1
            try:
                result, anomaly_map, elapsed = _predict(image_path, report_dir / label)
                record = _make_record(
                    split, label, image_path, result, anomaly_map, elapsed,
                    full_pixel_threshold, full_calibrator, shift_detector,
                    report_dir / label, len(normal_scores),
                )
            except Exception as exc:
                record = _failure_record(split, label, image_path, exc)
                print(f"FAILED [{index}/{len(images)}] {image_path}: {record['error']}")
            records.append(record)
            if progress_callback:
                progress_callback(len(records), len(images), record)
            if len(records) % 10 == 0 or len(records) == len(images):
                print(f"Processed {len(records)}/{len(images)} images.")

        summary = _summarize_test(
            records,
            time.perf_counter() - started,
            len(normal_scores),
            full_pixel_threshold,
        )
        summary["shift_reference_images"] = shift_reference_count
        summary["calibration_source"] = calibration_data["source"]

    report_path = _write_reports(report_dir, records, summary)
    success_count = sum(row["status"] == "success" for row in records)
    failure_count = len(records) - success_count
    print(f"\nBatch complete: {success_count} succeeded, {failure_count} failed.")
    print(f"CSV report: {report_path}")
    print(f"JSON report: {report_dir / 'report.json'}")
    return success_count, failure_count, report_path


def run_shift_experiment(output_dir=None):
    """Evaluate fixed synthetic acquisition changes against train-only references."""
    report_dir = Path(output_dir) if output_dir else OUTPUT_FOLDER / "shift_experiment"
    report_dir.mkdir(parents=True, exist_ok=True)
    detector, reference_count = _load_shift_detector()
    controls = collect_images("test")
    controls = [(label, path) for label, path in controls if label == "good"]
    rows = []
    rng = np.random.default_rng(2026)

    for _, path in controls:
        source = _read_rgb(path)
        variants = {
            "control_original": source,
            "synthetic_brightness_plus_35": np.clip(source.astype(np.float32) + 35, 0, 255).astype(np.uint8),
            "synthetic_contrast_1_5x": np.clip((source.astype(np.float32) - 127.5) * 1.5 + 127.5, 0, 255).astype(np.uint8),
            "synthetic_gaussian_blur_r2": np.asarray(Image.fromarray(source).filter(ImageFilter.GaussianBlur(radius=2))),
            "synthetic_gaussian_noise_20": np.clip(source.astype(np.float32) + rng.normal(0, 20, source.shape), 0, 255).astype(np.uint8),
            "synthetic_rgb_cast": np.clip(source.astype(np.int16) + np.array([20, 0, -20]), 0, 255).astype(np.uint8),
        }
        for variant, image in variants.items():
            result = detector.assess(image)
            rows.append({
                "source_image": str(path),
                "variant": variant,
                "synthetic": variant != "control_original",
                "shift_detected": result.shifted,
                "reliability": result.reliability,
                "reliability_score_heuristic": result.reliability_score,
                "max_z_score": result.max_z_score,
                "reasons": result.reasons,
                "quality_issues": getattr(result, "quality_issues", []),
            })

    csv_path = report_dir / "report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})

    by_variant = {}
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {
            "samples": len(selected),
            "shift_detected": sum(row["shift_detected"] for row in selected),
            "detection_rate": float(np.mean([row["shift_detected"] for row in selected])),
        }
    summary = {
        "synthetic_experiment": True,
        "shift_reference_images": reference_count,
        "base_images": len(controls),
        "metrics_by_variant": by_variant,
        "note": "These are controlled synthetic pixel changes, not real manufacturing-domain shifts.",
    }
    with (report_dir / "report.json").open("w", encoding="utf-8") as file:
        json.dump({"summary": summary, "images": rows}, file, indent=2, allow_nan=False)
    print(f"Shift experiment CSV: {csv_path}")
    print(json.dumps(by_variant, indent=2))
    return summary, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--split",
        choices=("train", "test"),
        help="Run all 209 training images separately or evaluate all 83 MVTec test images.",
    )
    mode.add_argument(
        "--shift-experiment",
        action="store_true",
        help="Run labeled synthetic acquisition-shift controls using normal images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory; defaults under backend/ai/outputs/.",
    )
    args = parser.parse_args()
    if args.shift_experiment:
        run_shift_experiment(args.output_dir)
        return
    _, failed, _ = run_batch(args.split, args.output_dir)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()