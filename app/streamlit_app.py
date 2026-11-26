"""
PROMETHEUS-X — Main Streamlit Application
==========================================
Sections: Inspection, History, Feedback.
Author  : Person 3 (Application & Integration Layer)
Project : PROMETHEUS-X — Open-Set Industrial Anomaly Detection
Hackathon Problem : A06

Entry point for the Streamlit application.
Run locally with:
    streamlit run app/streamlit_app.py

─── Architecture ────────────────────────────────────────────────────────────
  streamlit_app.py   ← this file (UI orchestration)
  components.py      ← reusable HTML/Streamlit widgets
  styles.py          ← CSS stylesheet
  utils.py           ← helper functions (IDs, formatting, recommendations)
    backend/ai/patchcore_service.py ← trained PatchCore inference
    src/severity.py                 ← defect area and region assessment
    src/calibration.py              ← normal-reference score calibration
    src/shift.py                    ← image quality and distribution shift
"""

import sys
import json
import logging
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

# ─── Path setup (allow imports from project root) ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Project imports ──────────────────────────────────────────────────────────
from app.styles import get_main_css
from app.components import (
    render_hero,
    section_label,
    model_status_badge,
    render_uploaded_image,
    render_heatmap,
    render_mask,
    render_verdict,
    render_metrics,
    render_shift_warning,
    render_recommendation,
    render_history_table,
    render_feedback_form,
)
from app.utils import (
    load_image_from_upload,
    generate_inspection_id,
    current_timestamp,
    severity_label,
    reliability_label,
)

# ─── Model and intelligence integrations ─────────────────────────────────────
try:
    from backend.ai.patchcore_service import (
        CHECKPOINT_PATH,
        OUTPUT_FOLDER,
        inspect_image,
    )
    _MODEL_IMPORT_ERROR = None
except Exception as exc:
    CHECKPOINT_PATH = None
    OUTPUT_FOLDER = ROOT / "backend" / "ai" / "outputs"
    inspect_image = None
    _MODEL_IMPORT_ERROR = exc

try:
    from src.calibration import Calibrator
    from src.decision import combine_decision
    from src.severity import estimate_severity
    from src.shift import ShiftDetector
    _INTELLIGENCE_IMPORT_ERROR = None
except Exception as exc:
    Calibrator = None
    estimate_severity = None
    ShiftDetector = None
    combine_decision = None
    _INTELLIGENCE_IMPORT_ERROR = exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_model_available():
    return inspect_image is not None and CHECKPOINT_PATH is not None and CHECKPOINT_PATH.is_file()


def is_intelligence_available():
    return _INTELLIGENCE_IMPORT_ERROR is None


def get_supported_categories():
    return ["Bottle"]


def _sample_paths(paths, count):
    if len(paths) <= count:
        return paths
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[index] for index in indices]


@st.cache_resource(show_spinner=False)
def _load_intelligence_references(checkpoint_mtime_ns, calibration_mtime_ns):
    if not is_model_available():
        raise RuntimeError("The PatchCore checkpoint or model service is unavailable.")
    if not is_intelligence_available():
        raise RuntimeError(f"Intelligence modules are unavailable: {_INTELLIGENCE_IMPORT_ERROR}")

    calibration_path = OUTPUT_FOLDER / "batch_test" / "calibration.json"
    if CHECKPOINT_PATH.stat().st_mtime_ns != checkpoint_mtime_ns:
        raise RuntimeError("The PatchCore checkpoint changed during reference loading. Retry the inspection.")
    current_calibration_mtime = calibration_path.stat().st_mtime_ns if calibration_path.is_file() else None
    if current_calibration_mtime != calibration_mtime_ns:
        raise RuntimeError("The held-out normal calibration file changed during loading. Retry the inspection.")

    if calibration_path.is_file():
        with calibration_path.open("r", encoding="utf-8") as file:
            calibration_data = json.load(file)
        if calibration_data.get("checkpoint_mtime_ns") != checkpoint_mtime_ns:
            raise RuntimeError("Calibration scores do not match the current PatchCore checkpoint.")
        reference_scores = calibration_data["normal_scores"]
        pixel_threshold = float(calibration_data["pixel_threshold"])
    else:
        normal_folder = ROOT / "dataset" / "bottle" / "test" / "good"
        reference_paths = sorted(
            path for path in normal_folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if len(reference_paths) < 5:
            raise RuntimeError(f"At least five held-out normal images are required in {normal_folder}.")

        reference_scores = []
        pixel_samples = []
        rng = np.random.default_rng(4200)
        for path in reference_paths:
            prediction = inspect_image(path, output_folder=OUTPUT_FOLDER / "calibration_reference")
            reference_scores.append(float(prediction["anomaly_score"]))
            anomaly_map = np.asarray(prediction["anomaly_map"], dtype=np.float32).ravel()
            sample_count = min(25000, anomaly_map.size)
            indices = rng.choice(anomaly_map.size, sample_count, replace=False)
            pixel_samples.append(anomaly_map[indices])
        pixel_threshold = max(float(np.quantile(np.concatenate(pixel_samples), 0.999)), 1e-6)
        calibration_data = {
            "normal_scores": reference_scores,
            "sample_count": len(reference_scores),
            "source": "dataset/bottle/test/good; held out from PatchCore training",
            "reference_paths": [str(path) for path in reference_paths],
            "checkpoint_mtime_ns": checkpoint_mtime_ns,
            "pixel_threshold": pixel_threshold,
            "pixel_reference_quantile": 0.999,
        }
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        with calibration_path.open("w", encoding="utf-8") as file:
            json.dump(calibration_data, file, indent=2, allow_nan=False)

    if len(reference_scores) < 5:
        raise RuntimeError("At least five normal calibration scores are required.")
    calibrator = Calibrator().fit(reference_scores)

    shift_images = []
    train_folder = ROOT / "dataset" / "bottle" / "train" / "good"
    train_paths = sorted(
        path for path in train_folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    for path in _sample_paths(train_paths, 20):
        with Image.open(path) as reference_image:
            shift_images.append(np.asarray(reference_image.convert("RGB")))
    shift_detector = ShiftDetector().fit(shift_images)

    return calibrator, shift_detector, len(reference_scores), pixel_threshold

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PROMETHEUS-X | Industrial Anomaly Detection",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject CSS ───────────────────────────────────────────────────────────────
st.markdown(get_main_css(), unsafe_allow_html=True)

# ─── Session State Initialisation ────────────────────────────────────────────
def _init_session():
    defaults = {
        "inspection_history": [],       # list of result dicts
        "feedback_log":       [],       # list of feedback dicts
        "last_inspection_id": None,     # string
        "uploaded_image":     None,     # PIL Image
        "uploaded_image_bytes": None,
        "uploaded_image_name": None,
        "last_results":       None,     # dict of last run outputs
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem 0;">
        <div style="font-size:1.6rem;font-weight:800;
                    background:linear-gradient(90deg,#1D4ED8,#1D4ED8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            PROMETHEUS-X
        </div>
        <div style="font-size:0.68rem;color:#475569;letter-spacing:0.15em;
                    text-transform:uppercase;margin-top:2px;">
            Industrial AI Platform
        </div>
    </div>
    <hr style="border-color:#CBD5E1;margin:0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    # ── System Status ─────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;'
                'text-transform:uppercase;color:#1D4ED8;margin-bottom:0.5rem;">System Status</div>',
                unsafe_allow_html=True)
    backend_ready = (
        is_intelligence_available()
    )
    model_status_badge(is_model_available(), backend_ready)

    st.markdown("<hr style='border-color:#CBD5E1;'>", unsafe_allow_html=True)

    # ── Category Selection ────────────────────────────────────────────────────
    st.markdown('<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;'
                'text-transform:uppercase;color:#1D4ED8;margin-bottom:0.5rem;">Product Category</div>',
                unsafe_allow_html=True)

    supported = get_supported_categories()

    category = st.selectbox(
        "Select product type",
        options=supported,
        index=0,
        label_visibility="collapsed",
    )
    if category not in supported:
        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
                    border-radius:8px;padding:0.6rem 0.8rem;font-size:0.78rem;color:#B45309;">
            Model for <strong>{category}</strong> is not yet available.
            Results cannot be generated for this category.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#CBD5E1;'>", unsafe_allow_html=True)

    # ── Sensitivity Slider ────────────────────────────────────────────────────
    st.markdown('<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;'
                'text-transform:uppercase;color:#1D4ED8;margin-bottom:0.5rem;">Inspection Sensitivity</div>',
                unsafe_allow_html=True)

    sensitivity = st.slider(
        "Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        label_visibility="collapsed",
    )
    sens_note = "Low" if sensitivity < 0.4 else ("High" if sensitivity > 0.7 else "Balanced")
    st.caption(
        f"**{sens_note}** sensitivity ({sensitivity:.2f})  \n"
        "Higher sensitivity may detect more suspicious regions "
        "but can increase false positives. "
        "This adjusts empirical thresholds derived from held-out normal Bottle calibration images."
        " Calibration values are heuristic, not probabilities."
    )

    st.markdown("<hr style='border-color:#CBD5E1;'>", unsafe_allow_html=True)

    # ── History Controls ──────────────────────────────────────────────────────
    st.markdown('<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;'
                'text-transform:uppercase;color:#1D4ED8;margin-bottom:0.5rem;">Session Controls</div>',
                unsafe_allow_html=True)

    if st.button("Clear History", key="clear_history"):
        st.session_state.inspection_history = []
        st.session_state.last_inspection_id = None
        st.session_state.last_results = None
        st.success("History cleared.")

    # Download inspection log
    if st.session_state.inspection_history:
        log_json = json.dumps(st.session_state.inspection_history, indent=2, default=str)
        st.download_button(
            label="Download Inspection Log",
            data=log_json,
            file_name=f"prometheus_x_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    # Download feedback log
    if st.session_state.feedback_log:
        fb_json = json.dumps(st.session_state.feedback_log, indent=2, default=str)
        st.download_button(
            label="Download Feedback Log",
            data=fb_json,
            file_name=f"prometheus_x_feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem;color:#475569;text-align:center;line-height:1.6;">
        PROMETHEUS-X v0.1.0<br>Hackathon Demo — A06<br>
        Results are decision support only.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════════

render_hero()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_inspect, tab_history, tab_feedback = st.tabs(["Inspection", "History", "Feedback"])




# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — INSPECTION
# ══════════════════════════════════════════════════════════════════════════════

with tab_inspect:

    # ── Upload Section ────────────────────────────────────────────────────────
    section_label("Image Upload")

    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("""
        <div class="px-card">
            <div class="px-card-title">Upload Product Image</div>
            <div style="font-size:0.82rem;color:#475569;margin-bottom:0.8rem;line-height:1.5;">
                Upload a JPG, JPEG, or PNG image of the product to inspect.
                Ensure the image is well-lit and shows the product surface clearly.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

        # Validate and load
        image: Optional[Image.Image] = None
        if uploaded_file is not None:
            image = load_image_from_upload(uploaded_file)
            if image is None:
                st.error(
                    "Unable to open the uploaded file as an image. "
                    "Please upload a valid JPG, JPEG, or PNG file."
                )
            else:
                st.session_state.uploaded_image = image
                st.session_state.uploaded_image_bytes = uploaded_file.getvalue()
                st.session_state.uploaded_image_name = uploaded_file.name
                w, h = image.size
                st.markdown(f"""
                <div style="font-size:0.8rem;color:#1D4ED8;margin-top:0.5rem;">
                    ✓ Image loaded — {w} × {h} px
                </div>
                """, unsafe_allow_html=True)
        else:
            # Keep the last uploaded image available across Streamlit reruns.
            if st.session_state.uploaded_image is not None:
                st.info("Using previously uploaded image. Upload a new file to replace it.")
                image = st.session_state.uploaded_image

        # ── Run Inspection Button ─────────────────────────────────────────────
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        run_btn = st.button(
            "Run AI Inspection",
            type="primary",
            width="stretch",
            disabled=(image is None),
        )

    with col_preview:
        if image is not None:
            st.markdown('<div class="px-card">', unsafe_allow_html=True)
            render_uploaded_image(image, label="Product Image Preview")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="px-card" style="min-height:220px;display:flex;
                        flex-direction:column;align-items:center;justify-content:center;
                        text-align:center;">
                <div style="font-size:2rem;margin-bottom:0.5rem;opacity:0.3;">⬡</div>
                <div style="color:#475569;font-size:0.82rem;">
                    No image uploaded yet.<br>Upload a product image to begin inspection.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#CBD5E1;'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  INSPECTION EXECUTION
    # ══════════════════════════════════════════════════════════════════════════

    if run_btn and image is not None:
        inspection_id = generate_inspection_id()
        timestamp = current_timestamp()
        result = None
        anomaly_map = None
        calibration_result = None
        severity_result = None
        shift_result = None
        decision_result = None
        mask_array = None
        mask_path = None
        reference_count = 0
        calibration_source = "Unavailable"
        calibration_error = None

        if not is_model_available():
            message = str(_MODEL_IMPORT_ERROR) if _MODEL_IMPORT_ERROR else f"Checkpoint unavailable: {CHECKPOINT_PATH}"
            st.error(f"PatchCore is unavailable. {message}")
        else:
            with st.spinner("Running PatchCore and preparing normal-reference intelligence…"):
                try:
                    upload_folder = OUTPUT_FOLDER / "uploads"
                    upload_folder.mkdir(parents=True, exist_ok=True)
                    suffix = Path(st.session_state.uploaded_image_name or "upload.png").suffix or ".png"
                    image_path = upload_folder / f"{inspection_id}{suffix}"
                    try:
                        image_path.write_bytes(st.session_state.uploaded_image_bytes)
                        result = inspect_image(image_path)
                    finally:
                        image_path.unlink(missing_ok=True)
                    score = float(result["anomaly_score"])
                    anomaly_map = np.asarray(result["anomaly_map"], dtype=np.float32)

                    if is_intelligence_available():
                        try:
                            calibration_path = OUTPUT_FOLDER / "batch_test" / "calibration.json"
                            calibration_mtime_ns = calibration_path.stat().st_mtime_ns if calibration_path.is_file() else None
                            calibrator, shift_detector, reference_count, pixel_threshold = _load_intelligence_references(
                                CHECKPOINT_PATH.stat().st_mtime_ns,
                                calibration_mtime_ns,
                            )
                            calibration_result = calibrator.predict(score, sensitivity=sensitivity)
                            shift_result = shift_detector.assess(np.asarray(image.convert("RGB")))
                            decision_result = combine_decision(
                                calibration_result.verdict,
                                shift_result.shifted,
                                shift_result.reliability,
                                shift_result.quality_issues,
                            )
                            severity_result = estimate_severity(
                                anomaly_map,
                                pixel_threshold,
                            )
                            mask_array = severity_result.mask.astype(np.uint8) * 255
                            mask_folder = OUTPUT_FOLDER / "inspection_masks"
                            mask_folder.mkdir(parents=True, exist_ok=True)
                            mask_path = mask_folder / f"{inspection_id}_mask.png"
                            Image.fromarray(mask_array).save(mask_path)
                            calibration_source = (
                                "dataset/bottle/test/good; held out from PatchCore training"
                            )
                        except Exception as exc:
                            calibration_error = str(exc)
                            logger.exception("Intelligence assessment failed")
                    else:
                        calibration_error = str(_INTELLIGENCE_IMPORT_ERROR)

                except Exception as exc:
                    st.error(f"PatchCore inference failed: {exc}")
                    logger.exception("PatchCore inference failed")

        if result is not None:
            model_prediction = "Anomaly Detected" if result["is_anomaly"] else "Normal"
            if decision_result is not None:
                verdict = {
                    "NORMAL": "Normal",
                    "DEFECT": "Anomaly Detected",
                    "UNCERTAIN": "Uncertain",
                    "MANUAL REVIEW REQUIRED": "Manual Review Required",
                }[decision_result.action]
                confidence = calibration_result.confidence
            else:
                verdict = "Manual Review Required"
                confidence = None

            severity = severity_label(severity_result.level) if severity_result else "Unavailable"
            shift_detected = shift_result.shifted if shift_result else False
            reliability = reliability_label(shift_result.reliability) if shift_result else "Low"

            record = {
                "id": inspection_id,
                "timestamp": timestamp,
                "category": category,
                "score": result["anomaly_score"],
                "verdict": verdict,
                "severity": severity,
                "confidence": confidence,
                "reliability": reliability,
                "shift": shift_detected,
                "decision": decision_result.action if decision_result else "Unavailable",
                "decision_rationale": decision_result.rationale if decision_result else [],
                "calibration_sample_count": reference_count,
                "heatmap_path": result["heatmap_path"],
                "overlay_path": result["overlay_path"],
                "mask_path": str(mask_path) if mask_path else None,
            }
            st.session_state.inspection_history.append(record)
            st.session_state.last_inspection_id = inspection_id
            st.session_state.last_results = record

            section_label("Inspection Results")
            if calibration_error:
                st.warning(f"Calibration or intelligence reference unavailable: {calibration_error}")
                st.warning("The raw model prediction is shown for context only; automated disposition is disabled and manual review is required.")
            st.caption(
                f"PatchCore binary prediction: **{model_prediction}**. "
                "This model output indicates anomaly evidence, not confirmation of a physical defect."
            )

            res_col1, res_col2, res_col3, res_col4 = st.columns(4, gap="medium")
            with res_col1:
                render_uploaded_image(image, label="Original Image")
            with res_col2:
                render_heatmap(result["heatmap_path"])
            with res_col3:
                st.markdown('<div class="px-card-title">Anomaly Overlay</div>', unsafe_allow_html=True)
                st.image(result["overlay_path"], width="stretch")
            with res_col4:
                render_mask(mask_array)

            render_verdict(verdict)
            render_metrics(
                result["anomaly_score"],
                severity,
                confidence,
                reliability,
            )

            if calibration_result is not None:
                st.info(
                    f"Calibration: {calibration_result.verdict}; "
                    f"normal-reference percentile {calibration_result.normality_percentile:.1f}%; "
                    f"reference images: {reference_count}. Source: {calibration_source}. "
                    "This empirical percentile is not a probability guarantee."
                )

            if severity_result is not None:
                severity_columns = st.columns(5)
                severity_columns[0].metric("Affected area", f"{severity_result.area_pct:.2f}%")
                severity_columns[1].metric("Defect regions", severity_result.num_regions)
                severity_columns[2].metric("Severity index (heuristic)", f"{severity_result.index:.3f}")
                severity_columns[3].metric("Peak anomaly", f"{severity_result.peak_score:.4f}")
                severity_columns[4].metric("Mean region anomaly", f"{severity_result.mean_defect_score:.4f}")
                if severity_result.bboxes:
                    st.caption(f"Defect regions (x_min, y_min, x_max, y_max): {severity_result.bboxes}")
                    st.caption(
                        f"Region sizes (% of image): {severity_result.region_sizes_pct}; "
                        f"centroids (x, y): {severity_result.region_centroids}; "
                        f"spatial extent: {severity_result.spatial_extent_pct:.2f}%"
                    )

            if shift_result is not None:
                render_shift_warning(shift_result.shifted)
                st.info(
                    f"Image quality/distribution shift: {'detected' if shift_result.shifted else 'not detected'}. "
                    f"Reliability: {shift_result.reliability} "
                    f"(heuristic index {shift_result.reliability_score:.2f}). "
                    f"Reasons: {', '.join(shift_result.reasons) if shift_result.reasons else 'none'}"
                )
                if shift_result.quality_issues:
                    st.warning(f"Image-quality flags: {', '.join(shift_result.quality_issues)}")
                st.caption(shift_result.recommendation)

            if decision_result is not None:
                st.write(f"**Decision rationale:** {'; '.join(decision_result.rationale)}")
                recommended = decision_result.note
            else:
                recommended = "Held-out calibration or reliability references are unavailable. Review the image and localization manually before disposition."
            render_recommendation(recommended)
            inspection_report = {
                "inspection_id": inspection_id,
                "timestamp": timestamp,
                "category": category,
                "patchcore": {
                    "image_name": result["image_name"],
                    "anomaly_score": result["anomaly_score"],
                    "binary_prediction": model_prediction,
                },
                "calibration": calibration_result.to_dict() if calibration_result else None,
                "decision": decision_result.to_dict() if decision_result else {
                    "action": "MANUAL REVIEW REQUIRED",
                    "rationale": ["Held-out calibration or reliability references were unavailable"],
                },
                "severity": severity_result.to_dict() if severity_result else None,
                "shift": shift_result.to_dict() if shift_result else None,
                "artifacts": {
                    "heatmap": result["heatmap_path"],
                    "overlay": result["overlay_path"],
                    "binary_mask": str(mask_path) if mask_path else None,
                },
                "limitation": "Scores and heuristic signals are decision support, not probabilities or physical-defect confirmation.",
            }
            st.download_button(
                "Download inspection report (JSON)",
                data=json.dumps(inspection_report, indent=2, default=str),
                file_name=f"{inspection_id}_inspection_report.json",
                mime="application/json",
            )
            st.markdown(
                f'<div style="font-size:0.72rem;color:#475569;text-align:right;margin-top:0.5rem;">'
                f'Inspection ID: <span style="font-family:monospace;color:#475569;">{inspection_id}</span>'
                f' &nbsp;|&nbsp; {timestamp}</div>',
                unsafe_allow_html=True,
            )

    elif image is None and not run_btn:
        # Introductory instructions
        section_label("Getting Started")
        st.markdown("""
        <div class="px-card">
            <div class="px-card-title">How to Inspect a Product</div>
            <ol style="color:#475569;font-size:0.88rem;line-height:2;">
                <li>Select the <strong style="color:#0F172A;">product category</strong> from the sidebar.</li>
                <li>Adjust the <strong style="color:#0F172A;">inspection sensitivity</strong> slider as needed.</li>
                <li>Upload a <strong style="color:#0F172A;">JPG, JPEG, or PNG</strong> image of the product.</li>
                <li>Click <strong style="color:#1D4ED8;">Run AI Inspection</strong>.</li>
                <li>Review the anomaly heatmap, verdict, and metrics in the results panel.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-banner">
            PatchCore and shift references use Bottle training normals. Score calibration uses held-out normal Bottle test images.
            Batch evaluation applies leave-one-out calibration to each normal test image; training results are never independent test results.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    section_label("Inspection History")
    st.markdown("""
    <div class="info-banner">
        History is session-based and cleared when you close or refresh the browser tab.
        Download the log before closing if you need to retain records.
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.inspection_history
    render_history_table(history)

    if history:
        col_dl, col_clr = st.columns([2, 1])
        with col_dl:
            log_json = json.dumps(history, indent=2, default=str)
            st.download_button(
                label="Download as JSON",
                data=log_json,
                file_name=f"prometheus_x_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        with col_clr:
            if st.button("Clear All History", key="clear_hist_tab"):
                st.session_state.inspection_history = []
                st.session_state.last_inspection_id = None
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

with tab_feedback:
    feedback = render_feedback_form(st.session_state.last_inspection_id)
    if feedback:
        st.session_state.feedback_log.append(feedback)
        st.success(
            f"Feedback recorded for Inspection ID **{feedback['inspection_id']}**. "
            "Thank you — this helps validate model performance."
        )
        st.markdown("""
        <div class="info-banner">
            Feedback is stored in session memory only. Use <em>Download Feedback Log</em>
            in the sidebar to export it. Feedback does <strong>not</strong> automatically
            retrain the model — it is intended for offline model validation and review.
        </div>
        """, unsafe_allow_html=True)
