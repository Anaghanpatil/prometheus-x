from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "backend" / "ai" / "patchcore_model.ckpt"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT / "backend" / "ai" / "outputs"
)

_MODEL = None
_ENGINE = None
_CHECKPOINT_LOADED = False


def _get_predictor():
    """Create one CPU predictor and restore the trained checkpoint once."""
    global _MODEL, _ENGINE

    if _MODEL is None:
        _MODEL = Patchcore()
        _ENGINE = Engine(accelerator="cpu", devices=1, logger=False)

    return _MODEL, _ENGINE


def inspect_image(image_path, output_folder=None):
    """
    Inspect an uploaded image using the trained PatchCore model.

    Returns:
        Dictionary containing image name, anomaly score, label, anomaly map,
        heatmap path, and overlay path.
    """

    image_path = Path(image_path)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    output_folder = Path(output_folder) if output_folder else OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)

    model, engine = _get_predictor()

    # Prepare uploaded image
    dataset = PredictDataset(
        path=str(image_path),
        image_size=(256, 256),
    )

    # Run AI prediction
    global _CHECKPOINT_LOADED
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT_PATH) if not _CHECKPOINT_LOADED else None,
    )
    _CHECKPOINT_LOADED = True

    if not predictions:
        raise RuntimeError("PatchCore returned no prediction.")

    prediction = predictions[0]

    # Extract prediction values
    score = float(
        torch.as_tensor(prediction.pred_score)
        .reshape(-1)[0]
        .item()
    )

    is_anomaly = bool(
        torch.as_tensor(prediction.pred_label)
        .reshape(-1)[0]
        .item()
    )

    # Load original image
    original_image = Image.open(image_path).convert("RGB")
    original_array = np.array(original_image)

    # Convert anomaly map to NumPy
    anomaly_map = torch.as_tensor(
        prediction.anomaly_map
    ).squeeze().detach().cpu().numpy().astype(np.float32)
    anomaly_map = np.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
    if anomaly_map.ndim != 2:
        raise RuntimeError(
            f"Expected a 2-D anomaly map, got shape {anomaly_map.shape}."
        )

    heatmap_image = Image.fromarray(anomaly_map, mode="F")
    heatmap_array = np.asarray(
        heatmap_image.resize(original_image.size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    # Output filenames
    image_name = image_path.stem

    heatmap_path = (
        output_folder / f"{image_name}_heatmap.png"
    )

    overlay_path = (
        output_folder / f"{image_name}_overlay.png"
    )

    # Save heatmap
    plt.imsave(heatmap_path, heatmap_array, cmap="jet")

    # Save overlay
    plt.figure(figsize=(8, 6))
    plt.imshow(original_array)
    plt.imshow(heatmap_array, cmap="jet", alpha=0.45)
    plt.title("AI Defect Localization")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=150)
    plt.close()

    return {
        "image_name": image_path.name,
        "anomaly_score": score,
        "is_anomaly": is_anomaly,
        "anomaly_map": heatmap_array.tolist(),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
    }