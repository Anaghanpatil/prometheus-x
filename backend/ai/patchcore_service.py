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


def inspect_image(image_path):
    """
    Inspect an uploaded image using the trained PatchCore model.

    Returns:
        Dictionary containing anomaly score, label,
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

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load model and prediction engine
    model = Patchcore()
    engine = Engine()

    # Prepare uploaded image
    dataset = PredictDataset(
        path=str(image_path),
        image_size=(256, 256),
    )

    # Run AI prediction
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT_PATH),
    )

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
    ).squeeze().detach().cpu().numpy()

    # Convert anomaly map into an image
    heatmap_image = Image.fromarray(
        (anomaly_map * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )

    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR,
    )

    heatmap_array = np.array(heatmap_image)

    # Output filenames
    image_name = image_path.stem

    heatmap_path = (
        OUTPUT_FOLDER / f"{image_name}_heatmap.png"
    )

    overlay_path = (
        OUTPUT_FOLDER / f"{image_name}_overlay.png"
    )

    # Save heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_array, cmap="jet")
    plt.colorbar(label="Anomaly intensity")
    plt.title("PatchCore Defect Heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

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
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
    }