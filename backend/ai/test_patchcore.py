from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def main():
    # Project root
    project_root = Path(__file__).resolve().parents[2]

    # Trained model
    checkpoint_path = (
        project_root / "backend" / "ai" / "patchcore_model.ckpt"
    )

    # Defective test image
    image_path = (
        project_root
        / "dataset"
        / "bottle"
        / "test"
        / "broken_large"
        / "000.png"
    )

    # Output folder
    output_folder = project_root / "backend" / "ai" / "outputs"
    output_folder.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        print("ERROR: Checkpoint not found:", checkpoint_path)
        return

    if not image_path.exists():
        print("ERROR: Test image not found:", image_path)
        return

    print("Loading PatchCore model...")

    model = Patchcore()
    engine = Engine()

    dataset = PredictDataset(
        path=str(image_path),
        image_size=(256, 256),
    )

    print("Inspecting image...")

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(checkpoint_path),
    )

    if not predictions:
        print("ERROR: No prediction returned.")
        return

    prediction = predictions[0]

    # Read original image
    original_image = Image.open(image_path).convert("RGB")
    original_array = np.array(original_image)

    # Convert anomaly map to NumPy
    anomaly_map = prediction.anomaly_map
    anomaly_map = torch.as_tensor(anomaly_map)
    anomaly_map = anomaly_map.squeeze().detach().cpu().numpy()

    # Resize anomaly map to original image dimensions
    heatmap_image = Image.fromarray(
        (anomaly_map * 255).clip(0, 255).astype(np.uint8)
    )
    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR,
    )
    heatmap_array = np.array(heatmap_image)

    # Save heatmap
    heatmap_path = output_folder / "bottle_000_heatmap.png"

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_array, cmap="jet")
    plt.colorbar(label="Anomaly intensity")
    plt.title("PatchCore Defect Heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    # Save heatmap overlay on original image
    overlay_path = output_folder / "bottle_000_overlay.png"

    plt.figure(figsize=(8, 6))
    plt.imshow(original_array)
    plt.imshow(heatmap_array, cmap="jet", alpha=0.45)
    plt.title("AI Defect Localization")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200)
    plt.close()

    print("\n========== AI INSPECTION ==========")
    print("Anomaly score:", prediction.pred_score)
    print("Anomaly label:", prediction.pred_label)
    print("Heatmap saved:", heatmap_path)
    print("Overlay saved:", overlay_path)
    print("=================================")


if __name__ == "__main__":
    main()