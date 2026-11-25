from pathlib import Path

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


def main():
    # Find the PROMETHEUS-X project folder
    project_root = Path(__file__).resolve().parents[2]

    # Dataset location
    bottle_path = project_root / "dataset" / "bottle"
    normal_images_path = bottle_path / "train" / "good"

    # Check that the normal training images exist
    if not normal_images_path.exists():
        raise FileNotFoundError(
            f"Normal training folder not found: {normal_images_path}"
        )

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_count = sum(
        1 for file in normal_images_path.iterdir()
        if file.is_file() and file.suffix.lower() in image_extensions
    )

    if image_count == 0:
        raise RuntimeError(
            f"No images found in: {normal_images_path}"
        )

    print(f"Found {image_count} normal bottle images.")
    print("Preparing dataset...")

    # Use only normal images for PatchCore training.
    # Synthetic validation avoids the Kaggle mask-path issue.
    datamodule = Folder(
        name="prometheus_bottle",
        root=str(bottle_path),
        normal_dir="train/good",
        test_split_mode="synthetic",
        val_split_mode="synthetic",
        num_workers=0,
    )

    # Create PatchCore anomaly detection model
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    # Training engine
    results_path = project_root / "results"

    engine = Engine(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        default_root_dir=str(results_path),
    )

    print("Starting PatchCore training...")
    print("Please wait. Training may take some time.")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    # Save the complete trained checkpoint
    checkpoint_path = (
        project_root / "backend" / "ai" / "patchcore_model.ckpt"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    engine.trainer.save_checkpoint(
        str(checkpoint_path)
    )

    print("\nPatchCore training completed!")
    print("Checkpoint saved at:")
    print(checkpoint_path)


if __name__ == "__main__":
    main()