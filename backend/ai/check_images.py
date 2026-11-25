from pathlib import Path
from PIL import Image

dataset_path = Path("dataset")

bad_images = []

for file in dataset_path.rglob("*"):
    if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        try:
            with Image.open(file) as image:
                image.verify()
        except Exception as error:
            bad_images.append(file)
            print("Damaged image:", file)
            print("Error:", error)

print("\nTotal damaged images:", len(bad_images))