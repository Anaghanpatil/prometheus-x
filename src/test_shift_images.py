
from pathlib import Path
import numpy as np
from PIL import Image

from src.shift import ShiftDetector


# 1. Locate the normal bottle images
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_FOLDER = PROJECT_ROOT / "data" / "shift_test" / "normal"

# 2. Supported image formats
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = sorted(
    path for path in IMAGE_FOLDER.iterdir()
    if path.suffix.lower() in extensions
)

if len(image_paths) < 6:
    raise ValueError("Please provide at least 6 normal images.")

# 3. Load images
images = []

for path in image_paths:
    with Image.open(path) as img:
        images.append(np.array(img.convert("RGB")))

print("Images loaded:", len(images))

# 4. Use all except the last image as reference
reference_images = images[:-1]
test_image = images[-1]

detector = ShiftDetector()
detector.fit(reference_images)

print("\nReference distribution fitted.")
print("Reference images:", len(reference_images))

# 5. Test an unseen normal image
normal_result = detector.assess(test_image)

print("\nNORMAL IMAGE TEST")
print("Shift detected:", normal_result.shifted)
print("Reliability:", normal_result.reliability)
print("Reasons:", normal_result.reasons)

# 6. Create an artificially brightness-shifted image
bright_image = np.clip(
    test_image.astype(np.float32) + 60,
    0,
    255
).astype(np.uint8)

shift_result = detector.assess(bright_image)

print("\nBRIGHTNESS SHIFT TEST")
print("Shift detected:", shift_result.shifted)
print("Reliability:", shift_result.reliability)
print("Reasons:", shift_result.reasons)

print("\nReal image shift testing completed.")
