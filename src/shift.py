
"""
PROMETHEUS-X: Distribution Shift Detection

Detects changes in image acquisition conditions:
- Brightness
- Contrast
- Sharpness / blur
- Noise
- Colour cast

Optional:
- PatchCore feature-based distribution shift detection.

Important:
This module detects changes in input distribution.
It does NOT determine whether a product has a defect.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MIN_TRAIN_IMAGES = 5
MIN_SIGMA = 1e-6
MIN_CAUTION_Z = 3.0

MEDIUM_SHIFT_RATIO = 1.0
LOW_SHIFT_RATIO = 2.0


# --------------------------------------------------
# RESULT STRUCTURE
# --------------------------------------------------

@dataclass
class ShiftResult:
    shifted: bool
    reliability: str
    reliability_score: float
    max_z_score: float
    reasons: list
    statistics: dict
    feature_distance: Optional[float]
    recommendation: str

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def _to_float255(image):
    """
    Convert an image to float32 in the range 0-255.

    Supports:
    - Grayscale: H x W
    - RGB: H x W x 3
    - RGBA: H x W x 4
    - CHW: 3 x H x W

    Input colour order must be RGB, not BGR.
    """

    image = np.asarray(image)

    if image.size == 0:
        raise ValueError("Image cannot be empty.")

    if image.ndim == 3:
        if image.shape[-1] in (3, 4):
            image = image[..., :3]
        elif image.shape[0] in (3, 4):
            image = np.transpose(image[:3], (1, 2, 0))
        else:
            raise ValueError("Unsupported image shape.")

    elif image.ndim != 2:
        raise ValueError("Expected a 2D or 3D image.")

    image = image.astype(np.float32)
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=255.0,
        neginf=0.0
    )

    if image.max() <= 1.5:
        image = image * 255.0

    return np.clip(image, 0, 255)


# --------------------------------------------------
# IMAGE STATISTICS
# --------------------------------------------------

def image_statistics(image):
    """
    Extract image acquisition statistics.

    These statistics are not defect scores.
    """

    image = _to_float255(image)

    if image.ndim == 3:
        gray = (
            0.299 * image[..., 0]
            + 0.587 * image[..., 1]
            + 0.114 * image[..., 2]
        )

        colour_cast = float(
            np.mean(image[..., 0]) - np.mean(image[..., 2])
        )

    else:
        gray = image
        colour_cast = 0.0

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Gradient-based sharpness approximation
    gy, gx = np.gradient(gray)
    sharpness = float(np.mean(gx ** 2 + gy ** 2))

    # Robust noise estimate using neighbouring pixel differences
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)

    noise = float(
        (np.median(np.abs(dx)) + np.median(np.abs(dy))) / 2.0
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "noise": noise,
        "colour_cast": colour_cast
    }


# --------------------------------------------------
# PATCHCORE FEATURE POOLING
# --------------------------------------------------

def pool_patch_features(feature_map, channel_first=False):
    """
    Convert a PatchCore feature map into one feature vector.

    channel_first=False:
        H x W x C

    channel_first=True:
        C x H x W

    Confirm the actual PatchCore feature-map layout
    with Member 1 before using this function.
    """

    features = np.asarray(feature_map, dtype=np.float32)

    if features.ndim == 3:
        if channel_first:
            features = np.mean(features, axis=(1, 2))
        else:
            features = np.mean(features, axis=(0, 1))

    elif features.ndim == 1:
        pass

    else:
        raise ValueError(
            "Feature map must be 1D or 3D."
        )

    if not np.all(np.isfinite(features)):
        raise ValueError("Features contain invalid values.")

    return features.astype(np.float64)


# --------------------------------------------------
# SHIFT DETECTOR
# --------------------------------------------------

class ShiftDetector:

    def __init__(self):
        self.fitted = False

        self.reference_median = None
        self.reference_mad = None
        self.caution_z = None

        self.feature_mean = None
        self.feature_inv_cov = None
        self.feature_threshold = None

    # ----------------------------------------------
    # FIT REFERENCE DISTRIBUTION
    # ----------------------------------------------

    def fit(self, train_images, train_features=None):
        """
        Fit using normal reference images only.

        Do not train this detector using defective images.
        """

        if len(train_images) < MIN_TRAIN_IMAGES:
            raise ValueError(
                f"At least {MIN_TRAIN_IMAGES} normal images are required."
            )

        stats_list = [
            image_statistics(image)
            for image in train_images
        ]

        keys = list(stats_list[0].keys())

        matrix = np.array([
            [stats[key] for key in keys]
            for stats in stats_list
        ])

        self.keys = keys

        self.reference_median = np.median(matrix, axis=0)

        self.reference_mad = np.median(
            np.abs(matrix - self.reference_median),
            axis=0
        )

        # Convert MAD into robust standard deviation
        self.reference_mad = np.maximum(
            self.reference_mad * 1.4826,
            MIN_SIGMA
        )

        # Conservative threshold for image statistics
        self.caution_z = MIN_CAUTION_Z

        if train_features is not None:
            self._fit_features(train_features)

        self.fitted = True
        return self

    # ----------------------------------------------
    # FIT PATCHCORE FEATURES
    # ----------------------------------------------

    def _fit_features(self, train_features):

        features = np.asarray(
            train_features,
            dtype=np.float64
        )

        if features.ndim != 2 or len(features) < MIN_TRAIN_IMAGES:
            raise ValueError(
                "Features must be a 2D array with at least 5 samples."
            )

        if not np.all(np.isfinite(features)):
            raise ValueError("Invalid feature values.")

        self.feature_mean = np.mean(features, axis=0)

        covariance = np.cov(features, rowvar=False)

        covariance = np.atleast_2d(covariance)

        regularization = 1e-4 * np.eye(
            covariance.shape[0]
        )

        self.feature_inv_cov = np.linalg.pinv(
            covariance + regularization
        )

        distances = self._mahalanobis(features)

        self.feature_threshold = float(
            np.quantile(distances, 0.99)
        )

    # ----------------------------------------------
    # MAHALANOBIS DISTANCE
    # ----------------------------------------------

    def _mahalanobis(self, features):

        difference = features - self.feature_mean

        distance_squared = np.einsum(
            "ij,jk,ik->i",
            difference,
            self.feature_inv_cov,
            difference
        )

        return np.sqrt(
            np.maximum(distance_squared, 0)
        )

    # ----------------------------------------------
    # ASSESS NEW IMAGE
    # ----------------------------------------------

    def assess(self, image, features=None):

        if not self.fitted:
            raise RuntimeError(
                "Call fit() before assess()."
            )

        current = image_statistics(image)

        z_scores = {}

        for index, key in enumerate(self.keys):

            difference = abs(
                current[key] - self.reference_median[index]
            )

            z_scores[key] = float(
                difference / self.reference_mad[index]
            )

        max_z = max(z_scores.values())

        reasons = [
            key
            for key, value in z_scores.items()
            if value >= self.caution_z
        ]

        feature_distance = None

        if features is not None:

            if self.feature_mean is None:
                raise ValueError(
                    "Detector was not fitted with PatchCore features."
                )

            features = np.asarray(
                features,
                dtype=np.float64
            ).reshape(1, -1)

            if features.shape[1] != len(self.feature_mean):
                raise ValueError(
                    "Feature dimension does not match training features."
                )

            if not np.all(np.isfinite(features)):
                raise ValueError("Invalid feature values.")

            feature_distance = float(
                self._mahalanobis(features)[0]
            )

            if feature_distance > self.feature_threshold:
                reasons.append("patchcore_features")

        shifted = len(reasons) > 0

        # Reliability is a heuristic, not a calibrated probability.
        if max_z < self.caution_z * MEDIUM_SHIFT_RATIO:
            reliability = "High"

        elif max_z < self.caution_z * LOW_SHIFT_RATIO:
            reliability = "Medium"

        else:
            reliability = "Low"

        reliability_score = float(
            np.clip(
                1.0 - max_z / (self.caution_z * LOW_SHIFT_RATIO),
                0.0,
                1.0
            )
        )

        if shifted:
            recommendation = (
                "Review image conditions and consider recalibration "
                "using verified normal images."
            )

        else:
            recommendation = (
                "No significant acquisition shift detected. "
                "Continue inspection."
            )

        return ShiftResult(
            shifted=shifted,
            reliability=reliability,
            reliability_score=reliability_score,
            max_z_score=float(max_z),
            reasons=reasons,
            statistics=current,
            feature_distance=feature_distance,
            recommendation=recommendation
        )

    # ----------------------------------------------
    # RESET
    # ----------------------------------------------

    def reset(self):

        self.fitted = False

        self.reference_median = None
        self.reference_mad = None
        self.caution_z = None

        self.feature_mean = None
        self.feature_inv_cov = None
        self.feature_threshold = None

        return True


# --------------------------------------------------
# SELF TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("\nPROMETHEUS-X SHIFT DETECTION TEST")
    print("-" * 45)

    rng = np.random.default_rng(42)

    # Generate five normal reference images
    normal_images = [
        np.clip(
            rng.normal(125, 25, (128, 128)),
            0,
            255
        ).astype(np.uint8)
        for _ in range(20)
    ]

    detector = ShiftDetector()
    detector.fit(normal_images)

    print("Reference distribution fitted successfully.")

    # Normal image
    normal_test = normal_images[0]

    result = detector.assess(normal_test)

    print("\nNormal image:")
    print("Shift detected:", result.shifted)
    print("Reliability:", result.reliability)
    print("Reasons:", result.reasons)

    # Brightness-shifted image
    bright_image = np.clip(
        normal_test.astype(np.float32) + 90,
        0,
        255
    ).astype(np.uint8)

    result = detector.assess(bright_image)

    print("\nBrightness-shifted image:")
    print("Shift detected:", result.shifted)
    print("Reliability:", result.reliability)
    print("Reasons:", result.reasons)

    print("\nShift detection test completed.")
