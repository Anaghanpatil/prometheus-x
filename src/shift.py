
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

from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MIN_TRAIN_IMAGES = 5
MIN_CAUTION_Z = 3.0
STATISTIC_SCALE_FLOORS = {
    "brightness": 2.0,
    "contrast": 2.0,
    "sharpness": 1.0,
    "noise": 0.5,
    "colour_cast": 2.0,
}
MIN_IMAGE_SIDE = 128
MIN_BRIGHTNESS = 12.0
MAX_BRIGHTNESS = 245.0
MIN_CONTRAST = 4.0
MIN_SHARPNESS = 0.1
MAX_NOISE = 60.0

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
    statistic_z_scores: dict = field(default_factory=dict)
    quality_issues: list = field(default_factory=list)

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

    def __init__(self, caution_z=MIN_CAUTION_Z, scale_floors=None):
        if not np.isfinite(caution_z) or caution_z <= 0:
            raise ValueError("caution_z must be a positive finite number.")
        self.fitted = False
        self._caution_z_setting = float(caution_z)
        self.scale_floors = dict(STATISTIC_SCALE_FLOORS)
        if scale_floors:
            self.scale_floors.update(scale_floors)

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
        robust_scale = self.reference_mad * 1.4826
        floors = np.array([self.scale_floors[key] for key in keys])
        self.reference_mad = np.maximum(robust_scale, floors)

        self.caution_z = self._caution_z_setting

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
        image_array = np.asarray(image)
        quality_issues = []

        if min(image_array.shape[:2]) < MIN_IMAGE_SIDE:
            quality_issues.append(f"image_side_below_{MIN_IMAGE_SIDE}px")
        if current["brightness"] < MIN_BRIGHTNESS:
            quality_issues.append("severely_underexposed")
        elif current["brightness"] > MAX_BRIGHTNESS:
            quality_issues.append("severely_overexposed")
        if current["contrast"] < MIN_CONTRAST:
            quality_issues.append("extremely_low_contrast")
        if current["sharpness"] < MIN_SHARPNESS:
            quality_issues.append("extreme_blur_or_focus_failure")
        if current["noise"] > MAX_NOISE:
            quality_issues.append("extreme_image_noise")

        z_scores = {}

        for index, key in enumerate(self.keys):
            difference = abs(
                current[key] - self.reference_median[index]
            )
            z_scores[key] = float(
                difference / self.reference_mad[index]
            )

        max_z = max(z_scores.values())

        # Keep the individual deviations for transparent reporting.
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

        # A single moderate statistical deviation is treated as caution,
        # not automatically as a major shift. Strong deviations,
        # multiple simultaneous deviations, feature shift, or actual
        # image-quality failures still trigger the shift gate.
        statistical_reasons = [
            key for key in reasons if key != "patchcore_features"
        ]

        strong_statistical_shift = any(
            value >= self.caution_z * LOW_SHIFT_RATIO
            for value in z_scores.values()
        )

        multiple_statistical_shifts = len(statistical_reasons) >= 2
        feature_shift = "patchcore_features" in reasons

        shifted = bool(
            strong_statistical_shift
            or multiple_statistical_shifts
            or feature_shift
            or quality_issues
        )

        # Reliability is a heuristic, not a calibrated probability.
        # One moderate deviation can lower reliability to Medium,
        # but does not by itself force manual review.
        if max_z < self.caution_z:
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

        if quality_issues:
            recommendation = (
                "Image quality is outside the configured operating limits. "
                "Retake or correct the image before automated disposition."
            )

        elif shifted:
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
            recommendation=recommendation,
            statistic_z_scores=z_scores,
            quality_issues=quality_issues,
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
        self.keys = None

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
