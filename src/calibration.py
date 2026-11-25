
"""
calibration.py - PROMETHEUS-X score calibration

Interprets raw image-level anomaly scores by comparing them
against scores from normal validation images.

Features:
    - Normality percentile
    - Normal / Uncertain / Defect verdict
    - Sensitivity adjustment
    - Human review flag
    - Few-shot recalibration for camera/lighting changes
    - Save and load calibration reference

Dependency: numpy
"""

from dataclasses import dataclass, asdict
from typing import Sequence
import json

import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEFAULT_SENSITIVITY = 0.5
MIN_NORMAL_SCORES = 5
EPS = 1e-12


# ---------------------------------------------------------
# Result structure
# ---------------------------------------------------------

@dataclass
class CalibrationResult:

    score: float
    verdict: str
    normality_percentile: float
    confidence: float
    lower_threshold: float
    upper_threshold: float
    needs_review: bool

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------

def _clean_scores(scores, name="scores"):

    arr = np.asarray(scores, dtype=np.float64).ravel()

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    if arr.size < MIN_NORMAL_SCORES:
        raise ValueError(
            f"At least {MIN_NORMAL_SCORES} normal scores are required."
        )

    return arr


# ---------------------------------------------------------
# Calibrator
# ---------------------------------------------------------

class Calibrator:

    def __init__(self):

        self._base = None
        self._reference = None
        self.recalibrated = False
        self.history = []

    # -----------------------------------------------------
    # Fit normal validation scores
    # -----------------------------------------------------

    def fit(self, normal_scores: Sequence[float]):

        scores = _clean_scores(
            normal_scores,
            "normal_scores"
        )

        self._base = np.sort(scores)
        self._reference = self._base.copy()

        self.recalibrated = False
        self.history = []

        return self

    # -----------------------------------------------------
    # Check calibration status
    # -----------------------------------------------------

    def _check_fitted(self):

        if self._reference is None:
            raise RuntimeError(
                "Calibrator is not fitted. Call fit() first."
            )

    # -----------------------------------------------------
    # Sensitivity thresholds
    # -----------------------------------------------------

    @staticmethod
    def _quantiles_for(sensitivity):

        if not isinstance(sensitivity, (int, float)):
            raise TypeError("Sensitivity must be a number.")

        if not np.isfinite(sensitivity):
            raise ValueError("Sensitivity must be finite.")

        if not 0 <= sensitivity <= 1:
            raise ValueError(
                "Sensitivity must be between 0 and 1."
            )

        # Higher sensitivity lowers the defect threshold.
        upper_quantile = float(
            np.interp(
                sensitivity,
                [0.0, 0.5, 1.0],
                [0.999, 0.99, 0.95]
            )
        )

        # Keep an uncertain region below the defect threshold.
        lower_quantile = max(upper_quantile - 0.04, 0.5)

        return lower_quantile, upper_quantile

    # -----------------------------------------------------
    # Calculate thresholds
    # -----------------------------------------------------

    def thresholds(self, sensitivity=DEFAULT_SENSITIVITY):

        self._check_fitted()

        q_lower, q_upper = self._quantiles_for(sensitivity)

        lower = float(
            np.quantile(self._reference, q_lower)
        )

        upper = float(
            np.quantile(self._reference, q_upper)
        )

        # Handle nearly identical scores.
        if upper <= lower:

            spread = max(abs(upper) * 0.01, 1e-6)
            upper = lower + spread

        return lower, upper

    # -----------------------------------------------------
    # Interpret anomaly score
    # -----------------------------------------------------

    def predict(self, score, sensitivity=DEFAULT_SENSITIVITY):

        self._check_fitted()

        score = float(score)

        if not np.isfinite(score):
            raise ValueError("Anomaly score must be finite.")

        lower, upper = self.thresholds(sensitivity)

        # Percentage of normal reference scores <= new score.
        percentile = (
            100.0
            * np.searchsorted(
                self._reference,
                score,
                side="right"
            )
            / self._reference.size
        )

        # Determine verdict.
        if score < lower:

            verdict = "Normal"
            confidence = 1.0 - percentile / 100.0

        elif score < upper:

            verdict = "Uncertain"

            # Keep uncertainty explicit instead of presenting
            # a misleading probability of correctness.
            confidence = 0.0

        else:

            verdict = "Defect"
            confidence = percentile / 100.0

        return CalibrationResult(

            score=round(score, 6),

            verdict=verdict,

            normality_percentile=round(
                float(percentile), 2
            ),

            confidence=round(
                float(np.clip(confidence, 0, 1)), 4
            ),

            lower_threshold=round(lower, 6),

            upper_threshold=round(upper, 6),

            needs_review=(verdict == "Uncertain")
        )

    # -----------------------------------------------------
    # Recalibration for new lighting/camera conditions
    # -----------------------------------------------------

    def recalibrate(self, new_normal_scores):

        self._check_fitted()

        new = _clean_scores(
            new_normal_scores,
            "new_normal_scores"
        )

        # Calculate original distribution statistics.
        base_median = float(np.median(self._base))

        base_mad = float(
            np.median(
                np.abs(self._base - base_median)
            ) * 1.4826
        )

        # Calculate new normal distribution statistics.
        new_median = float(np.median(new))

        new_mad = float(
            np.median(
                np.abs(new - new_median)
            ) * 1.4826
        )

        base_mad = max(base_mad, EPS)

        # Prevent unstable scaling.
        new_mad = float(
            np.clip(
                new_mad,
                0.25 * base_mad,
                8.0 * base_mad
            )
        )

        scale = new_mad / base_mad

        offset = new_median - scale * base_median

        # Adapt from original reference, not previous adaptation.
        self._reference = np.sort(
            scale * self._base + offset
        )

        self.recalibrated = True

        info = {
            "number_of_new_normal_samples": int(new.size),
            "scale": round(scale, 4),
            "offset": round(offset, 6),
            "old_median": round(base_median, 6),
            "new_median": round(new_median, 6)
        }

        self.history.append(info)

        return info

    # -----------------------------------------------------
    # Reset recalibration
    # -----------------------------------------------------

    def reset(self):

        self._check_fitted()

        self._reference = self._base.copy()

        self.recalibrated = False

        self.history = []

    # -----------------------------------------------------
    # Save calibration reference
    # -----------------------------------------------------

    def save(self, path):

        self._check_fitted()

        data = {
            "normal_scores": self._base.tolist()
        }

        with open(path, "w", encoding="utf-8") as file:

            json.dump(data, file, indent=4)

    # -----------------------------------------------------
    # Load calibration reference
    # -----------------------------------------------------

    @classmethod
    def load(cls, path):

        with open(path, "r", encoding="utf-8") as file:

            data = json.load(file)

        return cls().fit(data["normal_scores"])


# ---------------------------------------------------------
# Self-test
# Run: python src/calibration.py
# ---------------------------------------------------------

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    # Simulated normal validation scores.
    normal_scores = rng.gamma(
        shape=9.0,
        scale=0.05,
        size=300
    )

    calibrator = Calibrator().fit(normal_scores)

    print("\nPROMETHEUS-X CALIBRATION TEST")
    print("-" * 45)

    lower, upper = calibrator.thresholds()

    print("Lower threshold:", round(lower, 4))
    print("Upper threshold:", round(upper, 4))

    # Test different anomaly scores.
    test_scores = [0.30, 0.45, 0.60, 0.80, 1.20, 2.00]

    print("\nAnomaly score predictions:")

    for score in test_scores:

        result = calibrator.predict(score)

        print(
            f"Score: {score:.2f} | "
            f"Verdict: {result.verdict:9s} | "
            f"Normality percentile: "
            f"{result.normality_percentile:5.1f}% | "
            f"Review: {result.needs_review}"
        )

    # Test sensitivity.
    print("\nSensitivity test:")

    for sensitivity in [0.0, 0.5, 1.0]:

        lower, upper = calibrator.thresholds(sensitivity)

        print(
            f"Sensitivity: {sensitivity:.1f} | "
            f"Upper threshold: {upper:.4f}"
        )

    # Test recalibration.
    shifted_scores = normal_scores * 1.5 + 0.25

    recalibration_info = calibrator.recalibrate(
        shifted_scores[:15]
    )

    print("\nRecalibration test:")
    print(recalibration_info)

    print(
        "Recalibrated:",
        calibrator.recalibrated
    )

    # Test reset.
    calibrator.reset()

    print(
        "Reset successful:",
        not calibrator.recalibrated
    )

    print("\nCalibration tests completed.")
