
"""Combine calibrated anomaly and reliability signals into an action."""

from dataclasses import asdict, dataclass


@dataclass
class DecisionResult:
    action: str
    manual_review_required: bool
    rationale: list[str]
    note: str

    def to_dict(self):
        return asdict(self)


def combine_decision(
    calibration_verdict: str,
    shifted: bool,
    reliability: str,
    quality_issues=None,
    model_prediction=None,
) -> DecisionResult:
    """
    Prefer calibrated evidence.
    Use PatchCore's binary prediction when calibration is uncertain.
    Shift and reliability signals produce warnings, not automatic rejection.
    """

    issues = list(quality_issues or [])
    rationale = []

    if shifted:
        rationale.append("Distribution shift detected; prediction should be interpreted cautiously.")

    if str(reliability).lower() == "low":
        rationale.append("Reliability is low; prediction confidence may be reduced.")

    if issues:
        rationale.extend(
            f"Image quality warning: {issue}" for issue in issues
        )

    verdict = str(calibration_verdict or "").strip().lower()

    # First preference: calibrated prediction.
    if verdict == "normal":
        action = "NORMAL"
        rationale.append("Anomaly score is within the calibrated normal range.")

    elif verdict == "defect":
        action = "DEFECT"
        rationale.append("Anomaly score exceeds the calibrated defect threshold.")

    # Fallback: use the existing PatchCore binary prediction.
    elif verdict == "uncertain" or verdict in ("", "none", "unavailable"):
        prediction = str(model_prediction or "").strip().lower()

        if prediction in ("defect", "anomaly detected", "anomaly", "true", "1"):
            action = "DEFECT"
        elif prediction in ("normal", "false", "0"):
            action = "NORMAL"
        else:
            raise ValueError(
                "Calibration is uncertain or unavailable and no valid "
                "PatchCore binary prediction was supplied."
            )

        rationale.append(
            "Calibration was uncertain or unavailable; "
            "the existing PatchCore binary prediction was used as fallback."
        )

    else:
        raise ValueError(
            f"Unsupported calibration verdict: {calibration_verdict!r}"
        )

    if action == "DEFECT":
        note = (
            "PatchCore indicates anomaly evidence. "
            "Inspect the localization and severity; this is not physical-defect confirmation."
        )
    else:
        note = (
            "The image is classified as normal by the available prediction evidence. "
            "Continue following factory inspection policy."
        )

    if shifted or reliability.lower() == "low" or issues:
        note += " Caution: acquisition or image-quality warnings were detected."

    return DecisionResult(
        action=action,
        manual_review_required=False,
        rationale=rationale,
        note=note,
    )
