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
) -> DecisionResult:
    """Return a conservative action without treating anomalies as confirmed defects."""
    issues = list(quality_issues or [])
    rationale = []

    if shifted:
        rationale.append("Input distribution shift detected")
    if reliability.lower() == "low":
        rationale.append("Automated result reliability is low")
    if issues:
        rationale.extend(f"Image quality issue: {issue}" for issue in issues)

    if shifted or reliability.lower() == "low" or issues:
        return DecisionResult(
            action="MANUAL REVIEW REQUIRED",
            manual_review_required=True,
            rationale=rationale,
            note="Automated evidence is not reliable enough for an automatic disposition.",
        )

    verdict = calibration_verdict.strip().lower()
    if verdict == "uncertain":
        return DecisionResult(
            action="UNCERTAIN",
            manual_review_required=True,
            rationale=["Anomaly score falls in the calibrated uncertainty band"],
            note="Review the image and localization before disposition.",
        )

    if verdict == "defect":
        return DecisionResult(
            action="DEFECT",
            manual_review_required=False,
            rationale=["Anomaly score exceeds the normal-reference decision threshold"],
            note="Anomaly evidence is present; this is not physical-defect confirmation or safety certification.",
        )

    if verdict == "normal":
        return DecisionResult(
            action="NORMAL",
            manual_review_required=False,
            rationale=["Anomaly score is consistent with the normal reference"],
            note="Decision support only; follow factory inspection policy.",
        )

    raise ValueError(f"Unsupported calibration verdict: {calibration_verdict!r}")