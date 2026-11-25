"""
PROMETHEUS-X — Utility Functions
Helper functions for image processing, ID generation, timestamps, and
result formatting used throughout the application.
"""

import uuid
import datetime
import io
from typing import Optional, Tuple

from PIL import Image


# ─── Image Helpers ──────────────────────────────────────────────────────────

def load_image_from_upload(uploaded_file) -> Optional[Image.Image]:
    """
    Load a PIL Image from a Streamlit UploadedFile object.

    Parameters
    ----------
    uploaded_file : st.runtime.uploaded_file_manager.UploadedFile

    Returns
    -------
    PIL.Image or None on failure.
    """
    try:
        image = Image.open(uploaded_file).convert("RGB")
        return image
    except Exception:
        return None


def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
    """Return (width, height) of a PIL image."""
    return image.size


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL image to raw bytes (for display)."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def resize_image_for_display(image: Image.Image, max_px: int = 512) -> Image.Image:
    """
    Resize an image so its longest side is at most max_px pixels,
    preserving aspect ratio.
    """
    w, h = image.size
    if max(w, h) <= max_px:
        return image
    scale = max_px / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ─── Inspection ID / Timestamp ───────────────────────────────────────────────

def generate_inspection_id() -> str:
    """Generate a short unique inspection ID like 'PX-A3F7'."""
    return "PX-" + uuid.uuid4().hex[:4].upper()


def current_timestamp() -> str:
    """Return current local datetime as a readable string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─── Result Formatting ────────────────────────────────────────────────────────

def format_score(score: Optional[float]) -> str:
    """Format an anomaly score for display (4 decimal places)."""
    if score is None:
        return "N/A"
    return f"{score:.4f}"


def verdict_from_score(score: float, threshold: float = 0.5) -> str:
    """
    Map a normalised anomaly score to a verdict string.
    NOTE: This is a simple threshold rule — the actual decision should
    come from the AI model's calibrated output (Person 2's calibration.py).
    """
    if score < threshold * 0.6:
        return "Normal"
    elif score > threshold * 1.4:
        return "Anomaly Detected"
    else:
        return "Uncertain"


def severity_label(severity: str) -> str:
    """Return a display-safe severity string."""
    mapping = {
        "low":    "Low",
        "medium": "Medium",
        "high":   "High",
        "none":   "None",
    }
    return mapping.get(str(severity).lower(), str(severity).capitalize())


def reliability_label(reliability: str) -> str:
    """Return a display-safe reliability string."""
    mapping = {
        "high":   "High",
        "medium": "Medium",
        "low":    "Low",
    }
    return mapping.get(str(reliability).lower(), str(reliability).capitalize())


def get_recommended_action(verdict: str, severity: str, shift_detected: bool) -> str:
    """
    Provide a decision-support recommendation based on inspection results.

    IMPORTANT: This is decision support only, NOT a guaranteed safety decision.
    Factory safety policy must always take precedence.
    """
    if verdict == "Normal" and not shift_detected:
        return "Product appears normal. Continue standard production flow."
    elif verdict == "Normal" and shift_detected:
        return (
            "Product appears normal, but production conditions have shifted. "
            "Consider re-baselining the inspection system."
        )
    elif verdict == "Anomaly Detected":
        sev = str(severity).lower()
        if sev == "high":
            return "Reject or quarantine per factory policy. Manual inspection strongly recommended."
        elif sev == "medium":
            return "Manual inspection recommended before releasing product."
        else:
            return "Flag for quality review. Confirm whether defect is cosmetic or functional."
    elif verdict == "Uncertain":
        return "Result is inconclusive. Manual inspection recommended."
    else:
        return "AI model is not available. Manual inspection required."


def calculate_reliability(confidence: Optional[float], shift_detected: bool) -> str:
    """
    Estimate result reliability from calibrated confidence and shift status.
    Returns 'High', 'Medium', or 'Low'.

    This is a heuristic fallback. Person 2's backend functions should
    provide the authoritative reliability value when available.
    """
    if confidence is None:
        return "Low"
    if shift_detected:
        return "Low"
    if confidence >= 0.75:
        return "High"
    elif confidence >= 0.45:
        return "Medium"
    else:
        return "Low"


# ─── Colour Helpers ───────────────────────────────────────────────────────────

VERDICT_COLOURS = {
    "Normal":            ("#10b981", "verdict-normal"),
    "Anomaly Detected":  ("#ef4444", "verdict-anomaly"),
    "Uncertain":         ("#f59e0b", "verdict-uncertain"),
    "Model Not Available": ("#475569", "verdict-unavail"),
}

SEVERITY_COLOURS = {
    "High":   "#ef4444",
    "Medium": "#f59e0b",
    "Low":    "#10b981",
    "None":   "#475569",
}

RELIABILITY_COLOURS = {
    "High":   "#10b981",
    "Medium": "#f59e0b",
    "Low":    "#ef4444",
}
