"""
PROMETHEUS-X — Reusable UI Components
All Streamlit + HTML rendering helpers live here to keep streamlit_app.py clean.
"""

import streamlit as st
import numpy as np
from PIL import Image
from typing import Optional, Dict

from app.utils import (
    VERDICT_COLOURS,
    SEVERITY_COLOURS,
    RELIABILITY_COLOURS,
    format_score,
)


# ─── Hero Header ─────────────────────────────────────────────────────────────

def render_hero():
    st.markdown("""
    <div class="prometheus-hero">
        <div class="badge">Industrial AI Inspection Platform</div>
        <h1>PROMETHEUS-X</h1>
        <div class="subtitle">Open-Set Industrial Anomaly Detection</div>
        <div style="margin-top:1rem;font-size:0.82rem;color:#475569;max-width:680px;line-height:1.6;">
            PatchCore-based visual inspection system with anomaly localisation,
            severity estimation, confidence calibration, and distribution shift detection.
            All results are decision support — factory safety policy governs final disposition.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Section Labels ───────────────────────────────────────────────────────────

def section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


# ─── Status Badge (sidebar) ───────────────────────────────────────────────────

def model_status_badge(model_ready: bool, backend_ready: bool):
    """Render model and backend availability indicators in the sidebar."""
    m_col, b_col = "status-green" if model_ready else "status-red", \
                   "status-green" if backend_ready else "status-amber"
    m_txt = "AI Model: Connected" if model_ready else "AI Model: Not Connected"
    b_txt = "Backend: Connected"  if backend_ready else "Backend: Unavailable"

    st.sidebar.markdown(f"""
    <div style="margin-bottom:0.3rem;">
        <span class="status-dot {m_col}"></span>
        <span style="font-size:0.8rem;color:#475569;">{m_txt}</span>
    </div>
    <div>
        <span class="status-dot {b_col}"></span>
        <span style="font-size:0.8rem;color:#475569;">{b_txt}</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Image Panel ──────────────────────────────────────────────────────────────

def render_uploaded_image(image: Image.Image, label: str = "Uploaded Image"):
    st.markdown(f'<div class="px-card-title">{label}</div>', unsafe_allow_html=True)
    st.image(image, width="stretch")
    w, h = image.size
    st.caption(f"Dimensions: {w} × {h} px  |  Mode: {image.mode}")


def render_heatmap(heatmap: Optional[np.ndarray | str]):
    """Display an anomaly heatmap returned by the AI model."""
    st.markdown('<div class="px-card-title">Anomaly Heatmap</div>', unsafe_allow_html=True)
    if heatmap is not None:
        st.image(heatmap, width="stretch", clamp=True)
    else:
        st.markdown("""
        <div class="placeholder-banner">
            Heatmap unavailable — verify the model output and checkpoint.
        </div>""", unsafe_allow_html=True)


def render_mask(mask: Optional[np.ndarray]):
    """Display the binary defect mask returned by the AI model."""
    st.markdown('<div class="px-card-title">Defect Mask</div>', unsafe_allow_html=True)
    if mask is not None:
        st.image(mask, width="stretch", clamp=True)
    else:
        st.markdown("""
        <div class="placeholder-banner">
            Defect mask unavailable — run an inspection with a valid anomaly map.
        </div>""", unsafe_allow_html=True)


# ─── Verdict Banner ───────────────────────────────────────────────────────────

def render_verdict(verdict: str):
    _, css_class = VERDICT_COLOURS.get(verdict, ("#475569", "verdict-unavail"))
    icons = {
        "Normal":             "✓",
        "Anomaly Detected":   "✗",
        "Uncertain":          "?",
        "Manual Review Required": "!",
        "Model Not Available":"—",
    }
    icon = icons.get(verdict, "—")
    st.markdown(f"""
    <div class="px-card">
        <div class="px-card-title">Inspection Verdict</div>
        <div class="verdict-badge {css_class}">
            <span style="font-size:1.3rem;">{icon}</span>
            {verdict}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Metric Cards ─────────────────────────────────────────────────────────────

def render_metrics(
    score: Optional[float],
    severity: str,
    confidence: Optional[float],
    reliability: str,
):
    sev_col  = SEVERITY_COLOURS.get(severity, "#475569")
    rel_col  = RELIABILITY_COLOURS.get(reliability, "#475569")
    conf_txt = f"{confidence:.1%}" if confidence is not None else "N/A"
    score_txt = format_score(score)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-tile">
            <div class="metric-label">Anomaly Score</div>
            <div class="metric-value" style="color:#1D4ED8;">{score_txt}</div>
            <div class="metric-unit">Raw model output</div>
        </div>
        <div class="metric-tile">
            <div class="metric-label">Severity</div>
            <div class="metric-value" style="color:{sev_col};">{severity}</div>
            <div class="metric-unit">Defect area estimate</div>
        </div>
        <div class="metric-tile">
            <div class="metric-label">Calibration Signal</div>
            <div class="metric-value" style="color:#1D4ED8;">{conf_txt}</div>
            <div class="metric-unit">Heuristic, not probability</div>
        </div>
        <div class="metric-tile">
            <div class="metric-label">Reliability</div>
            <div class="metric-value" style="color:{rel_col};">{reliability}</div>
            <div class="metric-unit">Result trustworthiness</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Distribution Shift Warning ───────────────────────────────────────────────

def render_shift_warning(shift_detected: bool):
    if shift_detected:
        st.markdown("""
        <div class="shift-warning">
            ⚠ <strong>Distribution Shift Detected</strong> — Production conditions have changed.
            Inspection results may be less reliable. Consider recalibrating the baseline.
        </div>
        """, unsafe_allow_html=True)


# ─── Recommended Action ───────────────────────────────────────────────────────

def render_recommendation(action: str):
    st.markdown(f"""
    <div class="px-card">
        <div class="px-card-title">Recommended Action</div>
        <div style="font-size:0.92rem;color:#0F172A;line-height:1.6;">{action}</div>
        <div class="info-banner" style="margin-top:0.8rem;margin-bottom:0;">
            This is decision support only. Factory safety policy governs final product disposition.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Model Unavailable Notice ─────────────────────────────────────────────────

def render_model_unavailable():
    st.markdown("""
    <div class="placeholder-banner" style="padding:2rem;">
        <div style="font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;">
            PatchCore Model Unavailable
        </div>
        <div style="font-size:0.84rem;color:#475569;line-height:1.6;">
            Check that the existing checkpoint and project virtual environment are available.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Inspection History Table ─────────────────────────────────────────────────

def render_history_table(history: list):
    if not history:
        st.info("No inspections recorded in this session.")
        return

    rows = []
    for item in reversed(history):
        score = item.get("score")
        confidence = item.get("confidence")
        rows.append({
            "Inspection ID": item.get("id", ""),
            "Timestamp": item.get("timestamp", ""),
            "Category": item.get("category", ""),
            "Anomaly Score": f"{score:.4f}" if isinstance(score, (int, float)) else "N/A",
            "Verdict": item.get("verdict", ""),
            "Severity": item.get("severity", ""),
            "Confidence": f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "N/A",
            "Reliability": item.get("reliability", ""),
        })

    st.dataframe(rows, width="stretch", hide_index=True)


# ─── Feedback Form ────────────────────────────────────────────────────────────

def render_feedback_form(last_inspection_id: Optional[str]) -> Optional[Dict]:
    """
    Render the false-positive / quality feedback form.
    Returns a dict with feedback data if submitted, else None.
    """
    section_label("Inspection Feedback")
    with st.expander("Submit Inspection Feedback", expanded=True):
        st.markdown("""
        <div class="info-banner">
            Your feedback helps validate model performance and improve future versions.
            Feedback is stored locally and does <strong>not</strong> automatically retrain the model.
        </div>
        """, unsafe_allow_html=True)

        insp_id = st.text_input(
            "Inspection ID",
            value=last_inspection_id or "",
            placeholder="PX-XXXX",
        )
        fb_type = st.selectbox(
            "Feedback Type",
            [
                "Correct Detection",
                "False Positive (normal product flagged as defective)",
                "Missed Defect (defective product not flagged)",
                "Uncertain / Unclear Result",
            ],
        )
        notes = st.text_area(
            "Additional Notes (optional)",
            placeholder="Describe the actual condition of the product…",
            height=80,
        )
        submitted = st.button("Submit Feedback", key="feedback_submit")
        if submitted:
            if not insp_id.strip():
                st.warning("Please enter the Inspection ID for the result you are evaluating.")
                return None
            return {
                "inspection_id": insp_id.strip(),
                "feedback_type": fb_type,
                "notes":         notes.strip(),
                "timestamp":     __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
    return None
