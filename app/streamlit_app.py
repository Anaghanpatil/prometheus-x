"""
PROMETHEUS-X — Main Streamlit Application
==========================================
Author  : Person 3 (Application & Integration Layer)
Project : PROMETHEUS-X — Open-Set Industrial Anomaly Detection
Hackathon Problem : A06

Entry point for the Streamlit application.
Run locally with:
    streamlit run app/streamlit_app.py

─── Architecture ────────────────────────────────────────────────────────────
  streamlit_app.py   ← this file (UI orchestration)
  components.py      ← reusable HTML/Streamlit widgets
  styles.py          ← CSS stylesheet
  utils.py           ← helper functions (IDs, formatting, recommendations)

  ai-model/predict.py      ← Person 1's model integration adapter
  backend/severity.py      ← Person 2's severity estimator
  backend/calibration.py   ← Person 2's confidence calibrator
  backend/shift.py         ← Person 2's distribution shift detector

─── Integration Status ───────────────────────────────────────────────────────
  When Person 1 provides the AI model:
    1. Set MODEL_AVAILABLE = True in ai-model/predict.py
    2. Implement _load_model() and the real predict() body

  When Person 2 provides backend functions:
    1. Set SEVERITY_AVAILABLE = True in backend/severity.py
    2. Set CALIBRATION_AVAILABLE = True in backend/calibration.py
    3. Set SHIFT_AVAILABLE = True in backend/shift.py
    4. Implement the real function bodies
"""

import sys
import json
import logging
import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

# ─── Path setup (allow imports from project root) ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Project imports ──────────────────────────────────────────────────────────
from app.styles     import get_main_css
from app.components import (
    render_hero,
    section_label,
    model_status_badge,
    render_uploaded_image,
    render_heatmap,
    render_mask,
    render_verdict,
    render_metrics,
    render_shift_warning,
    render_recommendation,
    render_model_unavailable,
    render_history_table,
    render_feedback_form,
)
from app.utils import (
    load_image_from_upload,
    generate_inspection_id,
    current_timestamp,
    format_score,
    severity_label,
    reliability_label,
    get_recommended_action,
    calculate_reliability,
)

# ─── AI Model adapter (Person 1) ─────────────────────────────────────────────
# Try importing from the ai_model package (Python-safe name for ai-model/).
# If the package isn't present yet, fall back to safe no-op stubs so the UI
# still runs in full preview mode.
try:
    from ai_model.predict import predict, is_model_available, get_supported_categories
    _MODEL_IMPORT_OK = True
except ImportError:
    _MODEL_IMPORT_OK = False
    def predict(image, category="Bottle", threshold=0.5):
        return None, None, None
    def is_model_available():
        return False
    def get_supported_categories():
        return ["Bottle"]

# ─── Backend adapters (Person 2) ─────────────────────────────────────────────
try:
    from backend.severity    import estimate_severity, is_severity_available
    from backend.calibration import calibrate_score, is_calibration_available
    from backend.shift       import detect_shift, is_shift_available
    _BACKEND_IMPORT_OK = True
except ImportError:
    _BACKEND_IMPORT_OK = False
    def estimate_severity(heatmap, threshold=0.5):
        return 0.0, "Unavailable"
    def calibrate_score(score):
        return None
    def detect_shift(image):
        return False
    def is_severity_available():
        return False
    def is_calibration_available():
        return False
    def is_shift_available():
        return False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PROMETHEUS-X | Industrial Anomaly Detection",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject CSS ───────────────────────────────────────────────────────────────
st.markdown(get_main_css(), unsafe_allow_html=True)

# ─── Session State Initialisation ────────────────────────────────────────────
def _init_session():
    defaults = {
        "inspection_history": [],       # list of result dicts
        "feedback_log":       [],       # list of feedback dicts
        "last_inspection_id": None,     # string
        "uploaded_image":     None,     # PIL Image
        "last_results":       None,     # dict of last run outputs
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem 0;">
        <div style="font-size:1.6rem;font-weight:800;
                    background:linear-gradient(90deg,#22d3ee,#60a5fa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            PROMETHEUS-X
        </div>
        <div style="font-size:0.68rem;color:#475569;letter-spacing:0.15em;
                    text-transform:uppercase;margin-top:2px;">
            Industrial AI Platform
        </div>
    </div>
    <hr style="border-color:#1e2d45;margin:0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    # ── System Status ─────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;'
                'text-transform:uppercase;color:#06b6d4;margin-bottom:0.5rem;">System Status</div>',
                unsafe_allow_html=True)
    backend_ready = (
        is_severity_available()
        and is_calibration_available()
        and is_shift_available()
    )
    model_status_badge(is_model_available(), backend_ready)

    st.markdown("<hr style='border-color:#1e2d45;'>", unsafe_allow_html=True)

    # ── Category Selection ────────────────────────────────────────────────────
    st.markdown('<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;'
                'text-transform:uppercase;color:#06b6d4;margin-bottom:0.5rem;">Product Category</div>',
                unsafe_allow_html=True)

    all_categories = ["Bottle", "Cable", "Capsule", "Metal Nut", "Screw"]
    supported = get_supported_categories()

    category = st.selectbox(
        "Select product type",
        options=all_categories,
        index=0,
        label_visibility="collapsed",
    )
    if category not in supported:
        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
                    border-radius:8px;padding:0.6rem 0.8rem;font-size:0.78rem;color:#f59e0b;">
            Model for <strong>{category}</strong> is not yet available.
            Results cannot be generated for this category.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d45;'>", unsafe_allow_html=True)

    # ── Sensitivity Slider ────────────────────────────────────────────────────
    st.markdown('<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;'
                'text-transform:uppercase;color:#06b6d4;margin-bottom:0.5rem;">Inspection Sensitivity</div>',
                unsafe_allow_html=True)

    sensitivity = st.slider(
        "Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        label_visibility="collapsed",
    )
    sens_note = "Low" if sensitivity < 0.4 else ("High" if sensitivity > 0.7 else "Balanced")
    st.caption(
        f"**{sens_note}** sensitivity (threshold = {sensitivity:.2f})  \n"
        "Higher sensitivity may detect more suspicious regions "
        "but can increase false positives. "
        "This value controls the mask binarisation threshold passed to the model."
    )

    st.markdown("<hr style='border-color:#1e2d45;'>", unsafe_allow_html=True)

    # ── History Controls ──────────────────────────────────────────────────────
    st.markdown('<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.12em;'
                'text-transform:uppercase;color:#06b6d4;margin-bottom:0.5rem;">Session Controls</div>',
                unsafe_allow_html=True)

    if st.button("Clear History", key="clear_history"):
        st.session_state.inspection_history = []
        st.session_state.last_inspection_id = None
        st.session_state.last_results = None
        st.success("History cleared.")

    # Download inspection log
    if st.session_state.inspection_history:
        log_json = json.dumps(st.session_state.inspection_history, indent=2, default=str)
        st.download_button(
            label="Download Inspection Log",
            data=log_json,
            file_name=f"prometheus_x_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    # Download feedback log
    if st.session_state.feedback_log:
        fb_json = json.dumps(st.session_state.feedback_log, indent=2, default=str)
        st.download_button(
            label="Download Feedback Log",
            data=fb_json,
            file_name=f"prometheus_x_feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem;color:#334155;text-align:center;line-height:1.6;">
        PROMETHEUS-X v0.1.0<br>Hackathon Demo — A06<br>
        Results are decision support only.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════════

render_hero()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_inspect, tab_history, tab_feedback, tab_about = st.tabs([
    "Inspection Workspace",
    "Inspection History",
    "Feedback",
    "About",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — INSPECTION WORKSPACE
# ══════════════════════════════════════════════════════════════════════════════

with tab_inspect:

    # ── Upload Section ────────────────────────────────────────────────────────
    section_label("Image Upload")

    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("""
        <div class="px-card">
            <div class="px-card-title">Upload Product Image</div>
            <div style="font-size:0.82rem;color:#64748b;margin-bottom:0.8rem;line-height:1.5;">
                Upload a JPG, JPEG, or PNG image of the product to inspect.
                Ensure the image is well-lit and shows the product surface clearly.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

        # Validate and load
        image: Optional[Image.Image] = None
        if uploaded_file is not None:
            image = load_image_from_upload(uploaded_file)
            if image is None:
                st.error(
                    "Unable to open the uploaded file as an image. "
                    "Please upload a valid JPG, JPEG, or PNG file."
                )
            else:
                st.session_state.uploaded_image = image
                w, h = image.size
                st.markdown(f"""
                <div style="font-size:0.8rem;color:#22d3ee;margin-top:0.5rem;">
                    ✓ Image loaded — {w} × {h} px
                </div>
                """, unsafe_allow_html=True)
        else:
            # If a previous image is in session, we let the user keep it
            if st.session_state.uploaded_image is not None:
                st.info("Using previously uploaded image. Upload a new file to replace it.")
                image = st.session_state.uploaded_image

        # Quick category check
        if category not in supported:
            st.markdown(f"""
            <div class="info-banner">
                The selected category <strong>{category}</strong> does not have a trained model.
                Switch to <strong>Bottle</strong> to run an inspection (once the AI model is integrated).
            </div>
            """, unsafe_allow_html=True)

        # ── Run Inspection Button ─────────────────────────────────────────────
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        run_btn = st.button(
            "Run AI Inspection",
            type="primary",
            use_container_width=True,
            disabled=(image is None or category not in supported),
        )

    with col_preview:
        if image is not None:
            st.markdown('<div class="px-card">', unsafe_allow_html=True)
            render_uploaded_image(image, label="Product Image Preview")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="px-card" style="min-height:220px;display:flex;
                        flex-direction:column;align-items:center;justify-content:center;
                        text-align:center;">
                <div style="font-size:2rem;margin-bottom:0.5rem;opacity:0.3;">⬡</div>
                <div style="color:#334155;font-size:0.82rem;">
                    No image uploaded yet.<br>Upload a product image to begin inspection.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d45;'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  INSPECTION EXECUTION
    # ══════════════════════════════════════════════════════════════════════════

    if run_btn and image is not None:
        with st.spinner("Running inspection…"):

            inspection_id  = generate_inspection_id()
            timestamp      = current_timestamp()
            score          = None
            heatmap        = None
            mask           = None
            verdict        = "Model Not Available"
            area_pct       = 0.0
            severity       = "Unavailable"
            confidence     = None
            shift_detected = False
            reliability    = "Low"
            run_error      = None

            # ── Step 1: AI Model ─────────────────────────────────────────────
            if not is_model_available():
                run_error = "model_unavailable"
            else:
                try:
                    score, heatmap, mask = predict(
                        image,
                        category=category,
                        threshold=sensitivity,
                    )
                    if score is not None:
                        verdict = "Uncertain"
                    else:
                        run_error = "model_returned_none"
                except ValueError as ve:
                    run_error = f"unsupported_category: {ve}"
                except RuntimeError as re:
                    run_error = f"prediction_failed: {re}"
                except Exception as exc:
                    run_error = f"unexpected: {exc}"
                    logger.error("Unexpected prediction error: %s", exc)

            # ── Step 2: Distribution Shift ────────────────────────────────────
            try:
                shift_detected = detect_shift(image)
            except Exception as exc:
                logger.warning("Shift detection failed: %s", exc)
                shift_detected = False

            # ── Step 3: Severity Estimation ───────────────────────────────────
            if heatmap is not None:
                try:
                    area_pct, severity_raw = estimate_severity(heatmap, sensitivity)
                    severity = severity_label(severity_raw)
                except Exception as exc:
                    logger.warning("Severity estimation failed: %s", exc)
                    severity = "Unavailable"
            elif verdict == "Normal":
                severity = "None"

            # ── Step 4: Confidence Calibration ────────────────────────────────
            if score is not None:
                try:
                    confidence = calibrate_score(score)
                except Exception as exc:
                    logger.warning("Calibration failed: %s", exc)
                    confidence = None

            # ── Step 5: Reliability ───────────────────────────────────────────
            reliability = reliability_label(
                calculate_reliability(confidence, shift_detected)
            )

            # ── Persist to history ────────────────────────────────────────────
            record = {
                "id":          inspection_id,
                "timestamp":   timestamp,
                "category":    category,
                "score":       score,
                "verdict":     verdict,
                "severity":    severity,
                "confidence":  confidence,
                "reliability": reliability,
                "shift":       shift_detected,
            }
            st.session_state.inspection_history.append(record)
            st.session_state.last_inspection_id = inspection_id
            st.session_state.last_results = record

        # ── Results Dashboard ─────────────────────────────────────────────────
        section_label("Inspection Results")

        # Error / status notices
        if run_error == "model_unavailable":
            render_model_unavailable()
        elif run_error:
            st.error(f"Inspection could not be completed: {run_error}")

        render_shift_warning(shift_detected)

        # Main results layout
        res_col1, res_col2, res_col3 = st.columns(3, gap="medium")

        with res_col1:
            st.markdown('<div class="px-card">', unsafe_allow_html=True)
            render_uploaded_image(image, label="Original Image")
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="px-card">', unsafe_allow_html=True)
            render_heatmap(heatmap)
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col3:
            st.markdown('<div class="px-card">', unsafe_allow_html=True)
            render_mask(mask)
            st.markdown('</div>', unsafe_allow_html=True)

        # Verdict
        render_verdict(verdict)

        # Metric tiles
        render_metrics(score, severity, confidence, reliability)

        # Recommendation
        recommended = get_recommended_action(verdict, severity, shift_detected)
        render_recommendation(recommended)

        # Inspection ID footer
        st.markdown(f"""
        <div style="font-size:0.72rem;color:#334155;text-align:right;margin-top:0.5rem;">
            Inspection ID: <span style="font-family:monospace;color:#475569;">{inspection_id}</span>
            &nbsp;|&nbsp; {timestamp}
        </div>
        """, unsafe_allow_html=True)

    elif image is None and not run_btn:
        # Introductory instructions
        section_label("Getting Started")
        st.markdown("""
        <div class="px-card">
            <div class="px-card-title">How to Inspect a Product</div>
            <ol style="color:#94a3b8;font-size:0.88rem;line-height:2;">
                <li>Select the <strong style="color:#f1f5f9;">product category</strong> from the sidebar.</li>
                <li>Adjust the <strong style="color:#f1f5f9;">inspection sensitivity</strong> slider as needed.</li>
                <li>Upload a <strong style="color:#f1f5f9;">JPG, JPEG, or PNG</strong> image of the product.</li>
                <li>Click <strong style="color:#22d3ee;">Run AI Inspection</strong>.</li>
                <li>Review the anomaly heatmap, verdict, and metrics in the results panel.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-banner">
            <strong>Note:</strong> The AI model and intelligence backend are integration placeholders.
            Once Person 1 and Person 2 complete their modules, real predictions will appear here.
            All results will clearly state their source.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — INSPECTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    section_label("Inspection History")
    st.markdown("""
    <div class="info-banner">
        History is session-based and cleared when you close or refresh the browser tab.
        Download the log before closing if you need to retain records.
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.inspection_history
    render_history_table(history)

    if history:
        col_dl, col_clr = st.columns([2, 1])
        with col_dl:
            log_json = json.dumps(history, indent=2, default=str)
            st.download_button(
                label="Download as JSON",
                data=log_json,
                file_name=f"prometheus_x_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        with col_clr:
            if st.button("Clear All History", key="clear_hist_tab"):
                st.session_state.inspection_history = []
                st.session_state.last_inspection_id = None
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

with tab_feedback:
    feedback = render_feedback_form(st.session_state.last_inspection_id)
    if feedback:
        st.session_state.feedback_log.append(feedback)
        st.success(
            f"Feedback recorded for Inspection ID **{feedback['inspection_id']}**. "
            "Thank you — this helps validate model performance."
        )
        st.markdown("""
        <div class="info-banner">
            Feedback is stored in session memory only. Use <em>Download Feedback Log</em>
            in the sidebar to export it. Feedback does <strong>not</strong> automatically
            retrain the model — it is intended for offline model validation and review.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with tab_about:
    section_label("About PROMETHEUS-X")
    st.markdown("""
    <div class="px-card">
        <div class="px-card-title">Project Overview</div>
        <p style="color:#94a3b8;font-size:0.88rem;line-height:1.8;">
            <strong style="color:#f1f5f9;">PROMETHEUS-X</strong> is an open-set industrial anomaly detection system
            developed for the hackathon problem statement <strong>A06</strong>.
        </p>
        <p style="color:#94a3b8;font-size:0.88rem;line-height:1.8;">
            The system uses a <strong style="color:#f1f5f9;">PatchCore</strong> anomaly detection model
            trained on the <strong>MVTec Anomaly Detection (MVTec-AD)</strong> dataset.
            It provides pixel-level anomaly localisation via a heatmap,
            severity estimation, calibrated confidence scoring, and distribution shift detection.
        </p>
    </div>

    <div class="px-card">
        <div class="px-card-title">Team Responsibilities</div>
        <div style="font-size:0.85rem;color:#94a3b8;line-height:2.0;">
            <div><span style="color:#22d3ee;font-weight:600;">Person 1</span> — AI Model:
                PatchCore training (Anomalib), inference, heatmap &amp; mask generation.</div>
            <div><span style="color:#22d3ee;font-weight:600;">Person 2</span> — Intelligence Layer:
                Severity estimation, confidence calibration, distribution shift detection.</div>
            <div><span style="color:#22d3ee;font-weight:600;">Person 3</span> — Application:
                Streamlit UI, integration adapters, deployment, documentation.</div>
        </div>
    </div>

    <div class="px-card">
        <div class="px-card-title">Integration Checklist</div>
        <div style="font-size:0.82rem;color:#94a3b8;line-height:2.0;">
    """, unsafe_allow_html=True)

    checks = [
        ("AI Model (predict.py)", is_model_available()),
        ("Severity Estimation",   is_severity_available()),
        ("Confidence Calibration",is_calibration_available()),
        ("Shift Detection",       is_shift_available()),
    ]
    for name, ready in checks:
        dot   = "status-green" if ready else "status-red"
        label = "Connected" if ready else "Placeholder"
        st.markdown(
            f'<div><span class="status-dot {dot}"></span>'
            f'<span style="color:#94a3b8;">{name} — {label}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("""
        </div>
    </div>

    <div class="px-card">
        <div class="px-card-title">Technology Stack</div>
        <div style="font-size:0.82rem;color:#94a3b8;line-height:2.0;">
            Python 3.10+  |  Streamlit  |  Anomalib (PatchCore)  |  PyTorch  |
            Pillow  |  NumPy  |  Hugging Face Spaces
        </div>
    </div>

    <div class="info-banner">
        PROMETHEUS-X results are <strong>decision support only</strong>.
        All final quality-control decisions must follow factory safety policy.
    </div>
    """, unsafe_allow_html=True)
