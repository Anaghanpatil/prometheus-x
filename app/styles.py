"""
PROMETHEUS-X — Custom CSS Styles
Bright, high-contrast theme for the Streamlit application.

Class names match the original app so that all components keep working.
NOTE: fonts are applied to text elements only. Never set font-family on
`span` or `i` — Streamlit draws its icons (upload, arrows, etc.) with an icon
font in those elements, and overriding it prints the icon names as words.
"""


def get_main_css() -> str:
    """Return the complete CSS stylesheet for the PROMETHEUS-X dashboard."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --bg:        #F4F7FB;
    --surface:   #FFFFFF;
    --ink:       #0F172A;
    --slate:     #475569;
    --line:      #D5DEEA;
    --blue:      #1D4ED8;
    --blue-dk:   #1E40AF;
    --blue-bg:   #E8F0FF;
    --green:     #15803D;  --green-bg: #DCFCE7;
    --red:       #B91C1C;  --red-bg:   #FEE2E2;
    --amber:     #B45309;  --amber-bg: #FEF3C7;
    --radius:    12px;
    --shadow:    0 1px 2px rgba(15,23,42,.06), 0 4px 16px rgba(15,23,42,.06);
}

/* ─── Base (text elements only — no span / i) ──────────────────── */
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.stApp { background: var(--bg) !important; color: var(--ink) !important; }
.main .block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1240px !important; }

/* Keep the header (it holds the sidebar toggle); hide only deploy chrome */
.stDeployButton, #MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent !important; }

/* ─── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] .stMarkdown { color: var(--slate) !important; }

/* ─── Hero ─────────────────────────────────────────────────────── */
.prometheus-hero {
    background: linear-gradient(135deg, #FFFFFF 0%, #EAF1FF 100%);
    border: 1px solid var(--line);
    border-left: 8px solid var(--blue);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow);
}
.prometheus-hero h1 {
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em !important;
    color: var(--blue) !important;
    margin: 0.2rem 0 0.2rem 0 !important;
    padding: 0 !important;
}
.prometheus-hero .subtitle { font-size: 1rem; font-weight: 600; color: var(--ink); letter-spacing: 0.04em; }
.prometheus-hero .badge {
    display: inline-block; background: var(--blue); color: #fff;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em;
    padding: 4px 12px; border-radius: 20px; text-transform: uppercase;
}
.prometheus-hero div[style] { color: var(--slate) !important; }

/* ─── Tabs ─────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 2px solid var(--line); }
button[data-baseweb="tab"] { height: 3.3rem; padding: 0 1.5rem !important; background: transparent !important; border-radius: 10px 10px 0 0 !important; }
button[data-baseweb="tab"] p { font-size: 1.1rem !important; font-weight: 700 !important; color: var(--slate) !important; }
button[data-baseweb="tab"][aria-selected="true"] { background: var(--surface) !important; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--blue) !important; }
[data-baseweb="tab-highlight"] { background: var(--blue) !important; height: 4px !important; }

/* ─── Section label ────────────────────────────────────────────── */
.section-label {
    font-size: 1.15rem; font-weight: 800; color: var(--ink);
    margin: 1.2rem 0 0.8rem 0; display: flex; align-items: center; gap: 0.8rem;
}
.section-label::before { content: ''; width: 6px; height: 1.3rem; background: var(--blue); border-radius: 3px; }
.section-label::after  { content: ''; flex: 1; height: 1px; background: var(--line); }

/* ─── Cards ────────────────────────────────────────────────────── */
.px-card {
    background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 1.2rem 1.4rem; margin-bottom: 1rem; box-shadow: var(--shadow);
}
.px-card-title { font-size: 0.95rem; font-weight: 700; color: var(--ink); margin-bottom: 0.6rem; }

/* ─── Metric tiles (custom) ────────────────────────────────────── */
.metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.9rem; margin-bottom: 1rem; }
.metric-tile {
    background: var(--surface); border: 1px solid var(--line); border-top: 4px solid var(--blue);
    border-radius: var(--radius); padding: 1rem 1.1rem; text-align: left; box-shadow: var(--shadow);
}
.metric-tile .metric-label { font-size: 0.85rem; font-weight: 700; color: var(--slate); margin-bottom: 0.3rem; }
.metric-tile .metric-value { font-size: 1.7rem; font-weight: 800; color: var(--ink); font-family: 'JetBrains Mono', monospace; }
.metric-tile .metric-unit  { font-size: 0.8rem; color: var(--slate); margin-top: 0.15rem; }

/* ─── Native st.metric ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 0.8rem 1rem; box-shadow: var(--shadow);
}
[data-testid="stMetricLabel"] p { color: var(--slate) !important; font-weight: 600 !important; font-size: 0.85rem !important; }
[data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800 !important; }

/* ─── Verdict ──────────────────────────────────────────────────── */
.verdict-badge {
    display: inline-flex; align-items: center; gap: 0.7rem; border: 2px solid; border-radius: 12px;
    padding: 0.7rem 1.6rem; font-size: 1.5rem; font-weight: 800; letter-spacing: 0.02em;
}
.verdict-normal    { color: var(--green); border-color: var(--green); background: var(--green-bg); }
.verdict-anomaly   { color: var(--red);   border-color: var(--red);   background: var(--red-bg);   }
.verdict-uncertain { color: var(--amber); border-color: var(--amber); background: var(--amber-bg); }
.verdict-review    { color: var(--amber); border-color: var(--amber); background: var(--amber-bg); }
.verdict-unavail   { color: var(--slate); border-color: var(--line);  background: #EEF2F7; }

/* ─── Status dots ──────────────────────────────────────────────── */
.status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
.status-green { background: var(--green); } .status-amber { background: var(--amber); }
.status-red   { background: var(--red); }   .status-gray  { background: #94A3B8; }

/* ─── Banners ──────────────────────────────────────────────────── */
.shift-warning {
    background: var(--amber-bg); border: 1px solid #F2C46B; border-left: 6px solid var(--amber);
    border-radius: var(--radius); padding: 0.9rem 1.2rem; color: #78350F; font-size: 0.95rem; margin-bottom: 1rem;
}
.info-banner {
    background: var(--blue-bg); border: 1px solid #B6CCFF; border-left: 6px solid var(--blue);
    border-radius: var(--radius); padding: 0.9rem 1.2rem; color: #1E3A8A; font-size: 0.92rem; margin-bottom: 1rem;
}
.placeholder-banner {
    background: var(--surface); border: 2px dashed var(--line); border-radius: var(--radius);
    padding: 1.2rem; color: var(--slate); font-size: 0.92rem; text-align: center; margin-bottom: 1rem;
}

/* ─── Upload zone (box only; leave the button and icon alone) ──── */
[data-testid="stFileUploader"] section {
    background: var(--blue-bg) !important; border: 2px dashed var(--blue) !important; border-radius: var(--radius) !important;
}

/* ─── Buttons ──────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    min-height: 2.9rem; border-radius: 10px; font-weight: 700; font-size: 1rem;
    border: 2px solid var(--blue); background: var(--surface); color: var(--blue);
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background: var(--blue-bg); border-color: var(--blue-dk); color: var(--blue-dk);
}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"],
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
    background: var(--blue) !important; color: #fff !important; border-color: var(--blue) !important;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background: var(--blue-dk) !important; }
.stButton > button:disabled, button[data-testid="stBaseButton-primary"]:disabled {
    background: #E2E8F0 !important; color: #64748B !important; border-color: #CBD5E1 !important;
}

/* ─── Inputs, tables, expanders ────────────────────────────────── */
[data-testid="stWidgetLabel"] p { font-weight: 700 !important; color: var(--ink) !important; }
[data-baseweb="select"] > div, [data-baseweb="textarea"], [data-baseweb="input"] {
    background: var(--surface) !important; border-radius: 10px !important;
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow); }
[data-testid="stExpander"] { background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: var(--radius) !important; }
[data-testid="stImage"] img { border-radius: 10px; border: 1px solid var(--line); }
[data-testid="stCaptionContainer"] { color: var(--slate) !important; }
hr { border-color: var(--line) !important; }
code { font-family: 'JetBrains Mono', monospace; background: var(--blue-bg) !important; color: var(--blue-dk) !important; padding: 2px 6px; border-radius: 4px; }
</style>
"""
