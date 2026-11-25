"""
PROMETHEUS-X — Custom CSS Styles
Provides the dark industrial dashboard aesthetic for the Streamlit application.
"""


def get_main_css() -> str:
    """Return the complete CSS stylesheet for the PROMETHEUS-X dashboard."""
    return """
<style>
/* ─── Google Fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Root Variables ───────────────────────────────────────────── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-secondary:  #0d1526;
    --bg-card:       #111827;
    --bg-card-hover: #1a2235;
    --border-subtle: #1e2d45;
    --border-accent: #0ea5e9;
    --cyan:          #06b6d4;
    --cyan-light:    #22d3ee;
    --blue:          #3b82f6;
    --blue-light:    #60a5fa;
    --green:         #10b981;
    --amber:         #f59e0b;
    --red:           #ef4444;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #475569;
    --radius:        10px;
    --radius-lg:     16px;
    --shadow:        0 4px 24px rgba(0,0,0,0.5);
}

/* ─── App Container ────────────────────────────────────────────── */
.stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ─── Hide Streamlit Branding ───────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ─── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary) !important; }

/* ─── Main Content Padding ─────────────────────────────────────── */
.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1200px !important;
}

/* ─── Hero Header ──────────────────────────────────────────────── */
.prometheus-hero {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1a3a 50%, #0a1628 100%);
    border: 1px solid var(--border-subtle);
    border-top: 2px solid var(--cyan);
    border-radius: var(--radius-lg);
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.prometheus-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.prometheus-hero h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    background: linear-gradient(90deg, var(--cyan-light), var(--blue-light));
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 0.3rem 0 !important;
}
.prometheus-hero .subtitle {
    font-size: 0.95rem;
    color: var(--text-secondary);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 500;
}
.prometheus-hero .badge {
    display: inline-block;
    background: rgba(6,182,212,0.12);
    border: 1px solid rgba(6,182,212,0.3);
    color: var(--cyan);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 0.8rem;
    text-transform: uppercase;
}

/* ─── Section Label ────────────────────────────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-subtle);
}

/* ─── Cards ────────────────────────────────────────────────────── */
.px-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    transition: border-color 0.2s;
}
.px-card:hover { border-color: rgba(6,182,212,0.25); }
.px-card-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.8rem;
}

/* ─── Metric Tiles ─────────────────────────────────────────────── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.8rem;
    margin-bottom: 1rem;
}
.metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 1rem;
    text-align: center;
}
.metric-tile .metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.metric-tile .metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
}
.metric-tile .metric-unit {
    font-size: 0.7rem;
    color: var(--text-secondary);
}

/* ─── Verdict Badges ───────────────────────────────────────────── */
.verdict-normal    { color: var(--green);  border-color: rgba(16,185,129,0.4); background: rgba(16,185,129,0.08); }
.verdict-anomaly   { color: var(--red);    border-color: rgba(239,68,68,0.4);  background: rgba(239,68,68,0.08);  }
.verdict-uncertain { color: var(--amber);  border-color: rgba(245,158,11,0.4); background: rgba(245,158,11,0.08); }
.verdict-unavail   { color: var(--text-muted); border-color: var(--border-subtle); background: rgba(71,85,105,0.08); }

.verdict-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    border: 1px solid;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}

/* ─── Status Indicators ────────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-green  { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-amber  { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
.status-red    { background: var(--red);   box-shadow: 0 0 6px var(--red);   }
.status-gray   { background: var(--text-muted); }

/* ─── Warning Banner ───────────────────────────────────────────── */
.shift-warning {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.4);
    border-left: 4px solid var(--amber);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    color: var(--amber);
    font-size: 0.88rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* ─── Info / Disclaimer Banner ─────────────────────────────────── */
.info-banner {
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.25);
    border-left: 4px solid var(--blue);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    color: var(--blue-light);
    font-size: 0.84rem;
    margin-bottom: 1rem;
}
.placeholder-banner {
    background: rgba(6,182,212,0.07);
    border: 1px dashed rgba(6,182,212,0.3);
    border-radius: var(--radius);
    padding: 1.2rem;
    color: var(--cyan);
    font-size: 0.84rem;
    text-align: center;
    margin-bottom: 1rem;
}

/* ─── Upload Zone ──────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-accent) !important;
    border-radius: var(--radius) !important;
}

/* ─── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.6rem !important;
    transition: opacity 0.2s !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover { opacity: 0.9 !important; }
.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
}

/* ─── Selectbox / Slider ───────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"]    > div {
    background: var(--bg-card) !important;
}

/* ─── Tables ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: var(--radius) !important; }
thead tr th {
    background: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ─── Expander ─────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
}

/* ─── Divider ──────────────────────────────────────────────────── */
hr { border-color: var(--border-subtle) !important; }

/* ─── Code / Mono ──────────────────────────────────────────────── */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: rgba(6,182,212,0.08) !important;
    color: var(--cyan-light) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}
</style>
"""
