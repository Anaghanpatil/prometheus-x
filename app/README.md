# PROMETHEUS-X — Application Layer

**Author:** Person 3  
**Role:** Application development, AI & backend integration, Hugging Face Spaces deployment

---

## Purpose

This folder (`app/`) contains the entire user-facing Streamlit application for PROMETHEUS-X.  
It is Person 3's primary responsibility. No AI model training or backend algorithm code lives here.

---

## File Structure

```
app/
├── streamlit_app.py   ← Entry point — UI orchestration, tabs, inspection flow
├── components.py      ← Reusable Streamlit + HTML widgets
├── styles.py          ← Full CSS stylesheet (dark industrial theme)
├── utils.py           ← Helper functions (IDs, formatting, recommendations)
└── README.md          ← This file
```

---

## Features

| Feature | Status |
|---|---|
| Dark industrial dashboard UI | ✅ Complete |
| Image upload (JPG/JPEG/PNG) | ✅ Complete |
| Product category dropdown | ✅ Complete |
| Sensitivity slider | ✅ Complete |
| AI Model integration adapter | 🔌 Placeholder (Person 1) |
| Anomaly heatmap display | 🔌 Awaiting model |
| Defect mask display | 🔌 Awaiting model |
| Verdict display | ✅ Complete |
| Anomaly score display | ✅ Complete |
| Severity integration | 🔌 Placeholder (Person 2) |
| Calibrated confidence | 🔌 Placeholder (Person 2) |
| Distribution shift warning | 🔌 Placeholder (Person 2) |
| Reliability display | ✅ Complete (heuristic fallback) |
| Recommended action | ✅ Complete |
| Inspection history (session) | ✅ Complete |
| Download inspection log | ✅ Complete |
| False-positive feedback | ✅ Complete |
| Hugging Face Spaces config | ✅ Complete |

---

## Technology Stack

- **Python** 3.10+
- **Streamlit** ≥ 1.32
- **Pillow** — image loading and processing
- **NumPy** — array operations
- Streamlit native data tables — history and feedback rendering

---

## Running Locally

```bash
# From the project root
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

---

## Integrating the AI Model (Person 1)

1. Train PatchCore using Anomalib on MVTec-AD (Bottle category first).
2. Open `ai-model/predict.py`.
3. Implement `_load_model(category)` to load your saved checkpoint.
4. Fill in the real inference code inside `predict()`.
5. Set `MODEL_AVAILABLE = True` at the top of the file.
6. The application immediately picks up real predictions — no other changes needed.

**Expected interface:**
```python
score, heatmap, mask = predict(image, category="Bottle", threshold=0.5)
# score   : float          — raw anomaly score
# heatmap : np.ndarray     — shape (H, W), float32, values 0–1
# mask    : np.ndarray     — shape (H, W), uint8, 0 or 255
```

---

## Integrating the Backend (Person 2)

Open each file in `backend/` and implement the function body, then set the `*_AVAILABLE` flag to `True`.

| File | Function | Flag |
|---|---|---|
| `backend/severity.py` | `estimate_severity(heatmap, threshold)` | `SEVERITY_AVAILABLE` |
| `backend/calibration.py` | `calibrate_score(score)` | `CALIBRATION_AVAILABLE` |
| `backend/shift.py` | `detect_shift(image)` | `SHIFT_AVAILABLE` |

Do **not** change the function signatures without coordinating with Person 3.

---

## Deployment — Hugging Face Spaces

The Space is configured via the YAML frontmatter in the root `README.md`.

```yaml
sdk: streamlit
sdk_version: "1.32.0"
app_file: app/streamlit_app.py
```

**Steps:**
1. Create a Space on huggingface.co (Streamlit SDK).
2. Push this repository to the Space's git remote.
3. The Space builds automatically.
4. For large model weights: upload to Hugging Face Hub and load via `huggingface_hub.hf_hub_download()`.
5. Store any secrets (HF tokens, API keys) in Space Settings → Secrets — never hardcode them.

---

## Known Limitations

- History is session-based only; it clears on browser refresh.
- Feedback does not trigger model retraining.
- Only the **Bottle** category has a model slot; other categories show an unsupported notice.
- Real predictions require Person 1's model and Person 2's backend to be integrated.
- Large model weights must be managed separately (HF Hub / LFS) — not stored in the GitHub repository.

---

## Disclaimer

PROMETHEUS-X results are **decision support only**.  
All final quality-control decisions must follow factory safety policy.  
Do not use as the sole basis for safety-critical decisions.
