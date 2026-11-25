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
| AI Model integration | PatchCore service in `backend/ai/patchcore_service.py` |
| Anomaly heatmap and overlay | Saved and displayed from PatchCore inference |
| Defect mask display | Derived from calibrated anomaly-map threshold |
| Verdict display | ✅ Complete |
| Anomaly score display | ✅ Complete |
| Severity integration | `src/severity.py` |
| Score calibration | `src/calibration.py`, using held-out `dataset/bottle/test/good` scores |
| Distribution shift warning | `src/shift.py`, using normal training images |
| Reliability display | Heuristic shift assessment |
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

```bat
REM Run from the project root in CMD
.venv\Scripts\activate.bat
python -m streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

---

## Dataset Evaluation

Run all 209 normal training images (reference/training data, not independent test data):

```bat
python -m backend.ai.batch_test --split train
```

Run all 83 labeled test images separately:

```bat
python -m backend.ai.batch_test --split test
```

Run the labeled synthetic shift experiment:

```bat
python -m backend.ai.batch_test --shift-experiment
```

The 83-image evaluation keeps the 209 PatchCore training images out of calibration and test metrics. It calibrates defects using the 20 held-out `test/good` images and uses leave-one-out calibration for each normal test image. Actual MVTec ground-truth masks are matched by category and filename.

Reports are written under `backend/ai/outputs/batch_<split>/`. The test report includes per-image decisions, calibrated thresholds, severity, reliability, masks, bounding boxes, inference time, CSV/JSON summaries, image-level ROC/PR metrics, and pixel AUROC/AP/Dice/IoU. Pixel AUROC/AP are measured on resized maps to bound CPU memory; Dice/IoU use full-resolution masks. PRO/AUPRO is not reported because the repository has no validated implementation.

The training run is a separate training-set inspection only. Its scores must not be presented as independent test results. Calibration is empirical, based on a 20-image held-out normal sample, and is not a probability guarantee. Synthetic shift results are not evidence of real production-domain robustness.

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
- Only the **Bottle** category is supported by the existing checkpoint.
- The held-out normal calibration set contains 20 images; uncertainty in tail thresholds is therefore material.
- No external-image dataset is included. Uploaded external images are individual demonstrations, not open-set generalization evidence.
- Pixel localization metrics are available for the provided defective test images with ground-truth masks.
- Large model weights must be managed separately (HF Hub / LFS) — not stored in the GitHub repository.

---

## Disclaimer

PROMETHEUS-X results are **decision support only**.  
All final quality-control decisions must follow factory safety policy.  
Do not use as the sole basis for safety-critical decisions.
