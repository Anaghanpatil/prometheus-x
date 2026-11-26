# ⚡ PROMETHEUS-X
### Open-Set Industrial Anomaly Detection with Localization, Severity & Distribution Shift Awareness

<p align="center">
  <strong>Making industrial visual inspection smarter, more explainable, and more reliable.</strong>
</p>

<p align="center">
  <a href="https://prometheus-x-biet.streamlit.app/">🚀 Live Demo</a> •
  <a href="https://github.com/Anaghanpatil/prometheus-x">💻 GitHub Repository</a>
</p>

---

## 🔍 Overview

**PROMETHEUS-X** is an AI-powered industrial visual inspection platform designed to identify potential product defects, localize suspicious regions, and support more informed quality-control decisions.

Traditional visual inspection can be time-consuming and dependent on human judgment. PROMETHEUS-X explores how deep learning can assist this process through anomaly detection, visual heatmaps, severity assessment, and distribution-shift awareness.

Built around the **PatchCore anomaly detection model**, the platform provides an interactive workflow for inspecting product images, reviewing previous inspections, and submitting feedback.

The goal is not just to detect anomalies, but to make inspection results easier to understand and review.

## ✨ Key Features

### 1. 🧠 AI-Powered Inspection

Upload an industrial product image and let the trained PatchCore model analyse it.

- Detects potential visual anomalies.
- Generates an anomaly score.
- Produces a heatmap to highlight suspicious regions.
- Creates an overlay combining the original image and anomaly localization.
- Displays an inspection result to support quality-control decisions.

### 2. 📍 Anomaly Localization

Knowing that a product may be defective is not enough. Understanding *where* the suspicious region is matters too.

PROMETHEUS-X generates visual anomaly maps and heatmap overlays to help users identify potentially defective areas.

### 3. ⚠️ Severity Assessment

The platform includes a severity-assessment component to help interpret detected anomalies and support inspection decisions.

### 4. 🌐 Distribution Shift Awareness

Industrial environments can change. Differences in image quality, lighting, and visual appearance may affect model reliability.

PROMETHEUS-X includes distribution-shift analysis to help identify conditions that may require additional attention during inspection.

### 5. 📋 Inspection History

Keep track of previously inspected images and their results.

- Review earlier inspection records.
- Revisit anomaly scores and predictions.
- Maintain a convenient inspection reference.
- Make it easier to compare inspection outcomes.

### 6. 💬 Feedback Module

Inspection systems should support continuous improvement.

The Feedback module provides a way for users to submit feedback about inspection results. This helps capture human observations and identify cases that may need further review.

---

## 🖥️ Application Modules

| Module | Purpose |
|---|---|
| **Inspection** | Upload images, run AI analysis, and view prediction results and localization heatmaps. |
| **History** | Review previously performed inspections. |
| **Feedback** | Submit feedback about inspection results. |

## 🏗️ System Architecture

```text
                 PROMETHEUS-X
                       |
              Streamlit Interface
                       |
        +--------------+--------------+
        |              |              |
    Inspection       History       Feedback
        |
    Image Upload
        |
    Image Processing
        |
    PatchCore Model
        |
    Anomaly Prediction
        |
    +---+-------------------+
    |                       |
 Anomaly Score         Anomaly Map
    |                       |
    +-----------+-----------+
                |
       Decision & Analysis
                |
     Severity / Shift Analysis
                |
        Inspection Results
```

## 🤖 AI Model

**Model:** PatchCore  
**Framework:** Anomalib  
**Dataset:** MVTec AD – Bottle category  
**Inference:** CPU-based inference

PatchCore is an anomaly detection approach that learns visual representations of normal samples and identifies deviations during inspection.

PROMETHEUS-X uses a trained PatchCore checkpoint to analyse uploaded images and generate anomaly predictions and localization maps.

> **Important:** The current trained model is based on the MVTec AD Bottle category. Its predictions should not be interpreted as validated defect detection across every industrial product category.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Deep Learning | PyTorch |
| Anomaly Detection | Anomalib, PatchCore |
| Image Processing | OpenCV, Pillow |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Machine Learning Utilities | Scikit-learn |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## 📂 Project Structure

```text
prometheus-x/
│
├── app/
│   ├── streamlit_app.py
│   ├── components.py
│   ├── styles.py
│   └── utils.py
│
├── backend/
│   └── ai/
│       ├── patchcore_service.py
│       ├── patchcore_model.ckpt
│       └── outputs/
│
├── src/
│   ├── decision.py
│   ├── calibration.py
│   ├── severity.py
│   └── shift.py
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

Follow these steps to run PROMETHEUS-X locally.

### Prerequisites

- Python 3.11
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Anaghanpatil/prometheus-x.git
```

### 2. Navigate to the Project

```bash
cd prometheus-x
```

### 3. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser.

## 🔄 How It Works

1. Open the PROMETHEUS-X application.
2. Navigate to the Inspection module.
3. Upload an industrial product image.
4. Run the AI-powered inspection.
5. Review the prediction and anomaly score.
6. Examine the heatmap and localization overlay.
7. Review previous results in History.
8. Submit observations through Feedback.

## 🎯 Problem Statement

**A06 – Open-Set Industrial Anomaly Detection with Localization, Severity & Distribution Shift**

PROMETHEUS-X explores a practical approach to industrial anomaly inspection by combining visual anomaly detection with localization, severity assessment, and distribution-shift awareness.

It aims to make AI-assisted inspection more interpretable and provide useful information for human quality-control decisions.

## 🔮 Future Scope

- Expand training and evaluation across additional industrial product categories.
- Improve robustness under changing lighting and imaging conditions.
- Introduce more comprehensive model calibration and uncertainty estimation.
- Develop feedback-driven model improvement workflows.
- Add richer inspection analytics and reporting.
- Evaluate performance on unseen defect types and real-world industrial data.

## 👩‍💻 Contributors

Developed as a hackathon project by the PROMETHEUS-X team at **Bapuji Institute of Engineering and Technology (BIET)**.

## ⚠️ Disclaimer

PROMETHEUS-X is a prototype developed for research and demonstration purposes. Its predictions are intended to assist human inspection, not replace professional industrial quality-control procedures. Performance should be independently validated before real-world deployment.

---

<p align="center">
  <strong>PROMETHEUS-X</strong><br/>
  <em>Detect. Localize. Analyse. Improve.</em>
</p>
