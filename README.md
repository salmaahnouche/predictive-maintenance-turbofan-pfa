# Predictive Maintenance Turbofan — Edge-to-Cloud PFA

Edge-to-cloud predictive maintenance system for turbofan engines, combining
real-time embedded inference on an ESP32 microcontroller with a cloud-based
ML pipeline for Remaining Useful Life (RUL) estimation and fault
classification.

## Overview

- **Dataset**: NASA C-MAPSS (FD003 subset, 97 test engines)
- **Edge**: ESP32 running a quantized CNN model (TFLite Full INT8, 53.0 KB) for
  on-device RUL inference in milliseconds
- **Cloud**: FastAPI + SQLite backend, Streamlit monitoring dashboard
- **ML pipeline**: CNN for RUL regression (deployed), benchmarked against
  Linear Regression, Random Forest and LSTM; hybrid GMM + Random Forest for
  unsupervised fault classification (HPC Degradation vs. Fan Degradation)

## Architecture

```
[ESP32: sensors + CNN (TFLite INT8)] → JSON payload → [FastAPI + SQLite] → [Streamlit dashboard]
```

The ESP32 performs on-device RUL inference from raw sensor readings, then
sends compact JSON results to a FastAPI server, which persists them to
SQLite and exposes fault classification. Results are visualized in real
time on a Streamlit dashboard.

## Results

### RUL prediction — model benchmark (FD003, 97 engines)

| Model | RMSE (cycles) | MAE (cycles) | Classification accuracy |
|---|---|---|---|
| LSTM | 32.98 | 20.49 | 91.8% |
| Random Forest | 48.96 | 35.60 | 75.3% |
| Linear Regression (baseline) | 53.27 | 43.39 | 66.0% |
| **CNN (deployed on ESP32)** | 71.44 | 50.55 | 75.3% |

The CNN was selected for embedded deployment over the higher-accuracy LSTM
because its architecture (causal convolutions, batch norm, global average
pooling) is natively compatible with TFLite Micro's INT8 quantization
kernels, while LSTM's recurrent gates are not — a hardware constraint that
determined the final model choice despite the accuracy trade-off. No model
ever misclassifies a Critical engine as Healthy.

### Fault classification — hybrid GMM + Random Forest

- Unsupervised GMM clustering on 5 differential features (no ground-truth
  labels available in the dataset) reached a **97.5% average assignment
  confidence** and a 0.337 silhouette score
- The supervised Random Forest classifier trained on GMM pseudo-labels
  achieved **96.8% ± 4.2% accuracy** (5-fold stratified cross-validation)
  and **99.0% precision** on the full training set

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| HPC Degradation | 0.986 | 1.000 | 0.993 | 68 |
| Fan Degradation | 1.000 | 0.966 | 0.982 | 29 |

### End-to-end system

The embedded CNN classifies engine state correctly in **70.1%** of cases
directly on the ESP32; the cloud-side Random Forest reaches **99.0%
precision** on fault-type classification (HPC vs. Fan), demonstrating that
unsupervised pseudo-labeling is a viable strategy when field labels are
unavailable in industrial settings.

## Repository structure

```
├── edge/       ESP32 firmware + embedded model artifacts (TFLite, C header)
├── cloud/      FastAPI backend, SQLite persistence
│   └── dashboard/   Streamlit monitoring dashboard
├── ml/         Training pipeline, trained models (CNN, GMM, Random Forest), scalers
├── docs/       Reference documentation
└── README.md
```

## Getting started

1. Download the [NASA C-MAPSS dataset](https://www.nasa.gov/intelligent-systems-division/#turbofan) (FD003 subset)
2. Install dependencies: `pip install -r requirements.txt`
3. Train the pipeline (optional — trained artifacts are already included in `ml/`):
   `python ml/train_pipeline.py`
4. Run the cloud backend: `uvicorn cloud.API_server:app --reload`
5. Launch the dashboard: `streamlit run cloud/dashboard/Dine.py`
6. Flash `edge/` firmware to an ESP32 (Arduino IDE / PlatformIO)

## Tech stack

Python · TensorFlow / TFLite Micro · scikit-learn · FastAPI · SQLite ·
Streamlit · ESP32 (C++) · pandas / NumPy

## Context

Developed as a final-year project (PFA) at ENSEM Casablanca, Electrical
Engineering — Industrial Process Digitalization track.
