# 🚗 Smart Driver Monitoring Dashboard
**Internship Project — Weeks 4 & 5**
AI & Analytics for Smart Driver Monitoring

---

## Overview
An AI-powered dashboard that combines machine learning models built across Modules 1–6
into a single interactive Streamlit application for driver monitoring and analysis.
The app covers driver rating prediction, violation detection, passenger feedback sentiment
analysis, and document forgery detection.

---

## 📂 Datasets Used

### Real Datasets
| Dataset | File | Used In |
|---------|------|---------|
| Driver Behavior & Route Anomaly | `driver_behavior_cleaned.csv` | Violations detection (Tab 1), Sentiment behavior features (Tab 2) |
| Transportation & Logistics | `transportation_logistic_cleaned.csv` | Driver rating prediction (Tab 1) |
| Driving Behavior Sensor (Train) | `driving_behavior_train_cleaned.csv` | Violations sensor model (Module 5) |
| Driving Behavior Sensor (Test) | `driving_behavior_test_cleaned.csv` | Violations sensor model evaluation (Module 5) |
| Traffic Violations | `traffic_violations_cleaned.csv` | NLP topic modeling, violation analysis (Modules 4 & 5) |
| Delivery Truck Trips | `delivery_trucks_cleaned.csv` | EDA & feature engineering (Module 2) |

### Synthetic Datasets
| Dataset | File | Used In |
|---------|------|---------|
| Telemetry Samples | `data/telemetry_samples.csv` | Generated in Module 1 for experimentation |
| Passenger Feedback | `data/feedback.csv` | Sentiment analysis training & testing (Module 4, Tab 2) |

### Image Datasets
| Dataset | Used In |
|---------|---------|
| Driver License Images (`data/licenses/genuine/` — 20 images) | Forgery detection testing (Module 6, Tab 3) |
| Driver License Images (`data/licenses/forged/` — 20 images) | Forgery detection testing (Module 6, Tab 3) |
| Highway Traffic Videos | Visual analysis reference (Module 6) |
| YOLO Truck Datasets | CNN forgery model training (Module 6) |

---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create and Activate Environment
```bash
conda create -n driver-ai python=3.9 jupyter -y
conda activate driver-ai
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn \
            jupyterlab opencv-python pytesseract tensorflow \
            transformers sentencepiece spacy streamlit joblib scipy
```

### 4. Install Tesseract (OS-level)
- **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki
- **Mac:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

---

## 🚀 Running the App

```bash
conda activate driver-ai
cd demos/driver_demo
streamlit run streamlit_app.py
```

Then open your browser at: **http://localhost:8501**

---

## 📊 App Features

### Tab 1 — Driver Rating & Violations
- Upload `transportation_logistic_cleaned.csv` → predicted driver ratings
- Upload `driver_behavior_cleaned.csv` → violation flags per driver
- Models used: XGBoost (ratings), XGBoost with flags (violations)

### Tab 2 — Feedback Sentiment
- Type feedback manually → instant sentiment result with confidence score
- Upload `feedback.csv` → batch sentiment analysis with accuracy vs ground truth
- Model used: TF-IDF + Logistic Regression (trained with driver behavior features)
- Achieved: **100% accuracy** on feedback dataset

### Tab 3 — Document Forgery Check
- Upload a license/vehicle image → full forgery detection pipeline
- OCR plate extraction (Tesseract)
- Image forensics (blur + noise detection)
- CNN forgery classification (MobileNetV2)
- Ensemble scoring: OCR + CNN + Forensics combined

---

## 🤖 Models

| Module | Model | Trained On |
|--------|-------|-----------|
| 3 — Ratings | XGBoost Regressor | `transportation_logistic_cleaned.csv` |
| 4 — Sentiment | TF-IDF + Logistic Regression | `feedback.csv` + `driver_behavior_cleaned.csv` |
| 5 — Violations | XGBoost Classifier (with flags) | `driver_behavior_cleaned.csv` |
| 6 — Forgery | MobileNetV2 CNN + OCR + Forensics | YOLO Truck Datasets + `data/licenses/` |

---

## ⚠️ Known Limitations

- The forgery CNN was trained on truck imagery (YOLO dataset).
  Noise-based forgeries in license images may be classified as LOW risk.
  Retraining on license-specific data is recommended for production use.
- The sentiment model requires driver behavior features for best accuracy.
  Default mean values are used when behavior data is not provided.

---

## 📝 Deliverables

- ✅ Jupyter notebooks for Modules 1–6 (well-documented)
- ✅ Streamlit demo dashboard (Module 7)
- ✅ Trained models saved in `src/models/`
- ✅ GitHub repository with clear README

---

## 👤 Yodico, Jan Marcus C.