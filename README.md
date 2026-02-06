# Fraud Detection Analysis — Flask Application
[![CI](https://github.com/FORMATION-AI/fraud_detection_analysis/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FORMATION-AI/fraud_detection_analysis/actions/workflows/ci.yml)

End‑to‑end machine learning system that ingests digital banking transactions, engineers features, trains a fraud classifier, and exposes predictions through a Flask web UI (plus a Streamlit dashboard for interactive demos). This README explains how the project is structured, how to regenerate the ML artifacts, and how to run the web experience locally.

---

## Table of Contents
1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [Data & Artifacts Pipeline](#data--artifacts-pipeline)
6. [Running the Web Applications](#running-the-web-applications)
7. [Usage Guide](#usage-guide)
8. [Troubleshooting & Tips](#troubleshooting--tips)
9. [Roadmap / Next Steps](#roadmap--next-steps)

---

## Architecture

```
notebook/data/fraud_dataset_raw.csv
          │
          ▼
src/components/data_ingestion.py   ──► artifacts/data.csv, train.csv, test.csv
          │
          ▼
src/components/data_transformation.py
          ├─► artifacts/preprocessor.pkl (StandardScaler)
          ├─► artifacts/encoder.pkl (OneHotEncoder)
          ├─► artifacts/schema.json / feature_columns.json
          ▼
src/components/model_trainer.py    ──► artifacts/model.pkl (best classifier)
          │
          ▼
Flask app (main.py + templates) / Streamlit app (app.py)
```

Key highlights
- **Feature engineering** (hour of day, day of week, session duration, etc.) mirrors the exploratory notebook.
- **Model selection** compares Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting using ROC‑AUC.
- **Artifacts** are saved under `artifacts/` for both the Flask and Streamlit front ends.

---

## Project Structure

```
fraud_detection_analysis/
├── app.py                     # Streamlit dashboard
├── main.py                    # Flask entrypoint
├── notebook/                  # Exploratory notebooks & raw data
│   └── data/fraud_dataset_raw.csv
├── src/
│   ├── components/            # Ingestion, transformation, trainer modules
│   ├── pipeline/              # Predict pipeline & CustomData helper
│   ├── exception.py, logger.py, utils.py
├── templates/                 # Flask HTML templates
├── artifacts/                 # Generated models, preprocessors, schema
├── requirements.txt
├── pyproject.toml
└── README.md (this file)
```

---

## Prerequisites
- Python 3.11 (matches `.python-version`)
- PowerShell (instructions assume Windows) or bash/zsh on other OSes
- `uv` (optional) for fast dependency installs
- Git

---

## Environment Setup

```powershell
# 1. Clone and enter the project
git clone <repo-url>
cd fraud_detection_analysis

# 2. Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies (pip or uv)
pip install --upgrade pip
pip install -r requirements.txt
# or
uv pip install -r requirements.txt

# 4. (Optional) verify
python -c "import sklearn, streamlit, flask; print('OK')"
```

---

## Data & Artifacts Pipeline

1. **Ensure raw data exists**  
   `notebook/data/fraud_dataset_raw.csv` should contain the anonymized transaction history. Replace with your own dataset if needed (same schema).

2. **Run the ingestion → transformation → training pipeline**
   ```powershell
   # Virtualenv must be active
   python -m src.components.data_ingestion
   ```
   This command:
   - Reads the raw CSV, writes `artifacts/data.csv`, `train.csv`, `test.csv`
   - Fits preprocessing (scaler + encoder) and saves them to `artifacts/`
   - Trains the classifier and saves `artifacts/model.pkl`
   - Logs progress under `logs/` with timestamped files

3. **Artifacts produced**
   - `artifacts/preprocessor.pkl` – StandardScaler for numeric features
   - `artifacts/encoder.pkl` – OneHotEncoder for categorical (hour/day)
   - `artifacts/schema.json` – metadata for UI / prediction pipeline
   - `artifacts/feature_columns.json` – final feature order
   - `artifacts/model.pkl` – tuned classifier

Regenerate artifacts anytime data or code changes by rerunning the command above.

---

## Running the Web Applications

### Flask UI (form-based scoring)

```powershell
# From project root, venv activated
$env:FLASK_APP = "main.py"
$env:FLASK_ENV = "development"   # optional auto-reload
flask run
# or python main.py
```

Navigate to `http://127.0.0.1:5000/` and submit transaction details. The page reports whether the prediction executed and shows the model output (0 = NOT FRAUD, 1 = FRAUD).

### Streamlit dashboard (optional exploratory UI)

```powershell
streamlit run app.py
```

Streamlit will open at `http://localhost:8501`; enter values in the form to see fraud probabilities and labels.

---

## Usage Guide

1. **Prepare Data**  
   - Place updated raw transactions in `notebook/data/fraud_dataset_raw.csv`
   - Keep sensitive data out of Git (already ignored)

2. **Run Pipeline**  
   - `python -m src.components.data_ingestion`
   - Watch terminal logs or inspect `logs/*.log`

3. **Launch UI**  
   - Flask: `flask run` → use form
   - Streamlit: `streamlit run app.py`

4. **Sample POST values** (use in UI form):
   ```
   amount = 1500.50
   customer_age = 42
   minute_of_day = 875
   to_acc_volume = 3200
   session_duration = 65
   hour_of_day = 14
   day_of_week = 2
   ```

5. **Interpreting Results**  
   - Flask displays `"Prediction executed. Model output: X"`  
   - Streamlit shows both probability and label (threshold 0.2)

---

## Troubleshooting & Tips

| Issue | Fix |
| --- | --- |
| `ModuleNotFoundError` (dill, nbformat, etc.) | Re-run `pip install -r requirements.txt` inside the venv |
| `OneHotEncoder got unexpected keyword 'sparse'` | Upgrade scikit-learn (≥1.2) or keep `sparse_output=False` (already applied) |
| Flask page shows no result | Template now checks `results is not none`; ensure pipeline ran successfully and artifacts exist |
| Streamlit error `Missing artifact(s)` | Re-run the pipeline to regenerate `preprocessor.pkl`, `encoder.pkl`, `model.pkl`, `schema.json` |
| Warning `X has feature names...` | Prediction pipeline converts to numpy arrays (`values`) to avoid this |
| Need to inspect logs | `logs/<timestamp>/<timestamp>.log` created automatically |

---


## Face Enrollment Prototype (ArcFace)

Notebook: `notebooks/01_face_enrollment_prototype.ipynb`

Prereqs (once per env):
```powershell
%pip install -q insightface onnxruntime opencv-python numpy pandas pydantic matplotlib pyarrow
```

Data layout:
```
fraud_detection_analysis/
  data/
    enrollment_samples/
      user_123/
        img1.jpg
        img2.jpg
```

Run the notebook end-to-end on CPU. It enforces: exactly 1 face, min bbox width, min det score, and quality thresholds (blur, brightness, face area ratio). It writes artifacts to `artifacts/face_enrollment/` (JSON + parquet) and prints similarity thresholds (`T_low`, `T_high`) based on genuine vs impostor distributions.

pgvector schema migration: `infra/db/migrations/20260206_face_templates.sql`

## Roadmap / Next Steps
- Persist predictions & explanations to a database for audit
- Extend the pipeline with alternative models (LightGBM, CatBoost)
- Containerize (Docker) for reproducible deployment
- Add authentication & role-based access to the Flask UI
- Integrate alerting or webhook triggers when `prediction == FRAUD`

---

**Questions or contributions?**  
Open an issue / PR or reach out to maintainers. Happy experimenting with fraud analytics! 😊
