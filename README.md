# Fraud Detection — Adey Innovations Inc.

Unified fraud detection for **e-commerce** (`Fraud_Data.csv`) and **bank credit card** (`creditcard.csv`) transaction streams.

## Business context

False positives frustrate customers; false negatives cause direct financial loss. Models are evaluated with **AUC-PR**, **F1**, and **confusion matrices** — not accuracy alone — because both datasets are highly imbalanced.

## Project structure

```
fraud-detection/
├── .vscode/
├── .github/workflows/unittests.yml
├── data/
│   ├── raw/          # Place CSVs here (gitignored)
│   └── processed/    # Cleaned / feature-engineered outputs
├── notebooks/        # EDA, features, modeling, SHAP
├── src/              # Reusable pipeline code
├── tests/
├── models/           # Saved model artifacts
├── scripts/
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place the three datasets in `data/raw/`:

- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

4. From the project root, run notebooks or tests:

```bash
pytest tests/ -v
jupyter notebook notebooks/
```

## Pipeline overview

| Stage | Module | Description |
|-------|--------|-------------|
| Load | `src/data_loader.py` | Load CSVs with correct dtypes |
| Clean + geo | `src/preprocessing.py` | Dedup, IP→int, country range merge |
| Features | `src/features.py` | Time, velocity, signup lag |
| Transform | `src/pipeline.py` | Scale + one-hot encode |
| Imbalance | `src/imbalance.py` | SMOTE on **train only** |
| Models | `src/modeling.py` | Logistic Regression + XGBoost |
| Metrics | `src/evaluation.py` | AUC-PR, F1, confusion matrix |
| Explain | `src/explainability.py` | SHAP summary & force plots |

## Notebooks

1. `eda-fraud-data.ipynb` — E-commerce EDA + fraud by country  
2. `eda-creditcard.ipynb` — Bank card EDA + imbalance  
3. `feature-engineering.ipynb` — Features → `data/processed/`  
4. `modeling.ipynb` — LR vs XGBoost, CV, model selection  
5. `shap-explainability.ipynb` — Importance, SHAP, recommendations  

## Report

A written analysis with all figures lives in `reports/`:

- `reports/REPORT.md` — Medium-style article (markdown)
- `reports/REPORT.pdf` — same report as a styled PDF
- `reports/figures/` — generated plots and metrics tables

Regenerate both from the processed data and saved models:

```bash
python scripts/generate_report_figures.py   # rebuild all figures
python scripts/build_report_pdf.py          # rebuild REPORT.pdf
```

## Modeling choices

- **Baseline:** Logistic Regression (interpretable)  
- **Ensemble:** XGBoost (tuned `n_estimators`, `max_depth`)  
- **Resampling:** SMOTE on the training set only (preserves majority-class information vs. aggressive undersampling)  
- **Split / CV:** Stratified train-test + Stratified 5-Fold  

## License / course

KAIM Week 5–6 case study — Adey Innovations Inc. fraud detection.
