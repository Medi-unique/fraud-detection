# Notebooks

Thin notebooks that call `src/` for reproducible analysis.

| Notebook | Purpose |
|----------|---------|
| `eda-fraud-data.ipynb` | Univariate/bivariate EDA, class imbalance, fraud by country |
| `eda-creditcard.ipynb` | Credit-card EDA, Amount/Time vs Class, imbalance |
| `feature-engineering.ipynb` | Geolocation + temporal/velocity features → `data/processed/` |
| `modeling.ipynb` | Stratified split, SMOTE, LR vs XGBoost, CV, selection |
| `shap-explainability.ipynb` | Feature importance, SHAP plots, business recommendations |

Run from the **project root** (or ensure `sys.path` includes the repo root) so `import src...` works.
