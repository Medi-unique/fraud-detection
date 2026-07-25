"""Generate all figures used in reports/REPORT.md from the processed data and saved models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from src.evaluation import compute_metrics, metrics_table, plot_confusion_matrix, plot_pr_curve
from src.explainability import (
    find_example_indices,
    get_builtin_importance,
    plot_builtin_importance,
    top_shap_drivers,
)
from src.features import select_model_features_creditcard, select_model_features_fraud
from src.modeling import load_model, make_logistic_regression, stratified_split
from src.imbalance import resample_train
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid", context="talk")
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name: str):
    path = FIG / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", path.name)


# ----------------------------------------------------------------------------
# 1. E-commerce EDA figures
# ----------------------------------------------------------------------------
fraud = pd.read_csv(PROC / "fraud_features.csv", parse_dates=["signup_time", "purchase_time"])
print("fraud:", fraud.shape)

fig, ax = plt.subplots(figsize=(6, 4.5))
counts = fraud["class"].value_counts().sort_index()
ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["#4c72b0", "#c44e52"])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}\n({v / len(fraud):.1%})", ha="center", va="bottom", fontsize=12)
ax.set_ylim(0, counts.max() * 1.2)
ax.set_title("E-commerce class imbalance")
ax.set_ylabel("Transactions")
save(fig, "fraud_class_imbalance.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(fraud["purchase_value"], bins=40, ax=axes[0], color="#4c72b0")
axes[0].set_title("Purchase value ($)")
sns.histplot(fraud["age"], bins=30, ax=axes[1], color="#55a868")
axes[1].set_title("Customer age")
fig.tight_layout()
save(fig, "fraud_univariate.png")

# time_since_signup vs fraud — the key behavioral signal
fig, ax = plt.subplots(figsize=(9, 5))
hours = fraud["time_since_signup"] / 3600.0
sns.histplot(
    x=hours, hue=fraud["class"].map({0: "Legit", 1: "Fraud"}),
    bins=60, element="step", stat="density", common_norm=False, ax=ax,
)
ax.set_xlabel("Hours between signup and purchase")
ax.set_title("Time-since-signup by class")
save(fig, "fraud_time_since_signup.png")

by_country = (
    fraud.groupby("country")
    .agg(n=("class", "size"), fraud_rate=("class", "mean"))
    .query("n >= 200")
    .sort_values("fraud_rate", ascending=False)
    .head(12)
)
fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(x=by_country.index, y=by_country["fraud_rate"], ax=ax, color="#c44e52")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
ax.set_ylabel("Fraud rate")
ax.set_title("Fraud rate by country (n >= 200)")
save(fig, "fraud_by_country.png")

# ----------------------------------------------------------------------------
# 2. Credit-card EDA figures
# ----------------------------------------------------------------------------
cc = pd.read_csv(PROC / "creditcard_clean.csv")
print("cc:", cc.shape)

fig, ax = plt.subplots(figsize=(6, 4.5))
counts = cc["Class"].value_counts().sort_index()
ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["#4c72b0", "#c44e52"])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}\n({v / len(cc):.3%})", ha="center", va="bottom", fontsize=12)
ax.set_yscale("log")
ax.set_title("Credit-card class imbalance (log scale)")
ax.set_ylabel("Transactions (log)")
save(fig, "cc_class_imbalance.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=cc, x="Class", y="Amount", showfliers=False, ax=ax,
            hue="Class", palette=["#4c72b0", "#c44e52"], legend=False)
ax.set_xticklabels(["Legit", "Fraud"])
ax.set_title("Transaction amount by class (outliers hidden)")
save(fig, "cc_amount_by_class.png")

# ----------------------------------------------------------------------------
# 3. E-commerce modeling figures (recreate deterministic split, saved models)
# ----------------------------------------------------------------------------
X_f, y_f = select_model_features_fraud(fraud)
X_train, X_test, y_train, y_test = stratified_split(X_f, y_f)

pre = load_model(MODELS / "fraud_preprocessor.joblib")
feature_names = load_model(MODELS / "fraud_feature_names.joblib")
xgb = load_model(MODELS / "fraud_best_model.joblib")

X_train_t = pd.DataFrame(pre.transform(X_train), columns=feature_names)
X_test_t = pd.DataFrame(pre.transform(X_test), columns=feature_names)

# Re-fit LR on SMOTE'd train (same as pipeline) for the comparison figures
X_res, y_res, _ = resample_train(X_train_t, y_train, method="smote")
lr = make_logistic_regression()
lr.fit(X_res, y_res)

lr_m = compute_metrics(y_test, lr.predict(X_test_t), lr.predict_proba(X_test_t)[:, 1], name="Logistic Regression")
xgb_m = compute_metrics(y_test, xgb.predict(X_test_t), xgb.predict_proba(X_test_t)[:, 1], name="XGBoost")
table = metrics_table([lr_m, xgb_m])
table.to_csv(FIG / "fraud_metrics.csv", index=False)
print(table.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot_confusion_matrix(lr_m["confusion_matrix"], title="Logistic Regression", ax=axes[0])
plot_confusion_matrix(xgb_m["confusion_matrix"], title="XGBoost", ax=axes[1])
fig.suptitle("E-commerce — confusion matrices (test set)", y=1.03)
fig.tight_layout()
save(fig, "fraud_confusion_matrices.png")

fig, ax = plt.subplots(figsize=(8, 5.5))
plot_pr_curve(y_test, lr.predict_proba(X_test_t)[:, 1], label="Logistic Regression", ax=ax)
plot_pr_curve(y_test, xgb.predict_proba(X_test_t)[:, 1], label="XGBoost", ax=ax)
ax.set_title("E-commerce — precision-recall curves")
save(fig, "fraud_pr_curves.png")

# ----------------------------------------------------------------------------
# 4. Feature importance + SHAP
# ----------------------------------------------------------------------------
imp = get_builtin_importance(xgb, feature_names, top_n=10)
fig, ax = plt.subplots(figsize=(9, 5.5))
plot_builtin_importance(imp, title="XGBoost — top 10 built-in importances", ax=ax)
save(fig, "fraud_feature_importance.png")

rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_test_t), size=min(2000, len(X_test_t)), replace=False)
X_sample = X_test_t.iloc[sample_idx]

explainer = shap.TreeExplainer(xgb)
sv_sample = explainer.shap_values(X_sample)
if isinstance(sv_sample, list):
    sv_sample = sv_sample[1]

plt.figure(figsize=(10, 7))
shap.summary_plot(sv_sample, X_sample, max_display=12, show=False)
plt.title("SHAP summary — e-commerce fraud model", fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "fraud_shap_summary.png", dpi=130, bbox_inches="tight")
plt.close()
print("saved fraud_shap_summary.png")

drivers = top_shap_drivers(sv_sample, feature_names, top_n=5)
drivers.to_csv(FIG / "fraud_shap_drivers.csv", index=False)
print(drivers.to_string(index=False))

# Force plots for TP / FP / FN
y_pred = xgb.predict(X_test_t)
sv_test = explainer.shap_values(X_test_t)
if isinstance(sv_test, list):
    sv_test = sv_test[1]
expected = explainer.expected_value
if isinstance(expected, (list, np.ndarray)):
    expected = np.atleast_1d(expected)[-1]

for kind, label in [("tp", "true_positive"), ("fp", "false_positive"), ("fn", "false_negative")]:
    idx = find_example_indices(y_test, y_pred, kind)
    if len(idx) == 0:
        print("no example for", kind)
        continue
    i = int(idx[0])
    shap.force_plot(
        expected, sv_test[i], X_test_t.iloc[i],
        matplotlib=True, show=False, figsize=(22, 4), text_rotation=30,
    )
    plt.title(f"SHAP force plot — {label.replace('_', ' ')}", fontsize=13)
    plt.savefig(FIG / f"fraud_force_{label}.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved fraud_force_{label}.png (index {i})")

# ----------------------------------------------------------------------------
# 5. Credit-card modeling figures (same 30% sample as the smoke test)
# ----------------------------------------------------------------------------
X_c, y_c = select_model_features_creditcard(cc)
X_c, _, y_c, _ = train_test_split(X_c, y_c, train_size=0.3, stratify=y_c, random_state=42)
Xc_train, Xc_test, yc_train, yc_test = stratified_split(X_c, y_c)

cc_pre = load_model(MODELS / "creditcard_preprocessor.joblib")
cc_names = load_model(MODELS / "creditcard_feature_names.joblib")
cc_xgb = load_model(MODELS / "creditcard_best_model.joblib")

Xc_train_t = pd.DataFrame(cc_pre.transform(Xc_train), columns=cc_names)
Xc_test_t = pd.DataFrame(cc_pre.transform(Xc_test), columns=cc_names)
Xc_res, yc_res, _ = resample_train(Xc_train_t, yc_train, method="smote")
cc_lr = make_logistic_regression()
cc_lr.fit(Xc_res, yc_res)

cc_lr_m = compute_metrics(yc_test, cc_lr.predict(Xc_test_t), cc_lr.predict_proba(Xc_test_t)[:, 1], name="Logistic Regression")
cc_xgb_m = compute_metrics(yc_test, cc_xgb.predict(Xc_test_t), cc_xgb.predict_proba(Xc_test_t)[:, 1], name="XGBoost")
cc_table = metrics_table([cc_lr_m, cc_xgb_m])
cc_table.to_csv(FIG / "cc_metrics.csv", index=False)
print(cc_table.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot_confusion_matrix(cc_lr_m["confusion_matrix"], title="Logistic Regression", ax=axes[0])
plot_confusion_matrix(cc_xgb_m["confusion_matrix"], title="XGBoost", ax=axes[1])
fig.suptitle("Credit card — confusion matrices (test set)", y=1.03)
fig.tight_layout()
save(fig, "cc_confusion_matrices.png")

fig, ax = plt.subplots(figsize=(8, 5.5))
plot_pr_curve(yc_test, cc_lr.predict_proba(Xc_test_t)[:, 1], label="Logistic Regression", ax=ax)
plot_pr_curve(yc_test, cc_xgb.predict_proba(Xc_test_t)[:, 1], label="XGBoost", ax=ax)
ax.set_title("Credit card — precision-recall curves")
save(fig, "cc_pr_curves.png")

print("ALL FIGURES DONE")
