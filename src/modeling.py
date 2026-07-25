"""Train/evaluate Logistic Regression and XGBoost for fraud detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from xgboost import XGBClassifier

from src.evaluation import compute_metrics, metrics_table
from src.imbalance import resample_train
from src.pipeline import (
    FRAUD_CATEGORICAL,
    FRAUD_NUMERICAL,
    build_model_pipeline,
    build_preprocessor,
    infer_column_types,
)


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Stratified train-test split preserving class distribution."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def make_logistic_regression(**kwargs) -> LogisticRegression:
    defaults = dict(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


def make_xgboost(
    scale_pos_weight: float | None = None,
    **kwargs,
) -> XGBClassifier:
    defaults = dict(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="aucpr",
        n_jobs=-1,
    )
    if scale_pos_weight is not None:
        defaults["scale_pos_weight"] = scale_pos_weight
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def train_baseline(
    X_train,
    y_train,
    categorical: list[str] | None = None,
    numerical: list[str] | None = None,
) -> Any:
    """Train Logistic Regression pipeline (preprocessor + model)."""
    pipe = build_model_pipeline(
        make_logistic_regression(),
        X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train),
        categorical=categorical,
        numerical=numerical,
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(
    X_train,
    y_train,
    categorical: list[str] | None = None,
    numerical: list[str] | None = None,
    tune: bool = True,
    cv: int = 3,
) -> Any:
    """
    Train XGBoost with optional GridSearch over n_estimators and max_depth.

    When categorical columns are present, fits via a Pipeline. For pure-numeric
    data (creditcard), uses XGBClassifier directly after optional preprocess.
    """
    X_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    _, cat_cols = infer_column_types(
        X_df, categorical=categorical, numerical=numerical
    )

    pos = int((np.asarray(y_train) == 1).sum())
    neg = int((np.asarray(y_train) == 0).sum())
    spw = neg / max(pos, 1)

    base = make_xgboost(scale_pos_weight=spw)

    if cat_cols:
        pipe = build_model_pipeline(
            base, X_df, categorical=categorical, numerical=numerical
        )
        if tune:
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5, 7],
            }
            scorer = make_scorer(average_precision_score, needs_proba=True)
            # GridSearch needs predict_proba scoring via average_precision
            search = GridSearchCV(
                pipe,
                param_grid,
                scoring="average_precision",
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            return search.best_estimator_
        pipe.fit(X_train, y_train)
        return pipe

    # Numeric-only path
    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
        }
        search = GridSearchCV(
            base,
            param_grid,
            scoring="average_precision",
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_
    base.fit(X_train, y_train)
    return base


def cross_validate_model(
    estimator,
    X,
    y,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Stratified K-Fold CV; report mean/std of AUC-PR and F1."""
    scoring = {
        "auc_pr": "average_precision",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    rows = []
    for key in scoring:
        scores = results[f"test_{key}"]
        rows.append(
            {
                "metric": key,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "folds": scores.tolist(),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(model, X_test, y_test, name: str = "model") -> dict:
    """Predict and compute metrics."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)
    return compute_metrics(y_test, y_pred, y_prob, name=name)


def save_model(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path):
    return joblib.load(path)


def run_fraud_modeling(
    X: pd.DataFrame,
    y: pd.Series,
    resample_method: str = "smote",
    tune: bool = True,
) -> dict:
    """
    End-to-end Fraud_Data modeling: split → SMOTE(train) → LR + XGB → compare.
    """
    # For SMOTE we need numeric matrix; preprocess first then resample
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    pre = build_preprocessor(
        X_train, categorical=FRAUD_CATEGORICAL, numerical=FRAUD_NUMERICAL
    )
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    feature_names = list(pre.get_feature_names_out())

    X_train_df = pd.DataFrame(X_train_t, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_t, columns=feature_names)

    X_res, y_res, resample_report = resample_train(
        X_train_df, y_train, method=resample_method  # type: ignore[arg-type]
    )

    lr = make_logistic_regression()
    lr.fit(X_res, y_res)
    lr_metrics = evaluate_model(lr, X_test_df, y_test, name="LogisticRegression")

    xgb = train_xgboost(X_res, y_res, categorical=[], numerical=feature_names, tune=tune)
    xgb_metrics = evaluate_model(xgb, X_test_df, y_test, name="XGBoost")

    comparison = metrics_table([lr_metrics, xgb_metrics])

    # Pick best by AUC-PR then F1
    best_name = (
        "XGBoost"
        if xgb_metrics["auc_pr"] >= lr_metrics["auc_pr"]
        else "LogisticRegression"
    )
    best_model = xgb if best_name == "XGBoost" else lr

    return {
        "preprocessor": pre,
        "X_test": X_test_df,
        "y_test": y_test,
        "models": {"LogisticRegression": lr, "XGBoost": xgb},
        "metrics": {"LogisticRegression": lr_metrics, "XGBoost": xgb_metrics},
        "comparison": comparison,
        "best_name": best_name,
        "best_model": best_model,
        "resample_report": resample_report,
        "feature_names": feature_names,
    }


def run_creditcard_modeling(
    X: pd.DataFrame,
    y: pd.Series,
    resample_method: str = "smote",
    tune: bool = True,
) -> dict:
    """End-to-end creditcard modeling (numeric features only)."""
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    pre = build_preprocessor(X_train, categorical=[], numerical=list(X.columns))
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    feature_names = list(pre.get_feature_names_out())
    X_train_df = pd.DataFrame(X_train_t, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_t, columns=feature_names)

    X_res, y_res, resample_report = resample_train(
        X_train_df, y_train, method=resample_method  # type: ignore[arg-type]
    )

    lr = make_logistic_regression()
    lr.fit(X_res, y_res)
    lr_metrics = evaluate_model(lr, X_test_df, y_test, name="LogisticRegression")

    xgb = train_xgboost(X_res, y_res, categorical=[], numerical=feature_names, tune=tune)
    xgb_metrics = evaluate_model(xgb, X_test_df, y_test, name="XGBoost")

    comparison = metrics_table([lr_metrics, xgb_metrics])
    best_name = (
        "XGBoost"
        if xgb_metrics["auc_pr"] >= lr_metrics["auc_pr"]
        else "LogisticRegression"
    )
    best_model = xgb if best_name == "XGBoost" else lr

    return {
        "preprocessor": pre,
        "X_test": X_test_df,
        "y_test": y_test,
        "models": {"LogisticRegression": lr, "XGBoost": xgb},
        "metrics": {"LogisticRegression": lr_metrics, "XGBoost": xgb_metrics},
        "comparison": comparison,
        "best_name": best_name,
        "best_model": best_model,
        "resample_report": resample_report,
        "feature_names": feature_names,
    }
