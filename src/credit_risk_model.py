"""
Credit Risk / Loan Default Prediction Model
-----------------------------------------------
Binary classifier predicting loan default probability from applicant
financials, with a feature-importance explainability view and a risk-tier
report exported to Excel.

Model choice: GradientBoostingClassifier (scikit-learn) instead of XGBoost so
this runs fully offline. Explainability uses the model's built-in feature
importances (permutation importance is also computed for a more robust view)
rather than SHAP -- swap in `shap.TreeExplainer(model)` for per-applicant
SHAP values if the shap package is available in your environment.
"""
import os
from typing import NamedTuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance


class ModelResult(NamedTuple):
    scored: pd.DataFrame
    tier_summary: pd.DataFrame
    importance_df: pd.DataFrame
    metrics_df: pd.DataFrame
    confusion_matrix: np.ndarray

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "loan_applicants.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "credit_risk_report.xlsx")

NUMERIC_COLS = ["age", "annual_income", "loan_amount", "credit_history_years",
                 "existing_loans", "credit_score", "debt_to_income_ratio", "late_payments_last_year"]
CATEGORICAL_COLS = ["employment_type"]
TARGET = "defaulted"


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ], remainder="passthrough")

    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    return Pipeline([("prep", preprocessor), ("model", model)])


def risk_tier(prob: float) -> str:
    if prob < 0.15:
        return "Low"
    elif prob < 0.40:
        return "Medium"
    return "High"


def run(data_csv: str = DATA_CSV):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(data_csv)

    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    test_probs = pipe.predict_proba(X_test)[:, 1]
    test_preds = pipe.predict(X_test)
    auc = roc_auc_score(y_test, test_probs)
    report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)

    # --- Explainability: permutation importance on the held-out set ---
    perm = permutation_importance(pipe, X_test, y_test, n_repeats=15, random_state=42, scoring="roc_auc")
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean.round(4),
        "importance_std": perm.importances_std.round(4),
    }).sort_values("importance_mean", ascending=False)

    # --- Score the full portfolio and assign risk tiers ---
    all_probs = pipe.predict_proba(X)[:, 1]
    scored = df.copy()
    scored["predicted_default_prob"] = all_probs.round(4)
    scored["risk_tier"] = scored["predicted_default_prob"].apply(risk_tier)

    tier_summary = scored.groupby("risk_tier").agg(
        applicants=("predicted_default_prob", "count"),
        avg_predicted_prob=("predicted_default_prob", "mean"),
        actual_default_rate=(TARGET, "mean"),
    ).round(3).reindex(["Low", "Medium", "High"])

    metrics_df = pd.DataFrame({
        "metric": ["ROC-AUC (holdout)", "Accuracy (holdout)", "Precision (default=1)", "Recall (default=1)"],
        "value": [
            round(auc, 4),
            round(report["accuracy"], 4),
            round(report["1"]["precision"], 4),
            round(report["1"]["recall"], 4),
        ],
    })

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        scored.to_excel(writer, sheet_name="Scored Applicants", index=False)
        tier_summary.to_excel(writer, sheet_name="Risk Tier Summary")
        importance_df.to_excel(writer, sheet_name="Feature Importance", index=False)
        metrics_df.to_excel(writer, sheet_name="Model Metrics", index=False)

    return ModelResult(scored, tier_summary, importance_df, metrics_df, cm)


if __name__ == "__main__":
    result = run()
    scored, tier_summary, importance_df, metrics_df, cm = result
    print("=== Model Metrics (holdout) ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Confusion Matrix (holdout) [rows=actual, cols=predicted] ===")
    print(cm)
    print("\n=== Feature Importance (permutation, ROC-AUC drop) ===")
    print(importance_df.to_string(index=False))
    print("\n=== Risk Tier Summary (full portfolio) ===")
    print(tier_summary.to_string())
    print(f"\nSaved report to {OUTPUT_XLSX}")
