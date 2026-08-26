import os
import sys
import io
from typing import NamedTuple
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))
from credit_risk_model import build_pipeline, risk_tier, NUMERIC_COLS, CATEGORICAL_COLS, TARGET, DATA_CSV

st.set_page_config(page_title="Credit Risk Model", layout="wide")
st.title("📊 Credit Risk / Loan Default Prediction")


class TrainResult(NamedTuple):
    pipe: object
    scored: pd.DataFrame
    tier_summary: pd.DataFrame
    importance_df: pd.DataFrame
    metrics: dict
    cm: np.ndarray


@st.cache_resource
def train_model() -> TrainResult:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    from sklearn.inspection import permutation_importance

    df = pd.read_csv(DATA_CSV)
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    test_probs = pipe.predict_proba(X_test)[:, 1]
    test_preds = pipe.predict(X_test)
    auc = roc_auc_score(y_test, test_probs)
    report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)

    perm = permutation_importance(
        pipe, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc"
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": perm.importances_mean.round(4),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    all_probs = pipe.predict_proba(X)[:, 1]
    scored = df.copy()
    scored["predicted_default_prob"] = all_probs.round(4)
    scored["risk_tier"] = scored["predicted_default_prob"].apply(risk_tier)

    tier_summary = scored.groupby("risk_tier").agg(
        applicants=("predicted_default_prob", "count"),
        avg_predicted_prob=("predicted_default_prob", "mean"),
        actual_default_rate=(TARGET, "mean"),
    ).round(3).reindex(["Low", "Medium", "High"])

    metrics = {
        "ROC-AUC": round(auc, 4),
        "Accuracy": round(report["accuracy"], 4),
        "Precision (default)": round(report["1"]["precision"], 4),
        "Recall (default)": round(report["1"]["recall"], 4),
    }
    return TrainResult(pipe, scored, tier_summary, importance_df, metrics, cm)


result = train_model()
pipe, scored, tier_summary, importance_df, metrics, cm = result

# ── Sidebar: single applicant prediction ──────────────────────────────────────
with st.sidebar:
    st.header("🔍 Predict Single Applicant")
    age = st.slider("Age", 21, 65, 35)
    annual_income = st.number_input("Annual Income (₹)", 150000, 5000000, 600000, step=10000)
    loan_amount = st.number_input("Loan Amount (₹)", 20000, 2000000, 300000, step=10000)
    credit_score = st.slider("Credit Score", 300, 900, 680)
    credit_history_years = st.slider("Credit History (years)", 0, 25, 5)
    existing_loans = st.slider("Existing Loans", 0, 5, 1)
    late_payments = st.slider("Late Payments (last year)", 0, 10, 0)
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business Owner"])

    dti = (loan_amount * 0.12 + existing_loans * 40000) / annual_income

    applicant = pd.DataFrame([{
        "employment_type": employment_type,
        "age": age,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_history_years": credit_history_years,
        "existing_loans": existing_loans,
        "credit_score": credit_score,
        "debt_to_income_ratio": round(dti, 3),
        "late_payments_last_year": late_payments,
    }])

    prob = pipe.predict_proba(applicant)[0, 1]
    tier = risk_tier(prob)
    tier_color = {"Low": "green", "Medium": "orange", "High": "red"}[tier]
    tier_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[tier]

    st.metric("Default Probability", f"{prob:.1%}")
    st.markdown(f"**Risk Tier:** :{tier_color}[{tier_emoji} {tier}]")
    st.caption(f"Debt-to-Income Ratio: {dti:.3f}")

    st.divider()
    # Risk gauge bar
    st.markdown("**Risk Level**")
    gauge_color = tier_color
    st.progress(float(prob), text=f"{prob:.1%} default probability")

    st.divider()
    st.markdown("**💡 Key Risk Factors**")
    if credit_score < 600:
        st.warning("Low credit score increases default risk")
    if dti > 0.5:
        st.warning("High debt-to-income ratio")
    if late_payments >= 2:
        st.warning("Multiple late payments detected")
    if credit_score >= 700 and dti < 0.3 and late_payments == 0:
        st.success("Strong credit profile")

# ── Main dashboard ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
for col, (k, v) in zip([col1, col2, col3, col4], metrics.items()):
    col.metric(k, v)

st.divider()

# ── Row 1: Tier summary + Default rate bar chart ───────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Risk Tier Summary")
    tier_counts = tier_summary["applicants"].to_dict()
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("🟢 Low", tier_counts.get("Low", 0))
    tc2.metric("🟡 Medium", tier_counts.get("Medium", 0))
    tc3.metric("🔴 High", tier_counts.get("High", 0))
    st.dataframe(tier_summary, width="stretch")

with right:
    st.subheader("Default Rate by Risk Tier")
    fig, ax = plt.subplots(figsize=(5, 3))
    tiers = ["Low", "Medium", "High"]
    tier_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    vals = [tier_summary.loc[t, "actual_default_rate"] for t in tiers]
    ax.bar(tiers, vals, color=tier_colors)
    ax.set_ylabel("Actual Default Rate")
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Row 2: Feature importance + Confusion matrix ──────────────────────────────
left2, right2 = st.columns(2)

with left2:
    st.subheader("Feature Importance (Permutation, ROC-AUC drop)")
    feat_reversed = importance_df.iloc[::-1]
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.barh(feat_reversed["feature"], feat_reversed["importance"], color="#3498db")
    ax2.set_xlabel("Mean ROC-AUC Drop")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

with right2:
    st.subheader("Confusion Matrix (Holdout)")
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    im = ax3.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax3)
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["No Default", "Default"])
    ax3.set_yticklabels(["No Default", "Default"])
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
    ax3.set_title("Confusion Matrix")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

st.divider()

# ── Row 3: Probability distribution ───────────────────────────────────────────
st.subheader("Default Probability Distribution")
fig4, ax4 = plt.subplots(figsize=(10, 3))
for t, c in zip(["Low", "Medium", "High"], ["#2ecc71", "#f39c12", "#e74c3c"]):
    subset = scored[scored["risk_tier"] == t]["predicted_default_prob"]
    ax4.hist(subset, bins=30, color=c, alpha=0.7, label=t)
ax4.axvline(0.15, color="gray", linestyle="--", linewidth=1, label="Low/Med threshold")
ax4.axvline(0.40, color="black", linestyle="--", linewidth=1, label="Med/High threshold")
ax4.set_xlabel("Predicted Default Probability")
ax4.set_ylabel("Count")
ax4.legend()
plt.tight_layout()
st.pyplot(fig4)
plt.close()

st.divider()

# ── Scored applicants table ────────────────────────────────────────────────────
st.subheader("Scored Applicants")

filter_col, sort_col = st.columns([2, 2])
with filter_col:
    tier_filter = st.multiselect(
        "Filter by Risk Tier", ["Low", "Medium", "High"], default=["Low", "Medium", "High"]
    )
with sort_col:
    sort_by = st.selectbox("Sort by", ["predicted_default_prob", "credit_score", "annual_income", "loan_amount"])

filtered = scored[scored["risk_tier"].isin(tier_filter)].sort_values(sort_by, ascending=False)


def highlight_tier(val):
    return {"Low": "color: green", "Medium": "color: darkorange", "High": "color: red"}.get(val, "")


st.dataframe(
    filtered.style.map(highlight_tier, subset=["risk_tier"]),
    width="stretch",
    height=400,
)
st.caption(f"Showing {len(filtered):,} of {len(scored):,} applicants")

# Download button
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Filtered Results as CSV",
    data=csv_bytes,
    file_name="scored_applicants.csv",
    mime="text/csv",
)
