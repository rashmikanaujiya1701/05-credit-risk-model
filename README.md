# Credit Risk / Loan Default Prediction Model

Binary classification model predicting loan default probability from
applicant financials, with a permutation-importance explainability view and
a risk-tier report (Low/Medium/High) exportable to Excel.

## Why this project
Classic finance + ML crossover, pairing naturally with a churn model as a
second risk-scoring project — the kind of model valued in fintech/BFSI.

## Tech stack
- Python, pandas, numpy
- scikit-learn (GradientBoostingClassifier, permutation importance)
- openpyxl (Excel report)

Note: uses scikit-learn's GradientBoostingClassifier and permutation
importance instead of XGBoost + SHAP, so this runs fully offline. Swap in
`xgboost.XGBClassifier` and `shap.TreeExplainer` as drop-ins if those
packages are available in your environment — the pipeline structure
(ColumnTransformer + Pipeline) works with either.

## How to run
```bash
cd src
python3 generate_loan_data.py     # creates 2,000 synthetic loan applicants (~31% default rate)
python3 credit_risk_model.py      # trains, evaluates, and scores the portfolio
```

## Output
`output/credit_risk_report.xlsx` with four sheets:
- **Scored Applicants** — every applicant with predicted default probability + risk tier
- **Risk Tier Summary** — applicant counts and actual vs. predicted default rate per tier
- **Feature Importance** — permutation importance (ROC-AUC drop) per feature
- **Model Metrics** — ROC-AUC, accuracy, precision, recall on a held-out test set

## Sample results
ROC-AUC ~0.73 on holdout; risk tiers show clear separation — Low tier ~4%
actual default rate vs. High tier ~73%, showing the model ranks risk well
even before any threshold tuning.

## Using your own loan data
Replace `data/loan_applicants.csv` with your own (anonymized) portfolio,
matching the column names in `credit_risk_model.py`, and re-run.

## Suggested resume bullet
"Built a credit risk model achieving 0.73 ROC-AUC on held-out data, with
permutation-importance explainability and automatic risk-tier segmentation
showing an 18x default-rate spread between Low and High risk tiers."
