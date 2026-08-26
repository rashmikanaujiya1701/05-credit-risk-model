"""Generates a synthetic loan-applicant dataset with a realistic default-risk
relationship baked in, so the credit risk model has something meaningful to learn.
Swap this out for a real (anonymized) loan portfolio export with the same columns."""
import os
import numpy as np
import pandas as pd

OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "loan_applicants.csv")

def run(n=2000, seed=11):
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 65, n)
    annual_income = rng.normal(600000, 220000, n).clip(150000, None)
    loan_amount = rng.normal(300000, 150000, n).clip(20000, None)
    credit_history_years = rng.integers(0, 25, n)
    existing_loans = rng.integers(0, 5, n)
    credit_score = rng.normal(680, 80, n).clip(300, 900)
    debt_to_income = (loan_amount * 0.12 + existing_loans * 40000) / annual_income
    employment_type = rng.choice(["Salaried", "Self-Employed", "Business Owner"], n, p=[0.55, 0.25, 0.20])
    late_payments_last_year = rng.poisson(0.6, n)

    # Build a latent default-risk score from the features (higher = riskier), then add noise
    risk_score = (
        -2.2
        - 0.010 * (credit_score - 680)
        + 4.5 * debt_to_income
        + 0.35 * late_payments_last_year
        + 0.05 * existing_loans
        - 0.03 * credit_history_years
        - 0.000002 * (annual_income - 600000)
        + rng.normal(0, 1.0, n)
    )
    default_prob = 1 / (1 + np.exp(-risk_score))
    defaulted = (rng.random(n) < default_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income.round(0),
        "loan_amount": loan_amount.round(0),
        "credit_history_years": credit_history_years,
        "existing_loans": existing_loans,
        "credit_score": credit_score.round(0),
        "debt_to_income_ratio": debt_to_income.round(3),
        "employment_type": employment_type,
        "late_payments_last_year": late_payments_last_year,
        "defaulted": defaulted,
    })
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Generated {len(df)} loan applicants ({df['defaulted'].mean()*100:.1f}% default rate) -> {OUT_CSV}")

if __name__ == "__main__":
    run()
