"""Generate a synthetic student dataset (replace with OULAD or real school data later)."""
import numpy as np
import pandas as pd


def generate(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.choice(["low", "mid", "high"], n, p=[0.4, 0.4, 0.2])
    region = rng.choice(["urban", "rural"], n, p=[0.55, 0.45])
    gender = rng.choice(["female", "male"], n)

    # Resource access depends on background (this is the inequality we want to surface)
    p_net = np.where(income == "low", 0.45, np.where(income == "mid", 0.75, 0.95))
    p_net = np.where(region == "rural", p_net - 0.15, p_net)
    has_internet = (rng.random(n) < p_net).astype(int)
    has_device = (rng.random(n) < np.clip(p_net + 0.05, 0, 1)).astype(int)

    attendance = np.clip(rng.normal(82, 12, n) - 6 * (1 - has_internet), 30, 100)
    study_hours = np.clip(rng.normal(9, 3, n) - 2 * (1 - has_device), 0, 25)
    completion = np.clip(rng.normal(75, 15, n) - 8 * (1 - has_internet), 0, 100)
    avg_score = np.clip(
        0.35 * attendance + 0.3 * completion + 1.2 * study_hours + rng.normal(0, 6, n) - 5, 0, 100
    )

    # At-risk = driven by behaviour/performance and access, NOT by income/gender directly
    risk_score = (
        -0.06 * attendance - 0.05 * avg_score - 0.03 * completion
        - 0.12 * study_hours - 0.6 * has_internet - 0.4 * has_device + 8
        + rng.normal(0, 0.6, n)
    )
    at_risk = (risk_score > np.percentile(risk_score, 75)).astype(int)

    return pd.DataFrame({
        "attendance_pct": attendance.round(1),
        "avg_score": avg_score.round(1),
        "assignment_completion": completion.round(1),
        "study_hours_week": study_hours.round(1),
        "has_internet": has_internet,
        "has_device": has_device,
        "income_band": income,   # audit only - NOT used as a model feature
        "region": region,        # audit only
        "gender": gender,        # audit only
        "at_risk": at_risk,
    })


if __name__ == "__main__":
    df = generate()
    df.to_csv("students.csv", index=False)
    print(df.head(), "\nAt-risk rate:", df.at_risk.mean().round(3))
