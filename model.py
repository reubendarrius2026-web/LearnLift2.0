"""Train the risk model, audit fairness, and generate support recommendations."""
import joblib
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

FEATURES = ["attendance_pct", "avg_score", "assignment_completion",
            "study_hours_week", "has_internet", "has_device"]
AUDIT_COLS = ["income_band", "region", "gender"]


def train(df):
    X, y = df[FEATURES], df["at_risk"]
    X_tr, X_te, y_tr, y_te, _, df_te = train_test_split(
        X, y, df, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {"accuracy": accuracy_score(y_te, pred), "recall": recall_score(y_te, pred)}
    return model, metrics, df_te.assign(pred=pred)


def fairness_audit(test_df):
    """Per-group flag rate and recall. Large gaps = the model treats groups differently."""
    rows = []
    for col in AUDIT_COLS:
        for g, sub in test_df.groupby(col):
            rows.append({
                "attribute": col, "group": g, "n": len(sub),
                "flagged_rate": round(sub.pred.mean(), 3),
                "recall (at-risk caught)": round(recall_score(sub.at_risk, sub.pred), 3),
            })
    return pd.DataFrame(rows)


def oof_probabilities(df, seed=42):
    """Out-of-fold risk scores for every student (each student is scored by a model that
    never saw them). Gives the fairness audit all students instead of a small test set."""
    kf = StratifiedKFold(5, shuffle=True, random_state=seed)
    proba = np.zeros(len(df))
    for tr, te in kf.split(df, df["at_risk"]):
        m = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed)
        m.fit(df.iloc[tr][FEATURES], df.iloc[tr]["at_risk"])
        proba[te] = m.predict_proba(df.iloc[te][FEATURES])[:, 1]
    return proba


def fairness_report(df, proba, threshold=0.5):
    """Returns (group_table, verdicts, overall). Recall = share of truly at-risk students the
    model catches. A gap is only flagged if it is bigger than random noise would explain."""
    d = df.assign(pred=(proba >= threshold).astype(int))
    overall = {
        "recall": recall_score(d.at_risk, d.pred),
        "precision": precision_score(d.at_risk, d.pred),
        "accuracy": accuracy_score(d.at_risk, d.pred),
    }
    rows = []
    for col in AUDIT_COLS:
        for g, sub in d.groupby(col):
            n_pos = int(sub.at_risk.sum())
            rows.append({"attribute": col, "group": g, "students": len(sub),
                         "at_risk": n_pos, "flagged": sub.pred.mean(),
                         "caught": recall_score(sub.at_risk, sub.pred),
                         "se": (0 if n_pos == 0 else
                                (recall_score(sub.at_risk, sub.pred) *
                                 (1 - recall_score(sub.at_risk, sub.pred)) / n_pos) ** 0.5)})
    table = pd.DataFrame(rows)

    verdicts = []
    for col, sub in table.groupby("attribute"):
        hi, lo = sub.loc[sub.caught.idxmax()], sub.loc[sub.caught.idxmin()]
        gap = hi.caught - lo.caught
        noise = (hi.se ** 2 + lo.se ** 2) ** 0.5
        verdicts.append({"attribute": col, "gap": gap, "highest": hi.group, "lowest": lo.group,
                         "needs_attention": bool(gap > 0.05 and gap > 2 * noise)})
    return table, verdicts, overall


def recommend(s):
    """Map a student's specific needs to support. Framed as needs, not labels."""
    recs = []
    if not s["has_internet"]:
        recs.append("Offline learning pack + access to school/community internet lab")
    if not s["has_device"]:
        recs.append("Device lending program or shared computer-lab slots")
    if s["attendance_pct"] < 75:
        recs.append("Mentor check-in to understand barriers to attendance")
    if s["avg_score"] < 50:
        recs.append("Peer tutoring / remedial classes in weak subjects")
    if s["assignment_completion"] < 60:
        recs.append("Structured assignment plan with weekly teacher follow-up")
    if s["study_hours_week"] < 6:
        recs.append("Study-skills workshop and a guided study schedule")
    return recs or ["On track - suggest enrichment resources and scholarships"]


LABELS = {
    "attendance_pct": "Attendance",
    "avg_score": "Average score",
    "assignment_completion": "Assignment completion",
    "study_hours_week": "Study hours",
    "has_internet": "Internet access",
    "has_device": "Device access",
}


def reference_profile(df):
    """Typical 'on-track' student: the median of students who are not at risk."""
    return df[df["at_risk"] == 0][FEATURES].median()


def explain(model, s, ref):
    """Exact Shapley values: how many percentage points each factor adds to (or removes
    from) this student's support-need score, compared with a typical on-track student."""
    n = len(FEATURES)
    student = np.array([s[f] for f in FEATURES], dtype=float)
    base = ref[FEATURES].to_numpy(dtype=float)

    rows = []
    for mask in range(2 ** n):
        rows.append([student[i] if mask >> i & 1 else base[i] for i in range(n)])
    preds = model.predict_proba(pd.DataFrame(rows, columns=FEATURES))[:, 1]

    effects = []
    for i in range(n):
        total = 0.0
        for mask in range(2 ** n):
            if mask >> i & 1:
                continue
            k = bin(mask).count("1")
            weight = factorial(k) * factorial(n - k - 1) / factorial(n)
            total += weight * (preds[mask | (1 << i)] - preds[mask])
        effects.append(total * 100)

    out = pd.DataFrame({"factor": [LABELS[f] for f in FEATURES], "effect": effects})
    return out.sort_values("effect", ascending=False).reset_index(drop=True)


def save(model, path="model.joblib"):
    joblib.dump(model, path)


def load(path="model.joblib"):
    return joblib.load(path)
