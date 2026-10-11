"""Tiny SQLite storage for LearnLift (built into Python - nothing to install).
On Streamlit Cloud the file is temporary: it resets when the app restarts."""
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from data import generate
from model import FEATURES, explain

DB_PATH = "learnlift.db"


def _conn():
    return sqlite3.connect(DB_PATH)


def init():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            attendance_pct REAL, avg_score REAL, assignment_completion REAL,
            study_hours_week REAL, has_internet INTEGER, has_device INTEGER,
            score_pct REAL, reasons TEXT, recommendations TEXT)""")


def save_assessment(student_id, s, score_pct, reasons, recs, created_at=None):
    created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as c:
        c.execute(
            "INSERT INTO assessments (student_id, created_at, attendance_pct, avg_score, "
            "assignment_completion, study_hours_week, has_internet, has_device, "
            "score_pct, reasons, recommendations) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (student_id, created_at, s["attendance_pct"], s["avg_score"],
             s["assignment_completion"], s["study_hours_week"], s["has_internet"],
             s["has_device"], round(score_pct, 1), reasons, "; ".join(recs)))


def latest_per_student():
    """Most recent assessment for every student, highest support need first."""
    q = """SELECT a.student_id, a.score_pct, a.reasons, a.recommendations, a.created_at
           FROM assessments a
           JOIN (SELECT student_id, MAX(id) AS mid FROM assessments GROUP BY student_id) m
             ON a.id = m.mid
           ORDER BY a.score_pct DESC"""
    with _conn() as c:
        return pd.read_sql_query(q, c)


def history(student_id):
    with _conn() as c:
        return pd.read_sql_query(
            "SELECT created_at, score_pct FROM assessments WHERE student_id=? ORDER BY id",
            c, params=(student_id,))


def clear_all():
    with _conn() as c:
        c.execute("DELETE FROM assessments")


def reasons_text(exp):
    top = exp[exp.effect >= 5].head(3)
    return ", ".join(top.factor) if len(top) else "No single factor stands out"


def seed_demo(model, ref, recommend):
    """Add 8 demo students, plus one student whose score improves over 3 visits."""
    def add(sid, s, days_ago=0):
        p = model.predict_proba(pd.DataFrame([s])[FEATURES])[0][1]
        exp = explain(model, s, ref)
        when = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
        save_assessment(sid, s, p * 100, reasons_text(exp), recommend(s), when)

    sample = generate(200, seed=1).sample(8, random_state=3).reset_index(drop=True)
    for i, row in sample.iterrows():
        add(f"DEMO-{i + 1:02d}", {f: float(row[f]) for f in FEATURES})

    journey = [  # same student after support: access fixed, attendance and marks rise
        (60, dict(attendance_pct=62, avg_score=40, assignment_completion=45,
                  study_hours_week=4, has_internet=0, has_device=1)),
        (30, dict(attendance_pct=70, avg_score=48, assignment_completion=58,
                  study_hours_week=6, has_internet=1, has_device=1)),
        (0, dict(attendance_pct=82, avg_score=60, assignment_completion=72,
                 study_hours_week=8, has_internet=1, has_device=1)),
    ]
    for days_ago, s in journey:
        add("DEMO-IMPROVING", s, days_ago)
