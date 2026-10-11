import pandas as pd
import streamlit as st

import db
from data import generate
from model import (FEATURES, train, recommend, explain, reference_profile,
                   oof_probabilities, fairness_report)

st.set_page_config(page_title="LearnLift", page_icon="🎓")
st.title("🎓 LearnLift")
st.caption("Identify support needs early - not label students.")


@st.cache_resource
def get_model():
    df = generate()
    model, metrics, _ = train(df)
    return model, metrics, df, oof_probabilities(df), reference_profile(df)


model, metrics, audit_df, audit_proba, ref = get_model()
db.init()
tab1, tab2, tab3 = st.tabs(["Student support", "Fairness audit", "Class dashboard"])

with tab1:
    c1, c2 = st.columns(2)
    s = {
        "attendance_pct": c1.slider("Attendance %", 30, 100, 70),
        "avg_score": c1.slider("Average score", 0, 100, 45),
        "assignment_completion": c1.slider("Assignment completion %", 0, 100, 55),
        "study_hours_week": c2.slider("Study hours / week", 0, 25, 5),
        "has_internet": int(c2.checkbox("Has internet at home", value=False)),
        "has_device": int(c2.checkbox("Has a device (phone/PC)", value=True)),
    }
    prob = model.predict_proba(pd.DataFrame([s])[FEATURES])[0][1]
    st.metric("Support-need score", f"{prob:.0%}")
    st.progress(float(prob))

    st.subheader("Why this score?")
    exp = explain(model, s, ref)
    up = exp[exp.effect >= 1]
    down = exp[exp.effect <= -1]
    if len(up):
        st.write("**Raising the score:** " + ", ".join(
            f"{r.factor} (+{r.effect:.0f} pts)" for r in up.itertuples()))
    if len(down):
        st.write("**Lowering the score (strengths):** " + ", ".join(
            f"{r.factor} ({r.effect:.0f} pts)" for r in down.itertuples()))
    if not len(up) and not len(down):
        st.write("No single factor stands out - this student looks close to typical.")
    chart = exp.set_index("factor").rename(columns={"effect": "Effect on score (percentage points)"})
    st.bar_chart(chart)
    st.caption("Compared with a typical on-track student. Positive = increases support need. "
               "This guides support - a teacher should always review it.")

    st.subheader("Recommended support")
    recs = recommend(s)
    for r in recs:
        st.write("✅", r)

    st.divider()
    sid = st.text_input("Student ID (use an ID, not a name)", value="S001")
    if st.button("💾 Save this student"):
        if sid.strip():
            db.save_assessment(sid.strip(), s, prob * 100, db.reasons_text(exp), recs)
            st.success(f"Saved {sid.strip()}. See the Class dashboard tab.")
        else:
            st.error("Enter a student ID first.")

with tab2:
    st.subheader("Does the model work equally well for every group?")
    st.write("Income, region and gender are **not** model inputs. They are used only here, to check "
             "whether students in different groups get the same chance of being noticed. "
             "**'At-risk caught'** is the share of students who truly need help that the model flags.")

    thr = st.slider("Flagging threshold: flag a student when their score is at least", 0.20, 0.70, 0.50, 0.05,
                    format="%.2f",
                    help="Lower = catches more students who need help, but also more false alarms.")
    table, verdicts, overall = fairness_report(audit_df, audit_proba, thr)

    m1, m2, m3 = st.columns(3)
    m1.metric("At-risk students caught", f"{overall['recall']:.0%}")
    m2.metric("Flags that were right", f"{overall['precision']:.0%}")
    m3.metric("Overall accuracy", f"{overall['accuracy']:.0%}")
    st.caption("Lowering the threshold catches more students in need, at the cost of more false alarms.")

    st.markdown("**Verdict**")
    names = {"income_band": "Income", "region": "Region", "gender": "Gender"}
    for v in verdicts:
        line = (f"{names[v['attribute']]}: {v['gap'] * 100:.0f}-point gap in students caught "
                f"({v['highest']} highest, {v['lowest']} lowest)")
        if v["needs_attention"]:
            st.warning("⚠️ " + line + ". Larger than random variation, so this needs attention.")
        else:
            st.success("✅ " + line + ". Within normal variation.")

    chart = table.assign(label=table.attribute.map(names) + ": " + table.group.astype(str))
    chart = chart.set_index("label")[["caught"]].mul(100).rename(columns={"caught": "At-risk students caught (%)"})
    st.bar_chart(chart)

    with st.expander("See the full table"):
        show = table.assign(attribute=table.attribute.map(names))[
            ["attribute", "group", "students", "at_risk", "flagged", "caught"]]
        show = show.rename(columns={"attribute": "Attribute", "group": "Group", "students": "Students",
                                    "at_risk": "At-risk students", "flagged": "Flagged",
                                    "caught": "At-risk caught"})
        show["Flagged"] = (show["Flagged"] * 100).round(0)
        show["At-risk caught"] = (show["At-risk caught"] * 100).round(0)
        st.dataframe(show, width="stretch", hide_index=True, column_config={
            "Flagged": st.column_config.NumberColumn(format="%d%%"),
            "At-risk caught": st.column_config.NumberColumn(format="%d%%")})
        st.caption("More low-income and rural students are flagged because they face more barriers. "
                   "That is expected. The important column is 'At-risk caught'.")
    st.caption("Scores come from out-of-fold predictions on all 2,000 students. Small groups give "
               "noisy results, and this data is synthetic, so real data is needed to confirm.")

with tab3:
    st.write("Saved students, with the highest support need first.")
    b1, b2 = st.columns(2)
    if b1.button("Add 8 demo students"):
        db.seed_demo(model, ref, recommend)
        st.rerun()
    if b2.button("Clear all saved data"):
        db.clear_all()
        st.rerun()

    latest = db.latest_per_student()
    if latest.empty:
        st.info("No students saved yet. Save one from the Student support tab, "
                "or click 'Add 8 demo students'.")
    else:
        table = latest.rename(columns={
            "student_id": "Student", "score_pct": "Support need",
            "reasons": "Main reasons", "recommendations": "Recommended support",
            "created_at": "Last updated"})
        st.dataframe(
            table, width="stretch", hide_index=True,
            column_config={"Support need": st.column_config.ProgressColumn(
                "Support need", min_value=0, max_value=100, format="%.0f%%")})
        st.download_button("Download as CSV", table.to_csv(index=False),
                           "learnlift_students.csv", "text/csv")

        st.subheader("Score history")
        pick = st.selectbox("Student", latest["student_id"])
        h = db.history(pick)
        if len(h) >= 2:
            st.line_chart(h.set_index("created_at")["score_pct"])
            st.caption("Support-need score over time. A falling line means the support is working.")
        else:
            st.info("Only one assessment saved for this student. Save more over time to see a trend.")
    st.caption("Prototype storage resets when the hosted app restarts. A hosted database "
               "would be used in a real deployment.")
