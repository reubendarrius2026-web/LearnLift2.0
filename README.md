# 🎓 LearnLift

AI-powered educational support system that identifies students who may fall behind **early** and recommends
personalized support - to identify needs, not to label students.

Supports **UN SDG 4** (Quality Education), **SDG 10** (Reduced Inequalities) and **SDG 1** (No Poverty).

## How it works
1. **Risk model** - Random Forest on attendance, scores, assignment completion, study hours, internet/device access.
2. **Recommendations** - maps each student's specific needs to concrete support (offline packs, device lending, tutoring, mentoring).
3. **Fairness audit** - income, region and gender are *not* model inputs; they are used only to check that the model
   flags and catches at-risk students equally across groups.

## Run it
```bash
pip install -r requirements.txt
python train.py          # trains model + prints fairness audit
streamlit run app.py     # launches the demo
```

## Data
Uses a synthetic dataset (`data.py`) for the prototype. Can be swapped for the
[Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open-dataset) or real school data.

## Roadmap
- Train on OULAD / real data
- Mitigate fairness gaps found in the audit
- Teacher dashboard and multilingual support
