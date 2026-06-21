# 👥 HR Attrition Analytics

Why do employees leave, and who is at risk next? Built on the IBM HR
Analytics dataset (1,470 employees, 31 usable attributes).

## What this project demonstrates
- **People analytics EDA** — attrition by department, role, age, overtime,
  income, and tenure (7 interactive charts)
- **Imbalanced classification** — 16% positive class, handled with
  `class_weight="balanced"`; recall prioritised (catching leavers matters
  more than false alarms)
- **Interpretable risk drivers** — logistic regression coefficients chart
- **Deployment thinking** — saved model bundle + a live risk-scoring form
  in the Streamlit dashboard

## Key findings
- Overtime nearly triples attrition risk
- Employees under 26 have the highest attrition of any age band
- Logistic Regression beats Random Forest on ROC-AUC (0.81) and recall

## Run it
```bash
./run.sh        # or see torun.txt
```

## Recent upgrades
- **Cost-based threshold**: the flag/no-flag cut-off minimises expected dollar cost (missed leaver vs. needless intervention) instead of the arbitrary 0.5 — saves ~$2.4M vs. doing nothing on the test set (`reports/cost_threshold_curve`).
- **SHAP explanations**: per-employee driver breakdown in the dashboard (TreeExplainer / LinearExplainer depending on the best model) so each flag comes with a 'why'.

## Development

```bash
make install   # install dev dependencies
make test      # run tests
make lint      # lint with ruff
```

## Docker

```bash
docker build -t hr-attrition .
docker run -p 8501:8501 hr-attrition
```

## REST API

```bash
uvicorn src.api:app --reload
# POST /score with an employee JSON returns attrition probability and flag
```

## CLI

```bash
python -m src.cli segments   # top attrition segments
python -m src.cli analyse    # full pipeline
```
