"""HR Attrition Analytics — who leaves, why, and who is at risk next.

IBM HR Analytics dataset: 1,470 employees, 35 attributes.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "reports" / "plots"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "hr_attrition.csv", encoding="utf-8-sig")
    df = df.drop(columns=DROP_COLS)
    df["left"] = (df["Attrition"] == "Yes").astype(int)
    return df


def save_fig(fig, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.write_html(PLOTS / f"{name}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(PLOTS / f"{name}.png", scale=2)
    except Exception:
        pass


def rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col)["left"].agg(["mean", "count"]).reset_index()
    return g.rename(columns={"mean": "attrition_rate", "count": "employees"})


def build_charts(df: pd.DataFrame) -> None:
    for col, title in [
        ("Department", "Attrition rate by department"),
        ("OverTime", "Attrition rate by overtime status"),
        ("JobRole", "Attrition rate by job role"),
        ("MaritalStatus", "Attrition rate by marital status"),
    ]:
        g = rate_by(df, col).sort_values("attrition_rate")
        fig = px.bar(g, x="attrition_rate", y=col, orientation="h",
                     title=title, text_auto=".1%", hover_data=["employees"])
        fig.update_xaxes(tickformat=".0%")
        save_fig(fig, f"attrition_by_{col.lower()}")

    df["age_band"] = pd.cut(df["Age"], [17, 25, 35, 45, 55, 65],
                            labels=["18-25", "26-35", "36-45", "46-55", "56+"])
    g = rate_by(df, "age_band")
    fig = px.bar(g, x="age_band", y="attrition_rate", text_auto=".1%",
                 title="Attrition rate by age band")
    fig.update_yaxes(tickformat=".0%")
    save_fig(fig, "attrition_by_age")

    fig = px.box(df, x="Attrition", y="MonthlyIncome", color="Attrition",
                 title="Monthly income: stayers vs leavers")
    save_fig(fig, "income_vs_attrition")

    g = df.groupby(["YearsAtCompany"])["left"].mean().reset_index()
    g = g[g["YearsAtCompany"] <= 20]
    fig = px.line(g, x="YearsAtCompany", y="left", markers=True,
                  title="Attrition rate by tenure (years at company)")
    fig.update_yaxes(tickformat=".0%", title="attrition rate")
    save_fig(fig, "attrition_by_tenure")


# Asymmetric error costs (documented assumptions, editable):
#   A missed leaver (false negative) costs a full replacement — recruiting,
#   lost productivity and ramp-up — conservatively ~6 months of salary.
#   A false positive costs only a retention conversation / small perk.
COST_FN = 50_000   # cost of failing to flag someone who then leaves
COST_FP = 2_000    # cost of a needless retention intervention


def tune_cost_threshold(y_true, proba) -> dict:
    """Pick the probability cut-off that minimises expected dollar cost,
    not the arbitrary 0.5. Returns the threshold and a cost curve."""
    grid = np.linspace(0.05, 0.95, 91)
    costs = []
    for t in grid:
        pred = (proba >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        costs.append(fn * COST_FN + fp * COST_FP)
    costs = np.array(costs)
    best_i = int(costs.argmin())
    # Baseline: flag nobody (every leaver missed) — what the model saves against.
    baseline_cost = int((y_true == 1).sum()) * COST_FN
    return {
        "threshold": float(grid[best_i]),
        "expected_cost": int(costs[best_i]),
        "baseline_cost": baseline_cost,
        "savings": int(baseline_cost - costs[best_i]),
        "grid": grid,
        "costs": costs,
    }


def plot_cost_curve(tune: dict) -> None:
    fig = go.Figure()
    fig.add_scatter(x=tune["grid"], y=tune["costs"], mode="lines",
                    name="expected cost")
    fig.add_vline(x=tune["threshold"], line_dash="dash", line_color="red",
                  annotation_text=f"optimal {tune['threshold']:.2f}")
    fig.update_layout(title="Expected dollar cost vs. decision threshold",
                      xaxis_title="flag-if-probability ≥ threshold",
                      yaxis_title="expected cost ($)")
    save_fig(fig, "cost_threshold_curve")


def shap_global(model, X_columns, X_sample) -> None:
    """Save a global SHAP importance bar (mean |SHAP| per feature)."""
    try:
        import shap
    except ImportError:
        print("shap not installed — skipping SHAP plots (pip install shap)")
        return
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    # Binary RF returns a list [class0, class1]; take the positive class.
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_columns)
    top = mean_abs.nlargest(15).sort_values()
    fig = px.bar(x=top.values, y=top.index, orientation="h",
                 title="SHAP feature importance (mean |impact| on attrition risk)",
                 labels={"x": "mean |SHAP value|", "y": ""})
    save_fig(fig, "shap_importance")


def train_models(df: pd.DataFrame) -> dict:
    X = pd.get_dummies(df.drop(columns=["Attrition", "left", "age_band"]),
                       drop_first=True)
    y = df["left"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_tr)
    results, fitted, probas = [], {}, {}

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=2000,
                                class_weight="balanced"), True),
        "Random Forest": (RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=42), False),
    }
    for name, (model, needs_scaling) in models.items():
        Xtr = scaler.transform(X_tr) if needs_scaling else X_tr
        Xte = scaler.transform(X_te) if needs_scaling else X_te
        model.fit(Xtr, y_tr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = model.predict(Xte)
        results.append({
            "model": name,
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred), 4),
            "recall": round(recall_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4),
            "roc_auc": round(roc_auc_score(y_te, proba), 4),
        })
        fitted[name] = (model, needs_scaling)
        probas[name] = proba

    res = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    res.to_csv(REPORTS / "model_comparison.csv", index=False)
    print(res.to_string(index=False))

    best_name = res.iloc[0]["model"]
    best, needs_scaling = fitted[best_name]

    # --- Cost-based decision threshold (instead of the arbitrary 0.5) ---
    tune = tune_cost_threshold(y_te.to_numpy(), probas[best_name])
    plot_cost_curve(tune)
    print(f"\nCost-optimal threshold: {tune['threshold']:.2f} | "
          f"expected cost ${tune['expected_cost']:,} vs "
          f"${tune['baseline_cost']:,} doing nothing "
          f"(saves ${tune['savings']:,} on the test set).")

    # --- SHAP global importance (tree model only) ---
    if best_name == "Random Forest":
        shap_global(best, list(X.columns), X_te)

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": best, "name": best_name, "columns": list(X.columns),
                 "scaler": scaler, "needs_scaling": needs_scaling,
                 "cost_threshold": tune["threshold"],
                 "cost_fn": COST_FN, "cost_fp": COST_FP},
                MODELS / "attrition_model.joblib")

    # risk drivers from logistic coefficients (interpretable)
    logreg = fitted["Logistic Regression"][0]
    coefs = pd.Series(logreg.coef_[0], index=X.columns).sort_values()
    top = pd.concat([coefs.head(8), coefs.tail(8)])
    fig = px.bar(x=top.values, y=top.index, orientation="h",
                 title="Attrition risk drivers (logistic coefficients, scaled)",
                 labels={"x": "effect on attrition risk", "y": ""})
    save_fig(fig, "risk_drivers")
    return {"results": res, "best": best_name}


def write_report(df: pd.DataFrame, info: dict) -> None:
    overall = df["left"].mean()
    ot = rate_by(df, "OverTime").set_index("OverTime")["attrition_rate"]
    dept = rate_by(df, "Department").sort_values("attrition_rate")
    young = df[df["Age"] <= 25]["left"].mean()
    best = info["results"].iloc[0]
    lines = [
        "# HR Attrition — Executive Summary\n",
        f"- Workforce analysed: **{len(df):,} employees**, overall attrition "
        f"**{overall:.1%}**",
        f"- Overtime is the biggest controllable driver: attrition is "
        f"**{ot['Yes']:.1%}** with overtime vs **{ot['No']:.1%}** without",
        f"- Employees aged 25 or under leave at **{young:.1%}** — early-career "
        "retention needs attention",
        f"- Highest-risk department: **{dept.iloc[-1, 0]}** "
        f"({dept.iloc[-1]['attrition_rate']:.1%})",
        f"- Best model: **{best['model']}** (ROC-AUC {best['roc_auc']:.3f}, "
        f"recall {best['recall']:.1%})\n",
        "## Recommendations",
        "1. Cap or compensate overtime — it nearly triples attrition risk.",
        "2. Build an early-career mentoring track for employees under 26.",
        "3. Review pay bands: leavers earn materially less than stayers.",
        "4. Score employees monthly with the saved model and flag the top "
        "decile for manager check-ins.",
    ]
    (REPORTS / "executive_summary.md").write_text("\n".join(lines))


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df):,} employees, attrition {df['left'].mean():.1%}")
    build_charts(df)
    info = train_models(df)
    write_report(df, info)
    print("Done. See reports/ and models/")


if __name__ == "__main__":
    main()
