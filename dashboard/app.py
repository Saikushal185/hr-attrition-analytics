"""Streamlit dashboard — HR attrition analytics + employee risk lookup."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main import load_data, rate_by  # noqa: E402

st.set_page_config(page_title="HR Attrition Analytics", layout="wide")
st.title("👥 HR Attrition Analytics")


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    p = ROOT / "models" / "attrition_model.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_resource
def get_background(_bundle):
    """Encoded (and scaled, if needed) background sample for SHAP."""
    raw = load_data().drop(columns=["Attrition", "left"])
    Xb = pd.get_dummies(raw, drop_first=True).reindex(
        columns=_bundle["columns"], fill_value=0)
    return _bundle["scaler"].transform(Xb) if _bundle["needs_scaling"] else Xb


def explain_row(bundle, X_raw, X_enc):
    """Per-employee SHAP contributions for the saved best model."""
    try:
        import shap
    except ImportError:
        st.caption("Install `shap` to see per-employee explanations.")
        return None
    bg = get_background(bundle)
    if bundle["name"] == "Random Forest":
        sv = shap.TreeExplainer(bundle["model"]).shap_values(X_raw)
        sv = sv[1] if isinstance(sv, list) else sv
    else:
        # Linear model: SHAP over the scaled feature space.
        masker = bg if hasattr(bg, "shape") else np.asarray(bg)
        expl = shap.LinearExplainer(bundle["model"], masker)
        sv = expl.shap_values(X_enc)
    return pd.Series(np.ravel(sv), index=bundle["columns"])


df = get_data()

with st.sidebar:
    st.header("Filters")
    depts = st.multiselect("Department", sorted(df["Department"].unique()))
    roles = st.multiselect("Job role", sorted(df["JobRole"].unique()))

view = df.copy()
if depts:
    view = view[view["Department"].isin(depts)]
if roles:
    view = view[view["JobRole"].isin(roles)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", f"{len(view):,}")
c2.metric("Attrition rate", f"{view['left'].mean():.1%}")
c3.metric("Avg monthly income", f"${view['MonthlyIncome'].mean():,.0f}")
c4.metric("Avg tenure", f"{view['YearsAtCompany'].mean():.1f} yrs")

st.subheader("Charts")
plots = ROOT / "reports" / "plots"
for name in ["attrition_by_overtime", "attrition_by_department",
             "attrition_by_jobrole", "attrition_by_age",
             "income_vs_attrition", "attrition_by_tenure", "risk_drivers",
             "shap_importance", "cost_threshold_curve"]:
    f = plots / f"{name}.html"
    if f.exists():
        st.components.v1.html(f.read_text(), height=430)

st.subheader("🔍 Employee attrition risk scorer")
bundle = get_model()
if bundle is None:
    st.warning("Train first: python3 src/main.py")
else:
    with st.form("risk"):
        a, b, c = st.columns(3)
        age = a.number_input("Age", 18, 60, 30)
        income = a.number_input("Monthly income ($)", 1000, 20000, 5000)
        overtime = b.selectbox("Works overtime?", ["No", "Yes"])
        dept = b.selectbox("Department", sorted(df["Department"].unique()))
        years = c.number_input("Years at company", 0, 40, 3)
        satisfaction = c.slider("Job satisfaction (1-4)", 1, 4, 3)
        go = st.form_submit_button("Score risk")
    if go:
        row = df.drop(columns=["Attrition", "left"]).iloc[[0]].copy()
        row["Age"], row["MonthlyIncome"], row["OverTime"] = age, income, overtime
        row["Department"], row["YearsAtCompany"] = dept, years
        row["JobSatisfaction"] = satisfaction
        X = pd.get_dummies(row, drop_first=True).reindex(
            columns=bundle["columns"], fill_value=0)
        Xenc = bundle["scaler"].transform(X) if bundle["needs_scaling"] else X
        p = bundle["model"].predict_proba(Xenc)[0, 1]

        # Flag using the cost-optimal threshold, not an arbitrary 0.5.
        thr = bundle.get("cost_threshold", 0.5)
        if p >= thr:
            st.error(f"FLAG for retention — {p:.1%} risk "
                     f"(≥ cost-optimal threshold {thr:.0%})")
        else:
            st.success(f"No action — {p:.1%} risk (< threshold {thr:.0%})")
        st.caption(
            f"Threshold set to minimise expected cost "
            f"(missed leaver ≈ ${bundle.get('cost_fn', 0):,}, "
            f"needless intervention ≈ ${bundle.get('cost_fp', 0):,}), "
            "not the default 0.5.")

        # Per-employee SHAP: why is THIS person flagged? Works for both the
        # tree (TreeExplainer) and linear (LinearExplainer) best model.
        contrib = explain_row(bundle, X, Xenc)
        if contrib is not None:
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
            st.markdown("**Top drivers for this employee (SHAP):**")
            for feat, val in top.items():
                arrow = "↑ raises" if val > 0 else "↓ lowers"
                st.write(f"- `{feat}` {arrow} risk ({val:+.3f})")

mc = ROOT / "reports" / "model_comparison.csv"
if mc.exists():
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv(mc), use_container_width=True)
