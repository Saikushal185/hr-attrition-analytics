"""Identify the highest-attrition employee segments for HR to target."""
import pandas as pd

from src.main import load_data, rate_by


def top_risk_segments(by=("Department", "JobRole", "OverTime"),
                      min_employees: int = 30) -> pd.DataFrame:
    """Rank segments by attrition rate, keeping only sizeable groups."""
    df = load_data()
    rows = []
    for col in by:
        g = rate_by(df, col)
        g = g[g["employees"] >= min_employees]
        for _, r in g.iterrows():
            rows.append({"dimension": col, "segment": r[col],
                         "attrition_rate": round(r["attrition_rate"], 3),
                         "employees": int(r["employees"])})
    out = pd.DataFrame(rows).sort_values("attrition_rate", ascending=False)
    return out.reset_index(drop=True)


if __name__ == "__main__":
    print(top_risk_segments().head(10).to_string(index=False))
