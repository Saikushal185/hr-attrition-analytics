"""Cohort view: attrition rate by tenure bucket and job level."""
import pandas as pd

from src.main import load_data


def tenure_cohorts() -> pd.DataFrame:
    df = load_data()
    df["tenure_band"] = pd.cut(df["YearsAtCompany"],
                               [-1, 1, 3, 6, 10, 100],
                               labels=["<1y", "1-3y", "4-6y", "7-10y", "10y+"])
    g = (df.groupby(["tenure_band", "JobLevel"], observed=True)["left"]
         .agg(["mean", "count"]).reset_index()
         .rename(columns={"mean": "attrition_rate", "count": "employees"}))
    return g


if __name__ == "__main__":
    print(tenure_cohorts().to_string(index=False))
