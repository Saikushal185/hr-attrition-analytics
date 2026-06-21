"""Schema checks for the HR dataset."""
import pandas as pd

REQUIRED = ["Age", "Attrition", "Department", "JobRole", "MonthlyIncome",
            "OverTime", "YearsAtCompany"]


def validate(df: pd.DataFrame) -> list[str]:
    problems = []
    for col in REQUIRED:
        if col not in df.columns:
            problems.append(f"missing required column: {col}")
    if "Attrition" in df:
        bad = ~df["Attrition"].isin(["Yes", "No"])
        if bad.any():
            problems.append("Attrition must be 'Yes' or 'No'")
    if "Age" in df and (df["Age"] < 16).any():
        problems.append("Age below working minimum found")
    return problems
