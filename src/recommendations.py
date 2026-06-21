"""Translate model drivers into concrete retention actions."""

ACTIONS = {
    "OverTime_Yes": "Cap sustained overtime; redistribute workload on at-risk teams.",
    "MonthlyIncome": "Review pay bands for early-tenure, below-median earners.",
    "YearsAtCompany": "Strengthen 0-2 year onboarding and mentorship.",
    "JobSatisfaction": "Run targeted engagement surveys and 1:1 follow-ups.",
    "StockOptionLevel": "Extend equity to high-impact individual contributors.",
    "DistanceFromHome": "Offer hybrid/remote options for long commuters.",
}


def recommend(top_features: list[str]) -> list[str]:
    """Map ranked driver names to action strings (best-effort prefix match)."""
    out = []
    for feat in top_features:
        for key, action in ACTIONS.items():
            if feat.startswith(key.split("_")[0]):
                out.append(action)
                break
    # De-duplicate preserving order.
    seen, unique = set(), []
    for a in out:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique
