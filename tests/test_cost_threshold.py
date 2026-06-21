import numpy as np
from src import main


def test_cost_threshold_in_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    proba = rng.random(200)
    tune = main.tune_cost_threshold(y, proba)
    assert 0.05 <= tune["threshold"] <= 0.95
    assert tune["expected_cost"] >= 0
