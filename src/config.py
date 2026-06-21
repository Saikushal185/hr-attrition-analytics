"""Tunable cost assumptions and modelling constants."""

# Asymmetric error costs in dollars (see README for rationale).
COST_FN = 50_000   # missed leaver -> full replacement cost
COST_FP = 2_000    # needless retention intervention

RANDOM_STATE = 42
TEST_SIZE = 0.25
