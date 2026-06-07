import numpy as np
from driftgate.detector import (
    population_stability_index,
    ks_test,
    classify_drift,
)


def test_psi_no_drift():
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 5000)
    curr = rng.normal(0, 1, 5000)
    psi = population_stability_index(base, curr)
    assert psi < 0.1  # stable


def test_psi_significant_drift():
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 5000)
    curr = rng.normal(3, 1, 5000)  # mean shifted by 3
    psi = population_stability_index(base, curr)
    assert psi > 0.25  # significant


def test_ks_detects_shift():
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 5000)
    curr = rng.normal(3, 1, 5000)
    _, p_value = ks_test(base, curr)
    assert p_value < 0.05


def test_ks_no_shift():
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 5000)
    curr = rng.normal(0, 1, 5000)
    _, p_value = ks_test(base, curr)
    assert p_value > 0.05


def test_classify_drift_levels():
    assert classify_drift(0.05) == "stable"
    assert classify_drift(0.15) == "moderate"
    assert classify_drift(0.30) == "significant"