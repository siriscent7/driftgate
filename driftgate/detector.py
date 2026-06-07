import numpy as np
from scipy import stats


def population_stability_index(
    baseline: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Compute PSI between a baseline and current sample of one feature.

    PSI measures how much a distribution has shifted:
        < 0.1     -> stable
        0.1-0.25  -> moderate drift
        > 0.25    -> significant drift

    We bin both samples using the baseline's quantile edges, then compare
    the proportion of data falling in each bin.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    # Bin edges from the baseline distribution (quantile-based)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(baseline, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf  # catch out-of-range values

    base_counts, _ = np.histogram(baseline, bins=edges)
    curr_counts, _ = np.histogram(current, bins=edges)

    # Convert to proportions; add tiny epsilon to avoid log(0) / div-by-0
    eps = 1e-6
    base_prop = base_counts / len(baseline) + eps
    curr_prop = curr_counts / len(current) + eps

    psi = np.sum((curr_prop - base_prop) * np.log(curr_prop / base_prop))
    return float(psi)


def ks_test(baseline: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov test between two samples.

    Returns (statistic, p_value). A small p_value (< 0.05) suggests the
    two samples come from different distributions (drift).
    """
    statistic, p_value = stats.ks_2samp(
        np.asarray(baseline, dtype=float), np.asarray(current, dtype=float)
    )
    return float(statistic), float(p_value)


def classify_drift(psi: float) -> str:
    """Map a PSI value to a human-readable severity level."""
    if psi < 0.1:
        return "stable"
    elif psi < 0.25:
        return "moderate"
    else:
        return "significant"