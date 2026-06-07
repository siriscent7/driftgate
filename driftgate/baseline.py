import threading
from collections import defaultdict
import numpy as np

from driftgate.detector import (
    population_stability_index,
    ks_test,
    classify_drift,
)


class DriftMonitor:
    """Holds training baselines per feature and accumulates live observations,
    computing drift on demand. Thread-safe for concurrent requests.
    """

    def __init__(self, window: int = 5000):
        self._baselines: dict[str, np.ndarray] = {}
        self._observations: dict[str, list[float]] = defaultdict(list)
        self._window = window  # max live values kept per feature
        self._lock = threading.Lock()

    def set_baseline(self, feature: str, values: list[float]) -> None:
        with self._lock:
            self._baselines[feature] = np.asarray(values, dtype=float)
            self._observations[feature] = []  # reset live window

    def observe(self, feature: str, value: float) -> None:
        with self._lock:
            buf = self._observations[feature]
            buf.append(float(value))
            if len(buf) > self._window:
                # keep only the most recent `window` values (rolling)
                del buf[0 : len(buf) - self._window]

    def drift_report(self, min_samples: int = 50) -> dict:
        """Compute drift for every feature that has a baseline + enough samples."""
        report = {}
        with self._lock:
            for feature, baseline in self._baselines.items():
                current = self._observations.get(feature, [])
                if len(current) < min_samples:
                    report[feature] = {
                        "status": "insufficient_data",
                        "samples": len(current),
                    }
                    continue
                curr = np.asarray(current, dtype=float)
                psi = population_stability_index(baseline, curr)
                ks_stat, p_value = ks_test(baseline, curr)
                report[feature] = {
                    "status": classify_drift(psi),
                    "psi": round(psi, 4),
                    "ks_statistic": round(ks_stat, 4),
                    "ks_p_value": round(p_value, 4),
                    "samples": len(current),
                }
        return report

    def features(self) -> list[str]:
        with self._lock:
            return list(self._baselines.keys())