import numpy as np
import pytest

from mmm.data import CHANNEL_LABELS, CHANNELS


@pytest.fixture
def fake_results() -> dict:
    """Resultados con el esquema completo y muestras aleatorias (semilla 42)."""
    rng = np.random.default_rng(42)
    n = 200
    dates = [f"2019-{m:02d}-01" for m in range(1, 13)]
    scales = {c: float(s) for c, s in zip(CHANNELS, [1.5e5, 5e5, 3e4, 1.5e4, 1.7e4], strict=True)}
    return {
        "meta": {
            "generated_at": "2026-10-31T00:00:00Z",
            "source": "Robyn dt_simulated_weekly",
            "currency": "$",
            "horizon_weeks": 13,
            "n_draws": n,
            "holdout_weeks": 12,
        },
        "channels": CHANNELS,
        "channel_labels": CHANNEL_LABELS,
        "scales": {"channel": scales, "target": 3.8e6},
        "draws": {
            "lam": {c: rng.gamma(3, 1, n).tolist() for c in CHANNELS},
            "beta": {c: np.abs(rng.normal(0, 0.1, n)).tolist() for c in CHANNELS},
            "alpha": {c: rng.beta(1, 3, n).tolist() for c in CHANNELS},
            "baseline_weekly": rng.normal(1.2e6, 5e4, n).tolist(),
        },
        "spend_max_weekly": scales,
        "plans": {
            "current": {c: scales[c] * 0.3 for c in CHANNELS},
            "optimal": {c: scales[c] * 0.3 for c in CHANNELS},
        },
        "contributions": [
            {"date": d, "series": s, "value": 1.0} for d in dates for s in ["base", *CHANNELS]
        ],
        "roi": [{"channel": c, "median": 1.0, "p05": 0.5, "p95": 1.5} for c in CHANNELS],
        "holdout": [
            {"date": d, "actual": 1.0, "median": 1.0, "p05": 0.9, "p95": 1.1} for d in dates
        ],
        "diagnostics": {"divergences": 0, "max_rhat": 1.0, "min_ess": 1000.0, "holdout_mape": 0.1},
        "headline": {"media_uplift_pct": 0.0},
    }
