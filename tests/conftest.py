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


@pytest.fixture(scope="session")
def prepared_data():
    from mmm.data import load_raw, prepare_features
    from mmm.model import split_train_test

    df, controls = prepare_features(load_raw())
    train, test = split_train_test(df)
    return df, controls, train, test


@pytest.fixture(scope="session")
def smoke_model(prepared_data):
    from mmm.model import build_mmm, fit_mmm

    _, controls, train, _ = prepared_data
    mmm = build_mmm(controls)
    fit_mmm(mmm, train, chains=1, draws=50, tune=50)
    return mmm


@pytest.fixture(scope="session")
def last_date(prepared_data):
    return prepared_data[0]["DATE"].max()


@pytest.fixture(scope="session")
def smoke_results(smoke_model, prepared_data):
    from mmm.diagnostics import Diagnostics
    from mmm.export import HORIZON_WEEKS, build_results
    from mmm.model import predict_holdout

    df, _, train, test = prepared_data
    current = {c: float(df[c].tail(HORIZON_WEEKS).mean()) for c in CHANNELS}
    diagnostics = Diagnostics(divergences=0, max_rhat=1.0, min_ess=1000.0, holdout_mape=0.1)
    return build_results(
        smoke_model,
        smoke_model.idata,
        df,
        train,
        predict_holdout(smoke_model, test),
        diagnostics,
        optimal_plan=current,
    )
