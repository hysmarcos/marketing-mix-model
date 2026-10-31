import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mmm.data import CHANNELS
from mmm.validation import (
    correlation_summary,
    roi_from_contributions,
    summarize_recovery,
    variance_inflation,
)


def test_variance_inflation_independent_and_duplicated():
    rng = np.random.default_rng(42)
    independent = pd.DataFrame(rng.normal(size=(500, 3)), columns=["a", "b", "c"])
    vif = variance_inflation(independent)
    assert all(v == pytest.approx(1, abs=0.05) for v in vif.values())
    almost_copy = independent.assign(d=independent["a"] + rng.normal(scale=0.05, size=500))
    assert variance_inflation(almost_copy)["d"] > 100


def test_correlation_summary_shape_and_zero_weeks():
    df = pd.DataFrame(
        {
            "revenue": [1.0, 2.0, 3.0, 4.0],
            **{c: [0.0, 1.0, 0.0, 2.0] for c in CHANNELS},
            "newsletter": [1.0, 0.0, 1.0, 0.0],
            "competitor_sales_B": [2.0, 4.0, 6.0, 8.0],
        }
    )
    out = correlation_summary(df)
    n = len(out["columns"])
    assert len(out["matrix"]) == n and all(len(row) == n for row in out["matrix"])
    assert out["matrix"][0][0] == pytest.approx(1)
    assert out["zero_weeks"] == {c: 2 for c in CHANNELS}
    assert set(out["vif"]) == {*CHANNELS, "newsletter", "competitor_sales_B"}


def test_roi_from_contributions_known_values():
    # 2 muestras, 3 semanas, 2 canales. Aporte en escala interna; target_scale = 10.
    contrib = xr.DataArray(
        np.ones((2, 3, 2)),
        dims=("sample", "date", "channel"),
        coords={"channel": ["tv_S", "ooh_S"]},
    )
    spend = pd.DataFrame({"tv_S": [5.0, 5.0, 5.0], "ooh_S": [10.0, 10.0, 20.0]})
    roi = roi_from_contributions(contrib, target_scale=10.0, spend=spend)
    np.testing.assert_allclose(roi.sel(channel="tv_S"), [2.0, 2.0])  # 30 / 15
    np.testing.assert_allclose(roi.sel(channel="ooh_S"), [0.75, 0.75])  # 30 / 40


def test_summarize_recovery_flags_coverage():
    runs = [
        [{"channel": "tv_S", "true": 5.0, "median": 5.2, "p05": 4.0, "p95": 6.0}],
        [{"channel": "tv_S", "true": 5.0, "median": 7.0, "p05": 6.5, "p95": 8.0}],
    ]
    out = summarize_recovery(runs)
    assert len(out) == 1
    assert out[0]["channel"] == "tv_S" and out[0]["true"] == 5.0
    assert out[0]["covered"] == 1 and out[0]["n"] == 2
    assert out[0]["median"] == pytest.approx(6.1)


@pytest.mark.slow
def test_recovery_run_smoke(smoke_model, prepared_data):
    from mmm.validation import recovery_run

    _, controls, train, _ = prepared_data
    rows = recovery_run(smoke_model, train, controls, seed=1, chains=1, draws=50, tune=50)
    assert [r["channel"] for r in rows] == CHANNELS
    for r in rows:
        assert np.isfinite([r["true"], r["median"], r["p05"], r["p95"]]).all()
        assert r["true"] >= 0


@pytest.mark.slow
def test_prior_posterior_roi_smoke(smoke_model, prepared_data):
    from mmm.validation import prior_posterior_roi

    _, _, train, _ = prepared_data
    rows = prior_posterior_roi(smoke_model, train, draws=100)
    assert [r["channel"] for r in rows] == CHANNELS
    for r in rows:
        for k in ("prior", "posterior"):
            assert r[k]["p05"] <= r[k]["median"] <= r[k]["p95"]


def test_bias_corrected_uplift_without_bias_matches_plain(fake_results):
    from mmm.export import media_draws
    from mmm.validation import bias_corrected_uplift

    plan = {c: v * 1.5 for c, v in fake_results["plans"]["current"].items()}
    fake_results["plans"]["optimal"] = plan
    recovery = [{"channel": c, "true": 2.0, "median": 2.0} for c in CHANNELS]
    plain = (
        np.median(media_draws(fake_results, plan))
        / np.median(media_draws(fake_results, fake_results["plans"]["current"]))
        - 1
    )
    assert bias_corrected_uplift(fake_results, recovery) == pytest.approx(100 * plain)


def test_bias_corrected_uplift_lowers_overestimated_channels(fake_results):
    from mmm.validation import bias_corrected_uplift

    plan = {c: v * 1.5 for c, v in fake_results["plans"]["current"].items()}
    fake_results["plans"]["optimal"] = plan
    honest = [{"channel": c, "true": 1.0, "median": 1.0} for c in CHANNELS]
    biased = [{"channel": c, "true": 1.0, "median": 1.0 if c == "tv_S" else 2.0} for c in CHANNELS]
    assert bias_corrected_uplift(fake_results, biased) != pytest.approx(
        bias_corrected_uplift(fake_results, honest)
    )
