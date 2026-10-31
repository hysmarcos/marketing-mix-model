import json

import numpy as np
import xarray as xr
from pymc_marketing.mmm.transformers import logistic_saturation

from mmm.export import (
    logistic,
    simulate_plan,
    weekly_contribution,
    write_results,
    write_simulator_cases,
)

SCHEMA_KEYS = {
    "meta",
    "channels",
    "channel_labels",
    "scales",
    "draws",
    "spend_max_weekly",
    "plans",
    "contributions",
    "roi",
    "holdout",
    "diagnostics",
    "headline",
}


def test_logistic_matches_pymc_marketing():
    lam, x = np.array([0.5, 2.0, 7.0]), np.array([0.0, 0.3, 1.2])
    xr_x, xr_lam = xr.DataArray(x, dims="i"), xr.DataArray(lam, dims="i")
    expected = np.asarray(logistic_saturation(xr_x, xr_lam).eval())
    np.testing.assert_allclose(logistic(lam * x), expected, rtol=1e-12)


def test_weekly_contribution_known_value():
    # z = 2 * 5 / 10 = 1 -> (1 - e^-1) / (1 + e^-1) = 0.46211715726
    out = weekly_contribution(
        5.0, lam=np.array([2.0]), beta=np.array([0.5]), channel_scale=10.0, target_scale=100.0
    )
    np.testing.assert_allclose(out, [100 * 0.5 * 0.46211715726000974], rtol=1e-12)


def test_zero_spend_gives_zero_contribution():
    out = weekly_contribution(
        0.0, lam=np.ones(5), beta=np.ones(5), channel_scale=10.0, target_scale=100.0
    )
    assert np.all(out == 0) and np.isfinite(out).all()


def test_simulate_plan_bands_are_ordered(fake_results):
    r = simulate_plan(fake_results, fake_results["plans"]["current"])
    for k in ("media", "total"):
        assert r[k]["p05"] <= r[k]["median"] <= r[k]["p95"]
    assert r["total"]["median"] > r["media"]["median"]


def test_simulate_plan_all_zero_is_baseline_only(fake_results):
    r = simulate_plan(fake_results, {c: 0.0 for c in fake_results["channels"]})
    assert r["media"] == {"median": 0.0, "p05": 0.0, "p95": 0.0}
    base = 13 * np.array(fake_results["draws"]["baseline_weekly"])
    assert r["total"]["median"] == np.percentile(base, 50)


def test_results_schema_and_size(fake_results, tmp_path):
    write_results(fake_results, tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert set(data) == SCHEMA_KEYS
    assert all(len(v) == 200 for v in data["draws"]["lam"].values())
    assert (tmp_path / "r.json").stat().st_size < 1_000_000


def test_simulator_cases(fake_results, tmp_path):
    write_simulator_cases(fake_results, tmp_path / "cases.json")
    data = json.loads((tmp_path / "cases.json").read_text(encoding="utf-8"))
    assert set(data["results_subset"]) == {"channels", "scales", "draws", "meta"}
    assert len(data["cases"]) == 4
    first = data["cases"][0]
    assert first["expected"] == simulate_plan(fake_results, first["plan"])
