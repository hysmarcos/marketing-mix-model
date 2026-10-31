from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from mmm.diagnostics import Diagnostics, ModelQualityError, assert_quality, compute_diagnostics

GOOD = Diagnostics(divergences=0, max_rhat=1.005, min_ess=800, holdout_mape=0.05)


def make_idata(chain_means: list[float], diverging: int = 0) -> xr.DataTree:
    rng = np.random.default_rng(42)
    draws = 500
    values = np.stack([rng.normal(m, 1, draws) for m in chain_means])
    posterior = xr.Dataset({"theta": (("chain", "draw"), values)})
    div = np.zeros((len(chain_means), draws), dtype=bool)
    div.flat[:diverging] = True
    sample_stats = xr.Dataset({"diverging": (("chain", "draw"), div)})
    return xr.DataTree.from_dict({"posterior": posterior, "sample_stats": sample_stats})


def test_assert_quality_passes_on_good_values():
    assert_quality(GOOD)


@pytest.mark.parametrize(
    "field,value,name",
    [("divergences", 3, "divergencias"), ("max_rhat", 1.02, "R-hat"), ("min_ess", 120, "ESS")],
)
def test_assert_quality_fails(field, value, name):
    with pytest.raises(ModelQualityError, match=name):
        assert_quality(replace(GOOD, **{field: value}))


def test_rhat_detects_disagreeing_chains():
    d = compute_diagnostics(make_idata([0, 5]), actual=np.ones(12), predicted_median=np.ones(12))
    assert d.max_rhat > 1.01


def test_good_chains_and_divergence_count():
    d = compute_diagnostics(
        make_idata([0, 0, 0, 0], diverging=2), actual=np.ones(3), predicted_median=np.ones(3)
    )
    assert d.max_rhat < 1.01
    assert d.min_ess > 400
    assert d.divergences == 2


def test_mape():
    d = compute_diagnostics(
        make_idata([0, 0]),
        actual=np.array([100.0, 200.0]),
        predicted_median=np.array([110.0, 180.0]),
    )
    assert d.holdout_mape == pytest.approx(0.10)
