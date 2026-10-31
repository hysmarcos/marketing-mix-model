import json

import numpy as np
import pytest

from mmm.data import CHANNELS
from mmm.meridian_compare import MERIDIAN_PATH, roi_bands


def test_roi_bands_per_channel():
    rng = np.random.default_rng(42)
    # Forma (cadena, muestra, canal).
    samples = rng.normal(loc=[1, 2, 3, 4, 5], scale=0.1, size=(4, 250, 5))
    bands = roi_bands(samples, CHANNELS)
    assert [b["channel"] for b in bands] == CHANNELS
    for b, expected in zip(bands, [1, 2, 3, 4, 5], strict=True):
        assert b["p05"] < b["median"] < b["p95"]
        assert b["median"] == pytest.approx(expected, abs=0.02)


@pytest.mark.skipif(not MERIDIAN_PATH.exists(), reason="falta meridian.json")
def test_published_meridian_results():
    data = json.loads(MERIDIAN_PATH.read_text(encoding="utf-8"))
    assert set(data) == {"meta", "roi", "diagnostics"}
    assert [r["channel"] for r in data["roi"]] == CHANNELS
    assert all(r["p05"] <= r["median"] <= r["p95"] for r in data["roi"])
    assert data["diagnostics"]["max_rhat"] < 1.2
    assert 0 < data["diagnostics"]["holdout_mape"] < 1


def test_input_data_matches_dataset(prepared_data):
    pytest.importorskip("meridian")
    from mmm.meridian_compare import build_input_data

    df, controls, _, _ = prepared_data
    data = build_input_data(df, controls)
    assert data.kpi.shape[-1] == 208
    assert list(data.media_channel.values) == CHANNELS
