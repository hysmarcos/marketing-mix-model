import numpy as np
import pytest

from mmm.data import load_raw, prepare_features
from mmm.model import build_mmm, fit_mmm, predict_holdout, split_train_test


@pytest.fixture(scope="module")
def prepared():
    return prepare_features(load_raw())


def test_split_keeps_last_12_weeks_out(prepared):
    df, _ = prepared
    train, test = split_train_test(df)
    assert len(train) == 196 and len(test) == 12
    assert train["DATE"].max() < test["DATE"].min()


@pytest.mark.slow
def test_smoke_fit_and_predict(prepared):
    df, controls = prepared
    train, test = split_train_test(df)
    mmm = build_mmm(controls)
    idata = fit_mmm(mmm, train, chains=1, draws=50, tune=50)
    assert "posterior" in idata.children
    pred = predict_holdout(mmm, test)
    assert pred.sizes["date"] == 12
    assert np.isfinite(pred.values).all()
    # escala original: las ventas reales rondan los millones
    assert 1e5 < float(pred.median()) < 1e8
