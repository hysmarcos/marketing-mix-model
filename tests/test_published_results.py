"""Tests sobre site/data/resultados.json, lo que ve el visitante de la página."""

import json

import numpy as np
import pytest

from mmm.cli import RESULTS_PATH
from mmm.data import load_raw
from mmm.export import media_draws

pytestmark = pytest.mark.skipif(not RESULTS_PATH.exists(), reason="falta resultados.json")


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def mean_media(results, plan):
    return float(media_draws(results, plan).mean())


def test_optimal_plan_is_not_beaten_by_random_reallocations(results):
    channels = results["channels"]
    total = sum(results["plans"]["current"].values())
    upper = np.array([1.5 * results["spend_max_weekly"][c] for c in channels])
    best = mean_media(results, results["plans"]["optimal"])
    rng = np.random.default_rng(42)
    for _ in range(500):
        shares = rng.dirichlet(np.ones(len(channels)))
        spend = np.minimum(shares * total, upper)
        if spend.sum() < total * 0.999:
            continue
        plan = dict(zip(channels, spend, strict=True))
        assert mean_media(results, plan) <= best * 1.001


def test_total_sales_match_the_recent_quarter(results):
    recent = load_raw()["revenue"].tail(results["meta"]["horizon_weeks"]).mean()
    baseline = float(np.median(results["draws"]["baseline_weekly"]))
    weeks = results["meta"]["horizon_weeks"]
    media = mean_media(results, results["plans"]["current"]) / weeks
    assert baseline + media == pytest.approx(recent, rel=0.25)
