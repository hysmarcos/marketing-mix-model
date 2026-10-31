import numpy as np
import pytest

from mmm.export import media_draws, simulate_plan
from mmm.optimize import MAX_SPEND_FACTOR, optimal_plan


def mean_media(results, plan):
    return media_draws(results, plan).mean()


def test_optimal_plan_same_budget_and_bounds(fake_results):
    opt = optimal_plan(fake_results)
    current = fake_results["plans"]["current"]
    assert sum(opt.values()) == pytest.approx(sum(current.values()), rel=1e-6)
    upper = fake_results["spend_max_weekly"]
    assert all(0 <= opt[c] <= MAX_SPEND_FACTOR * upper[c] * (1 + 1e-9) for c in opt)


def test_optimal_plan_beats_random_reallocations(fake_results):
    opt = optimal_plan(fake_results)
    best = mean_media(fake_results, opt)
    channels = fake_results["channels"]
    total = sum(fake_results["plans"]["current"].values())
    upper = np.array([MAX_SPEND_FACTOR * fake_results["spend_max_weekly"][c] for c in channels])
    rng = np.random.default_rng(7)
    for _ in range(300):
        spend = np.minimum(rng.dirichlet(np.ones(len(channels))) * total, upper)
        if spend.sum() < total * 0.999:
            continue
        assert mean_media(fake_results, dict(zip(channels, spend, strict=True))) <= best * 1.0001


@pytest.mark.slow
def test_smoke_results_are_complete(smoke_results):
    assert len(smoke_results["holdout"]) == 12
    assert len(smoke_results["contributions"]) == 196 * 6
    assert all(r["p05"] <= r["median"] <= r["p95"] for r in smoke_results["roi"])
    assert all(v > 0 for v in smoke_results["draws"]["baseline_weekly"])
    current = simulate_plan(smoke_results, smoke_results["plans"]["current"])["media"]["median"]
    optimal = simulate_plan(smoke_results, smoke_results["plans"]["optimal"])["media"]["median"]
    assert optimal >= current * 0.999


def test_optimal_plan_accounts_for_carryover(fake_results):
    channels = fake_results["channels"]
    # Canales con arrastre muy distinto y sin inversión previa: el estado estable ya no sirve.
    for i, c in enumerate(channels):
        fake_results["draws"]["alpha"][c] = [0.05 + 0.2 * i] * 200
        fake_results["history"][c] = [0.0] * 7
    opt = optimal_plan(fake_results)
    best = mean_media(fake_results, opt)
    step = 0.01 * sum(opt.values())
    # Ningún traspaso chico entre dos canales mejora el resultado (condición de óptimo local).
    for a in channels:
        for b in channels:
            if a == b or opt[a] < step:
                continue
            moved = {**opt, a: opt[a] - step, b: opt[b] + step}
            if moved[b] > MAX_SPEND_FACTOR * fake_results["spend_max_weekly"][b]:
                continue
            assert mean_media(fake_results, moved) <= best * (1 + 1e-7), (a, b)
