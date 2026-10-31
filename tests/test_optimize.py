import pytest

from mmm.export import simulate_plan
from mmm.optimize import optimal_plan


@pytest.mark.slow
def test_smoke_results_are_complete(smoke_results):
    assert len(smoke_results["holdout"]) == 12
    assert len(smoke_results["contributions"]) == 196 * 6
    assert all(r["p05"] <= r["median"] <= r["p95"] for r in smoke_results["roi"])
    assert all(v > 0 for v in smoke_results["draws"]["baseline_weekly"])


@pytest.mark.slow
def test_optimal_plan_same_budget_and_not_worse(smoke_model, smoke_results, last_date):
    current = smoke_results["plans"]["current"]
    spend_max = smoke_results["spend_max_weekly"]
    opt = optimal_plan(smoke_model, current, spend_max, last_date)
    assert sum(opt.values()) == pytest.approx(sum(current.values()), rel=0.005)
    assert all(0 <= opt[c] <= 1.5 * spend_max[c] for c in opt)
    cur = simulate_plan(smoke_results, current)["media"]["median"]
    new = simulate_plan(smoke_results, opt)["media"]["median"]
    assert new >= cur * 0.999
