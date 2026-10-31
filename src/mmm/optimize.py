"""Reparto óptimo del presupuesto con el optimizador de PyMC-Marketing."""

import numpy as np
import pandas as pd
import xarray as xr
from pymc_marketing.mmm import MMM, BudgetOptimizerWrapper

from mmm.data import CHANNELS
from mmm.export import HORIZON_WEEKS

MAX_SPEND_FACTOR = 1.5


def optimal_plan(
    mmm: MMM,
    current: dict[str, float],
    spend_max_weekly: dict[str, float],
    last_date: pd.Timestamp,
) -> dict[str, float]:
    """Inversión semanal por canal con el mismo total semanal que el plan actual."""
    start = pd.Timestamp(last_date) + pd.Timedelta(weeks=1)
    end = start + pd.Timedelta(weeks=HORIZON_WEEKS - 1)
    wrapper = BudgetOptimizerWrapper(
        model=mmm, start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d")
    )
    num_periods = wrapper.num_periods if hasattr(wrapper, "num_periods") else HORIZON_WEEKS

    # PyMC-Marketing reparte el presupuesto total del período entre las semanas.
    weekly_total = sum(current[c] for c in CHANNELS)
    upper = [MAX_SPEND_FACTOR * spend_max_weekly[c] * num_periods for c in CHANNELS]
    bounds = xr.DataArray(
        np.array([[0.0, u] for u in upper]),
        dims=("channel", "bound"),
        coords={"channel": CHANNELS, "bound": ["lower", "upper"]},
    )
    result = wrapper.optimize_budget(budget=weekly_total * num_periods, budget_bounds=bounds)
    budgets = result.budgets
    return {c: float(budgets.sel(channel=c)) / num_periods for c in CHANNELS}
