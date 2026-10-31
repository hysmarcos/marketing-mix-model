"""Simulador de referencia y exportación de resultados para la página."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pymc_marketing.mmm import MMM

from mmm.data import CHANNEL_LABELS, CHANNELS
from mmm.diagnostics import Diagnostics
from mmm.model import HOLDOUT_WEEKS, SEED, TARGET, model_inputs

N_DRAWS = 200
HORIZON_WEEKS = 13
QUANTILES = (5, 50, 95)


def logistic(z: np.ndarray) -> np.ndarray:
    return (1 - np.exp(-z)) / (1 + np.exp(-z))


def weekly_contribution(
    spend: float,
    lam: np.ndarray,
    beta: np.ndarray,
    channel_scale: float,
    target_scale: float,
) -> np.ndarray:
    """Aporte semanal en estado estable, una entrada por muestra del modelo."""
    lam, beta = np.asarray(lam, dtype=float), np.asarray(beta, dtype=float)
    return target_scale * beta * logistic(lam * spend / channel_scale)


def band(values: np.ndarray) -> dict:
    p05, median, p95 = np.percentile(np.asarray(values, dtype=float), QUANTILES)
    return {"median": float(median), "p05": float(p05), "p95": float(p95)}


def simulate_plan(results: dict, plan: dict[str, float]) -> dict:
    draws, scales = results["draws"], results["scales"]
    weeks = results["meta"]["horizon_weeks"]
    media = sum(
        weekly_contribution(
            plan[c],
            draws["lam"][c],
            draws["beta"][c],
            scales["channel"][c],
            scales["target"],
        )
        for c in results["channels"]
    )
    media = weeks * np.asarray(media)
    total = media + weeks * np.asarray(draws["baseline_weekly"], dtype=float)
    return {"media": band(media), "total": band(total)}


def _stack(idata: xr.DataTree) -> xr.Dataset:
    return idata["posterior"].to_dataset().stack(sample=("chain", "draw"))


def _base_contribution(post: xr.Dataset) -> xr.DataArray:
    """Todo lo que no es un canal: intercepto, controles y estacionalidad (escala interna)."""
    control = post["control_contribution"]
    control = control.sum([d for d in control.dims if d not in ("date", "sample")])
    return post["intercept_contribution"] + control + post["yearly_seasonality_contribution"]


def _recent_baseline(mmm: MMM, df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """Base semanal promedio de las últimas semanas del dataset, por muestra (escala interna).

    Usa las mismas semanas que el plan actual. Para las semanas de prueba, PyMC-Marketing
    recalcula intercepto, controles y estacionalidad con las muestras del modelo.
    """
    recent = df.tail(HORIZON_WEEKS)
    pred = mmm.sample_posterior_predictive(
        model_inputs(mmm, recent),
        var_names=[
            "intercept_contribution",
            "control_contribution",
            "yearly_seasonality_contribution",
        ],
        extend_idata=False,
        random_seed=SEED,
        progressbar=False,
    )
    base = _base_contribution(pred).transpose("date", "sample")
    return base.mean("date").values[idx]


def _date(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def build_results(
    mmm: MMM,
    idata: xr.DataTree,
    df: pd.DataFrame,
    train: pd.DataFrame,
    holdout_pred: xr.DataArray,
    diagnostics: Diagnostics,
) -> dict:
    post = _stack(idata)
    n = post.sizes["sample"]
    idx = np.random.default_rng(SEED).choice(n, N_DRAWS, replace=n < N_DRAWS)
    picked = post.isel(sample=idx)

    scales = mmm.get_scales_as_xarray()
    target_scale = float(scales["target_scale"])
    channel_scale = scales["channel_scale"]

    channel = post["channel_contribution"] * target_scale
    base = _base_contribution(post) * target_scale
    base = base.transpose("date", "sample")

    contributions = []
    for date in channel["date"].values:
        contributions.append(
            {"date": _date(date), "series": "base", "value": float(base.sel(date=date).median())}
        )
        for c in CHANNELS:
            value = float(channel.sel(date=date, channel=c).median())
            contributions.append({"date": _date(date), "series": c, "value": value})

    roi = []
    for c in CHANNELS:
        per_sample = channel.sel(channel=c).sum("date").values / train[c].sum()
        roi.append({"channel": c, **band(per_sample)})

    holdout = []
    for date in holdout_pred["date"].values:
        actual = float(df.loc[df["DATE"] == pd.Timestamp(date), TARGET].item())
        holdout.append({"date": _date(date), "actual": actual, **band(holdout_pred.sel(date=date))})

    baseline = _recent_baseline(mmm, df, idx) * target_scale

    results = {
        "meta": {
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "Robyn dt_simulated_weekly",
            "currency": "$",
            "horizon_weeks": HORIZON_WEEKS,
            "n_draws": N_DRAWS,
            "holdout_weeks": HOLDOUT_WEEKS,
        },
        "channels": CHANNELS,
        "channel_labels": CHANNEL_LABELS,
        "scales": {
            "channel": {c: float(channel_scale.sel(channel=c)) for c in CHANNELS},
            "target": target_scale,
        },
        "draws": {
            "lam": {c: picked["saturation_lam"].sel(channel=c).values.tolist() for c in CHANNELS},
            "beta": {c: picked["saturation_beta"].sel(channel=c).values.tolist() for c in CHANNELS},
            "alpha": {c: picked["adstock_alpha"].sel(channel=c).values.tolist() for c in CHANNELS},
            "baseline_weekly": baseline.tolist(),
        },
        "spend_max_weekly": {c: float(df[c].max()) for c in CHANNELS},
        "plans": {
            "current": {c: float(df[c].tail(HORIZON_WEEKS).mean()) for c in CHANNELS},
            "optimal": {},
        },
        "contributions": contributions,
        "roi": roi,
        "holdout": holdout,
        "diagnostics": diagnostics.to_dict(),
    }
    from mmm.optimize import optimal_plan

    results["plans"]["optimal"] = optimal_plan(results)
    current = simulate_plan(results, results["plans"]["current"])["media"]["median"]
    optimal = simulate_plan(results, results["plans"]["optimal"])["media"]["median"]
    results["headline"] = {"media_uplift_pct": (optimal / current - 1) * 100}
    return results


def _write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def write_results(results: dict, path: Path) -> None:
    _write_json(results, path)


def write_simulator_cases(results: dict, path: Path) -> None:
    channels = results["channels"]
    first = channels[0]
    plans = [
        results["plans"]["current"],
        results["plans"]["optimal"],
        {c: 0.0 for c in channels},
        {c: (1.5 * results["spend_max_weekly"][c] if c == first else 0.0) for c in channels},
    ]
    subset = {k: results[k] for k in ("channels", "scales", "draws", "meta")}
    cases = [{"plan": plan, "expected": simulate_plan(results, plan)} for plan in plans]
    _write_json({"results_subset": subset, "cases": cases}, path)
