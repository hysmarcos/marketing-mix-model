"""Pruebas para saber si el modelo es confiable, más allá de los controles del sampler.

1. Correlaciones y VIF de los datos: si los canales se mueven juntos, el modelo no puede
   separar sus efectos.
2. Prior contra posterior del ROI: si el posterior se parece al prior, el resultado sale
   de los supuestos y no de los datos.
3. Recuperación de parámetros: se inventan ventas con efectos conocidos, se entrena el
   modelo con esas ventas y se mira si encuentra los efectos.
"""

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc_marketing.mmm import MMM

from mmm.data import CHANNELS, SEED, TARGET
from mmm.export import band
from mmm.model import build_mmm, fit_mmm

CORRELATION_COLUMNS = [TARGET, *CHANNELS, "newsletter", "competitor_sales_B"]


def variance_inflation(df: pd.DataFrame) -> dict[str, float]:
    """VIF de cada columna: 1 / (1 - R²) al explicarla con las demás. Más de 5 es alto."""
    z = (df - df.mean()) / df.std()
    out = {}
    for column in z.columns:
        others = np.c_[np.ones(len(z)), z.drop(columns=column).to_numpy()]
        y = z[column].to_numpy()
        coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        r2 = 1 - ((y - others @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[column] = float(1 / (1 - r2))
    return out


def correlation_summary(df: pd.DataFrame) -> dict:
    corr = df[CORRELATION_COLUMNS].corr()
    return {
        "columns": CORRELATION_COLUMNS,
        "matrix": corr.round(4).to_numpy().tolist(),
        "zero_weeks": {c: int((df[c] == 0).sum()) for c in CHANNELS},
        "vif": variance_inflation(df[CORRELATION_COLUMNS[1:]]),
    }


def roi_from_contributions(
    contributions: xr.DataArray, target_scale: float, spend: pd.DataFrame
) -> xr.DataArray:
    """ROI por muestra y canal. `contributions` está en escala interna, con dim `date`."""
    total = contributions.sum("date") * target_scale
    spent = xr.DataArray(
        [spend[c].sum() for c in contributions["channel"].values],
        dims="channel",
        coords={"channel": contributions["channel"].values},
    )
    return total / spent


def _target_scale(mmm: MMM) -> float:
    return float(mmm.get_scales_as_xarray()["target_scale"])


def prior_posterior_roi(mmm: MMM, train: pd.DataFrame, draws: int = 1000) -> list[dict]:
    """Rango del ROI antes de ver los datos (prior) y después (posterior), por canal."""
    with mmm.model:
        prior = pm.sample_prior_predictive(
            draws=draws, var_names=["channel_contribution"], random_seed=SEED
        )
    scale = _target_scale(mmm)
    stack = {"sample": ("chain", "draw")}
    prior_roi = roi_from_contributions(
        prior["prior"]["channel_contribution"].stack(stack), scale, train
    )
    post_roi = roi_from_contributions(
        mmm.idata["posterior"]["channel_contribution"].stack(stack), scale, train
    )
    return [
        {
            "channel": c,
            "prior": band(prior_roi.sel(channel=c).values),
            "posterior": band(post_roi.sel(channel=c).values),
        }
        for c in CHANNELS
    ]


def recovery_run(
    mmm: MMM,
    train: pd.DataFrame,
    controls: list[str],
    seed: int,
    *,
    chains: int = 4,
    draws: int = 1000,
    tune: int = 1000,
) -> list[dict]:
    """Inventa ventas con los parámetros del modelo como verdad y entrena un modelo nuevo.

    La verdad es la mediana del posterior de cada parámetro libre. Las ventas inventadas
    usan la inversión real, así el test tiene los mismos huecos y picos que los datos.
    """
    posterior = mmm.idata["posterior"].to_dataset()
    truth = {
        rv.name: posterior[rv.name].median(("chain", "draw")).values for rv in mmm.model.free_RVs
    }
    with pm.do(mmm.model, truth):
        sim = pm.sample_prior_predictive(
            draws=1, var_names=["y", "channel_contribution"], random_seed=seed
        )
    scale = _target_scale(mmm)
    true_roi = roi_from_contributions(
        sim["prior"]["channel_contribution"].isel(chain=0, draw=0).expand_dims(sample=[0]),
        scale,
        train,
    ).isel(sample=0)
    y = sim["prior_predictive"]["y"].isel(chain=0, draw=0).values * scale

    synthetic = train.copy()
    synthetic[TARGET] = y
    new = build_mmm(controls)
    idata = fit_mmm(new, synthetic, chains=chains, draws=draws, tune=tune)
    recovered = roi_from_contributions(
        idata["posterior"]["channel_contribution"].stack(sample=("chain", "draw")),
        _target_scale(new),
        synthetic,
    )
    return [
        {"channel": c, "true": float(true_roi.sel(channel=c)), **band(recovered.sel(channel=c))}
        for c in CHANNELS
    ]


def summarize_recovery(runs: list[list[dict]]) -> list[dict]:
    """Por canal: verdad, mediana promedio y en cuántas corridas el rango del 90% la cubre."""
    out = []
    for c in dict.fromkeys(r["channel"] for run in runs for r in run):
        rows = [r for run in runs for r in run if r["channel"] == c]
        out.append(
            {
                "channel": c,
                "true": float(np.mean([r["true"] for r in rows])),
                "median": float(np.mean([r["median"] for r in rows])),
                "p05": float(np.mean([r["p05"] for r in rows])),
                "p95": float(np.mean([r["p95"] for r in rows])),
                "covered": sum(r["p05"] <= r["true"] <= r["p95"] for r in rows),
                "n": len(rows),
            }
        )
    return out
