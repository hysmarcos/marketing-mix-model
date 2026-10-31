"""Comparación con Meridian, el MMM de Google.

Entrena Meridian con los mismos datos, los mismos controles y las mismas 12 semanas de
prueba que el modelo de PyMC-Marketing, y guarda su ROI por canal y sus controles de
calidad en site/data/meridian.json.

Meridian y PyMC-Marketing piden versiones distintas de ArviZ, así que Meridian corre en
su propio entorno con `uv run --script scripts/run_meridian.py`. Este módulo no importa
PyMC-Marketing.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mmm.data import CHANNELS, HOLDOUT_WEEKS, SEED, TARGET

QUANTILES = (5, 50, 95)

MERIDIAN_PATH = Path(__file__).resolve().parents[2] / "site" / "data" / "meridian.json"


def roi_bands(samples: np.ndarray, channels: list[str]) -> list[dict]:
    """Mediana y rango del 90% por canal. `samples` tiene forma (cadena, muestra, canal)."""
    flat = np.asarray(samples, dtype=float).reshape(-1, len(channels))
    p05, median, p95 = np.percentile(flat, QUANTILES, axis=0)
    return [
        {"channel": c, "median": float(m), "p05": float(lo), "p95": float(hi)}
        for c, m, lo, hi in zip(channels, median, p05, p95, strict=True)
    ]


def build_input_data(df: pd.DataFrame, controls: list[str]):
    from meridian.data import data_frame_input_data_builder as builder

    data = df.assign(time=df["DATE"].dt.strftime("%Y-%m-%d"))
    return (
        builder.DataFrameInputDataBuilder(kpi_type="revenue")
        .with_kpi(data, kpi_col=TARGET, time_col="time")
        .with_controls(data, control_cols=controls, time_col="time")
        .with_media(
            data,
            media_cols=CHANNELS,
            media_spend_cols=CHANNELS,
            media_channels=CHANNELS,
            time_col="time",
        )
        .build()
    )


def fit_meridian(
    input_data,
    n_weeks: int,
    *,
    n_chains: int = 4,
    n_adapt: int = 1000,
    n_burnin: int = 500,
    n_keep: int = 1000,
):
    from meridian.model import model, spec

    holdout = np.zeros(n_weeks, dtype=bool)
    holdout[-HOLDOUT_WEEKS:] = True
    meridian = model.Meridian(
        input_data=input_data, model_spec=spec.ModelSpec(max_lag=8, holdout_id=holdout)
    )
    meridian.sample_posterior(
        n_chains=n_chains, n_adapt=n_adapt, n_burnin=n_burnin, n_keep=n_keep, seed=SEED
    )
    return meridian


def summarize(meridian, df: pd.DataFrame) -> dict:
    import meridian as meridian_pkg
    from meridian.analysis import analyzer

    analysis = analyzer.Analyzer(meridian)
    times = df["DATE"].dt.strftime("%Y-%m-%d").tolist()
    train_times = times[:-HOLDOUT_WEEKS]

    roi = np.asarray(analysis.roi(selected_times=train_times))
    predicted = np.asarray(analysis.expected_outcome(aggregate_times=False))
    predicted = predicted.reshape(-1, len(times))[:, -HOLDOUT_WEEKS:]
    actual = df[TARGET].tail(HOLDOUT_WEEKS).to_numpy()
    mape = float(np.mean(np.abs(actual - np.median(predicted, axis=0)) / actual))
    max_rhat = max(float(np.nanmax(np.asarray(v))) for v in analysis.get_rhat().values())

    return {
        "meta": {
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "meridian_version": meridian_pkg.__version__,
            "holdout_weeks": HOLDOUT_WEEKS,
        },
        "roi": roi_bands(roi, CHANNELS),
        "diagnostics": {"max_rhat": max_rhat, "holdout_mape": mape},
    }


def run(path: Path = MERIDIAN_PATH) -> dict:
    from mmm.data import load_raw, prepare_features

    df, controls = prepare_features(load_raw())
    meridian = fit_meridian(build_input_data(df, controls), len(df))
    results = summarize(meridian, df)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results
