"""Arma, entrena y usa el MMM con PyMC-Marketing."""

import pandas as pd
import xarray as xr
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

from mmm.data import CHANNELS

HOLDOUT_WEEKS = 12
SEED = 42
TARGET = "revenue"


def split_train_test(
    df: pd.DataFrame, holdout_weeks: int = HOLDOUT_WEEKS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.iloc[:-holdout_weeks].copy(), df.iloc[-holdout_weeks:].copy()


def build_mmm(controls: list[str]) -> MMM:
    return MMM(
        date_column="DATE",
        channel_columns=CHANNELS,
        target_column=TARGET,
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        control_columns=controls,
        yearly_seasonality=2,
    )


def model_inputs(mmm: MMM, df: pd.DataFrame) -> pd.DataFrame:
    return df[["DATE", *CHANNELS, *mmm.control_columns]]


def fit_mmm(
    mmm: MMM,
    train: pd.DataFrame,
    *,
    chains: int = 4,
    draws: int = 1000,
    tune: int = 1000,
    sampler: str = "nutpie",
    target_accept: float = 0.9,
) -> xr.DataTree:
    return mmm.fit(
        model_inputs(mmm, train),
        train[TARGET],
        nuts_sampler=sampler,
        chains=chains,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        random_seed=SEED,
        progressbar=False,
    )


def predict_holdout(mmm: MMM, test: pd.DataFrame) -> xr.DataArray:
    pred = mmm.sample_posterior_predictive(
        model_inputs(mmm, test),
        include_last_observations=True,
        extend_idata=False,
        random_seed=SEED,
        progressbar=False,
    )
    # PyMC-Marketing devuelve "y" en la escala interna (ventas / máximo de entrenamiento).
    target_scale = float(mmm.get_scales_as_xarray()["target_scale"])
    return (pred["y"] * target_scale).transpose("sample", "date")
