"""Controles de calidad del modelo."""

from dataclasses import asdict, dataclass

import arviz as az
import numpy as np
import xarray as xr

MAX_RHAT = 1.01
MIN_ESS = 400


@dataclass(frozen=True)
class Diagnostics:
    divergences: int
    max_rhat: float
    min_ess: float
    holdout_mape: float

    def to_dict(self) -> dict:
        return asdict(self)


class ModelQualityError(RuntimeError):
    """El modelo no pasa un control de calidad."""


def _as_dataset(stat) -> xr.Dataset:
    return stat.to_dataset() if isinstance(stat, xr.DataTree) else stat


def _extreme(stat: xr.Dataset, reducer) -> float:
    return float(reducer([float(reducer(v.values)) for v in stat.data_vars.values()]))


def compute_diagnostics(
    idata: xr.DataTree,
    actual: np.ndarray,
    predicted_median: np.ndarray,
    var_names: list[str] | None = None,
) -> Diagnostics:
    posterior = idata["posterior"].to_dataset()
    if var_names is not None:
        posterior = posterior[var_names]
    rhat = _as_dataset(az.rhat(posterior))
    ess = _as_dataset(az.ess(posterior, method="bulk"))
    actual = np.asarray(actual, dtype=float)
    predicted_median = np.asarray(predicted_median, dtype=float)
    return Diagnostics(
        divergences=int(idata["sample_stats"]["diverging"].sum()),
        max_rhat=_extreme(rhat, np.max),
        min_ess=_extreme(ess, np.min),
        holdout_mape=float(np.mean(np.abs(actual - predicted_median) / actual)),
    )


def assert_quality(d: Diagnostics) -> None:
    if d.divergences > 0:
        raise ModelQualityError(f"El sampler tuvo {d.divergences} divergencias (deben ser 0).")
    if d.max_rhat >= MAX_RHAT:
        raise ModelQualityError(f"R-hat máximo {d.max_rhat:.4f} (debe ser menor que {MAX_RHAT}).")
    if d.min_ess <= MIN_ESS:
        raise ModelQualityError(f"ESS mínimo {d.min_ess:.0f} (debe ser mayor que {MIN_ESS}).")
