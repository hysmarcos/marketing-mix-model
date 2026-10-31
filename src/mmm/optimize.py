"""Reparto óptimo del presupuesto semanal.

Maximiza la misma función que usa el simulador de la página: las ventas de medios por
semana en estado estable, promediadas sobre las muestras del modelo. Así el reparto
óptimo es óptimo justo en los números que ve el visitante.
"""

import numpy as np
from scipy.optimize import minimize

MAX_SPEND_FACTOR = 1.5


def optimal_plan(results: dict) -> dict[str, float]:
    """Inversión semanal por canal con el mismo total semanal que el plan actual."""
    channels = results["channels"]
    draws, scales = results["draws"], results["scales"]
    current = np.array([results["plans"]["current"][c] for c in channels])
    total = current.sum()

    # Se trabaja en fracciones del total para que el optimizador vea números de orden 1.
    lam = np.array([draws["lam"][c] for c in channels])  # (canal, muestra)
    beta = np.array([draws["beta"][c] for c in channels])
    k = lam * total / np.array([scales["channel"][c] for c in channels])[:, None]
    upper = np.array([MAX_SPEND_FACTOR * results["spend_max_weekly"][c] for c in channels])

    def negative_media(u: np.ndarray) -> tuple[float, np.ndarray]:
        h = np.tanh(k * u[:, None] / 2)  # logistic(z) = tanh(z / 2)
        value = (beta * h).sum(axis=0).mean()
        grad = (beta * k * 0.5 * (1 - h**2)).mean(axis=1)
        return -value, -grad

    bounds = [(0.0, ub / total) for ub in upper]
    constraint = {"type": "eq", "fun": lambda u: u.sum() - 1, "jac": lambda u: np.ones_like(u)}
    starts = [current / total, np.full(len(channels), 1 / len(channels)), upper / upper.sum()]
    best = None
    for start in starts:
        start = np.clip(start, 0, upper / total)
        fit = minimize(
            negative_media,
            start / start.sum(),
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if fit.success and (best is None or fit.fun < best.fun):
            best = fit
    if best is None:
        raise RuntimeError("El optimizador no encontró un reparto válido.")
    shares = np.clip(best.x, 0, None)
    return {c: float(s * total) for c, s in zip(channels, shares / shares.sum(), strict=True)}
