"""Gráficos de matplotlib para los notebooks y el README. Misma paleta que la página."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from mmm.data import CHANNEL_LABELS, CHANNELS
from mmm.export import band, weekly_contribution

ROOT = Path(__file__).resolve().parents[2]

# Paleta categórica validada para daltonismo y contraste (modo claro). Debe coincidir con site/css/styles.css.
PALETTE = {
    "tv_S": "#2a78d6",
    "ooh_S": "#eb6834",
    "print_S": "#1baf7a",
    "facebook_S": "#eda100",
    "search_S": "#e87ba4",
}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 1,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "ytick.left": False,
        "text.color": INK,
        "font.size": 10,
        "legend.frameon": False,
    }
)


def _money(v: float, _=None) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"$ {v / 1e6:.1f} M".replace(".", ",")
    if a >= 1e3:
        return f"$ {v / 1e3:.0f} k"
    return f"$ {v:.0f}"


MONEY = FuncFormatter(_money)


def _title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=24)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=10)


def plot_revenue(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(df["DATE"], df["revenue"], color=INK, linewidth=2)
    ax.yaxis.set_major_formatter(MONEY)
    _title(ax, "Ventas por semana", "208 semanas, desde noviembre de 2015")
    fig.tight_layout()
    return fig


def plot_spend(df: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(10, 8), sharex=True)
    for ax, c in zip(axes, CHANNELS, strict=True):
        ax.fill_between(df["DATE"], df[c], color=PALETTE[c], alpha=0.85, linewidth=0)
        ax.yaxis.set_major_formatter(MONEY)
        ax.text(0, 1.02, CHANNEL_LABELS[c], transform=ax.transAxes, color=INK, fontsize=10)
    fig.suptitle("Inversión semanal por canal", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_roi(results: dict) -> Figure:
    rows = sorted(results["roi"], key=lambda r: r["median"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, r in enumerate(rows):
        color = PALETTE[r["channel"]]
        ax.plot([r["p05"], r["p95"]], [i, i], color=color, linewidth=2, solid_capstyle="round")
        ax.scatter(r["median"], i, s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(r["p95"], i, f"  {r['median']:.1f}".replace(".", ","), va="center", color=INK)
    ax.axvline(1, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(rows)), [CHANNEL_LABELS[r["channel"]] for r in rows])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Ventas por cada peso invertido (mediana y rango del 90%)")
    _title(ax, "Retorno por canal")
    fig.tight_layout()
    return fig


def plot_contributions(results: dict) -> Figure:
    df = pd.DataFrame(results["contributions"])
    df = df[df["series"] != "base"]
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    wide = df.pivot_table(index="month", columns="series", values="value", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(wide))
    for c in CHANNELS:
        ax.bar(
            wide.index,
            wide[c],
            bottom=bottom,
            width=24,
            color=PALETTE[c],
            edgecolor=SURFACE,
            linewidth=0.8,
            label=CHANNEL_LABELS[c],
        )
        bottom += wide[c].to_numpy()
    ax.yaxis.set_major_formatter(MONEY)
    ax.legend(ncols=5, loc="upper left", bbox_to_anchor=(0, 1.0))
    _title(ax, "Aporte mensual de cada canal", "Ventas que el modelo atribuye a cada canal")
    ax.set_ylim(0, bottom.max() * 1.18)
    fig.tight_layout()
    return fig


def plot_holdout(results: dict) -> Figure:
    df = pd.DataFrame(results["holdout"])
    df["date"] = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    accent = PALETTE["tv_S"]
    ax.fill_between(df["date"], df["p05"], df["p95"], color=accent, alpha=0.12, linewidth=0)
    ax.plot(df["date"], df["median"], color=accent, linewidth=2, label="Predicción del modelo")
    ax.plot(df["date"], df["actual"], color=INK, linewidth=2, marker="o", label="Ventas reales")
    ax.yaxis.set_major_formatter(MONEY)
    ax.legend(loc="upper left")
    mape = results["diagnostics"]["holdout_mape"] * 100
    subtitle = f"12 semanas que el modelo no vio. Error promedio: {mape:.1f}%".replace(".", ",")
    _title(ax, "Prueba del modelo", subtitle)
    fig.tight_layout()
    return fig


def plot_saturation(results: dict) -> Figure:
    draws, scales = results["draws"], results["scales"]
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(14, 3), sharey=True)
    for ax, c in zip(axes, CHANNELS, strict=True):
        spend = np.linspace(0, 1.5 * results["spend_max_weekly"][c], 60)
        bands = [
            band(
                weekly_contribution(
                    s, draws["lam"][c], draws["beta"][c], scales["channel"][c], scales["target"]
                )
            )
            for s in spend
        ]
        median = [b["median"] for b in bands]
        ax.fill_between(
            spend,
            [b["p05"] for b in bands],
            [b["p95"] for b in bands],
            color=PALETTE[c],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(spend, median, color=PALETTE[c], linewidth=2)
        current = results["plans"]["current"][c]
        point = band(
            weekly_contribution(
                current, draws["lam"][c], draws["beta"][c], scales["channel"][c], scales["target"]
            )
        )
        ax.scatter(current, point["median"], s=50, color=PALETTE[c], edgecolor=SURFACE, zorder=3)
        ax.set_title(CHANNEL_LABELS[c], loc="left", fontsize=11)
        ax.xaxis.set_major_formatter(MONEY)
        ax.yaxis.set_major_formatter(MONEY)
        ax.locator_params(axis="x", nbins=3)
    fig.suptitle("Curvas de saturación (ventas por semana)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def write_readme_images(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, make in (("roi.png", plot_roi), ("aportes.png", plot_contributions)):
        fig = make(results)
        fig.savefig(out_dir / name, dpi=160)
        plt.close(fig)
