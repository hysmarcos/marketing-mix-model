"""Gráficos de matplotlib para los notebooks y el README. Misma paleta que la página.

Cada gráfico existe en dos idiomas (es, en) y dos temas (light, dark). El tema y el idioma
se eligen con `with style(lang, mode):`. Por defecto, español y claro.
"""

import textwrap
from contextlib import contextmanager
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from mmm.data import CHANNELS
from mmm.export import band, weekly_contribution

ROOT = Path(__file__).resolve().parents[2]
LANGS = ("es", "en")
MODES = ("light", "dark")

# Paleta categórica validada para daltonismo y contraste, en modo claro y oscuro.
# Debe coincidir con site/css/styles.css.
THEMES = {
    "light": {
        "series": {
            "tv_S": "#2a78d6",
            "ooh_S": "#eb6834",
            "print_S": "#1baf7a",
            "facebook_S": "#eda100",
            "search_S": "#e87ba4",
        },
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "surface": "#fcfcfb",
        "diverging": ["#2a78d6", "#f0efec", "#e34948"],
    },
    "dark": {
        "series": {
            "tv_S": "#3987e5",
            "ooh_S": "#d95926",
            "print_S": "#199e70",
            "facebook_S": "#c98500",
            "search_S": "#d55181",
        },
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "surface": "#1a1a19",
        "diverging": ["#3987e5", "#383835", "#e66767"],
    },
}
PALETTE = THEMES["light"]["series"]

TEXT = {
    "es": {
        "channels": {
            "tv_S": "TV",
            "ooh_S": "Vía pública",
            "print_S": "Prensa",
            "facebook_S": "Facebook",
            "search_S": "Búsqueda",
        },
        "variables": {
            "revenue": "Ventas",
            "newsletter": "Newsletter",
            "competitor_sales_B": "Competencia",
        },
        "roi_axis": "Ventas por cada peso invertido (mediana y rango del 90%)",
        "roi_title": "Retorno por canal",
        "revenue_title": "Ventas por semana",
        "revenue_sub": "208 semanas, desde noviembre de 2015",
        "spend_title": "Inversión semanal por canal",
        "contrib_title": "Aporte mensual de cada canal",
        "contrib_sub": "Ventas que el modelo atribuye a cada canal",
        "holdout_title": "Prueba del modelo",
        "holdout_sub": "12 semanas que el modelo no vio. Error promedio: {mape}%",
        "predicted": "Predicción del modelo",
        "actual": "Ventas reales",
        "saturation_title": "Curvas de saturación (ventas por semana)",
        "this_model": "PyMC-Marketing (este modelo)",
        "meridian": "Meridian (Google)",
        "meridian_title": "Retorno por canal: PyMC-Marketing y Meridian",
        "corr_title": "Correlación entre las variables",
        "corr_sub": "1 = se mueven juntas, 0 = no se relacionan",
        "corr_bar": "Correlación",
        "prior": "Antes de ver los datos (prior)",
        "posterior": "Después de ver los datos (posterior)",
        "pp_axis": "Ventas por cada peso invertido (escala logarítmica, rango del 90%)",
        "pp_title": "Qué aprendió el modelo de los datos",
        "estimated": "Estimado por el modelo",
        "true": "Valor real (conocido)",
        "covered": "cubierto {k} de {n}",
        "rec_axis": "Ventas por cada peso invertido (promedio de las corridas, rango del 90%)",
        "rec_title": "Recuperación de parámetros con ventas inventadas",
        "banner_kicker": "MARKETING MIX MODEL · PYMC-MARKETING · MERIDIAN",
        "banner_title": "¿Dónde rinde cada peso de publicidad?",
        "banner_kpis": [
            "más ventas de medios con el mismo presupuesto",
            "probabilidad de que el nuevo reparto mejore",
            "error en 12 semanas que el modelo no vio",
            "efectos conocidos recuperados en la prueba",
        ],
    },
    "en": {
        "channels": {
            "tv_S": "TV",
            "ooh_S": "Out-of-home",
            "print_S": "Print",
            "facebook_S": "Facebook",
            "search_S": "Search",
        },
        "variables": {
            "revenue": "Sales",
            "newsletter": "Newsletter",
            "competitor_sales_B": "Competitor",
        },
        "roi_axis": "Sales per dollar spent (median and 90% range)",
        "roi_title": "Return per channel",
        "revenue_title": "Weekly sales",
        "revenue_sub": "208 weeks, from November 2015",
        "spend_title": "Weekly spend per channel",
        "contrib_title": "Monthly contribution per channel",
        "contrib_sub": "Sales the model attributes to each channel",
        "holdout_title": "Model test",
        "holdout_sub": "12 weeks the model never saw. Mean error: {mape}%",
        "predicted": "Model prediction",
        "actual": "Actual sales",
        "saturation_title": "Saturation curves (weekly sales)",
        "this_model": "PyMC-Marketing (this model)",
        "meridian": "Meridian (Google)",
        "meridian_title": "Return per channel: PyMC-Marketing and Meridian",
        "corr_title": "Correlation between variables",
        "corr_sub": "1 = move together, 0 = unrelated",
        "corr_bar": "Correlation",
        "prior": "Before seeing the data (prior)",
        "posterior": "After seeing the data (posterior)",
        "pp_axis": "Sales per dollar spent (log scale, 90% range)",
        "pp_title": "What the model learned from the data",
        "estimated": "Model estimate",
        "true": "True value (known)",
        "covered": "covered {k} of {n}",
        "rec_axis": "Sales per dollar spent (mean over runs, 90% range)",
        "rec_title": "Parameter recovery on simulated sales",
        "banner_kicker": "MARKETING MIX MODEL · PYMC-MARKETING · MERIDIAN",
        "banner_title": "Where does each ad dollar pay off?",
        "banner_kpis": [
            "more media-driven sales on the same budget",
            "probability the new split does better",
            "error on 12 weeks the model never saw",
            "known effects recovered in the test",
        ],
    },
}

_current = {"lang": "es", "mode": "light"}


def _rc(mode: str) -> dict:
    t = THEMES[mode]
    return {
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["axis"],
        "axes.labelcolor": t["ink2"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": t["grid"],
        "grid.linewidth": 1,
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "ytick.left": False,
        "text.color": t["ink"],
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": t["ink"],
    }


plt.rcParams.update(_rc("light"))


@contextmanager
def style(lang: str = "es", mode: str = "light"):
    """Elige idioma y tema para los gráficos que se crean adentro del bloque."""
    previous = dict(_current)
    _current.update(lang=lang, mode=mode)
    try:
        with plt.rc_context(_rc(mode)):
            yield
    finally:
        _current.update(previous)


def _t() -> dict:
    return THEMES[_current["mode"]]


def _s(key: str):
    return TEXT[_current["lang"]][key]


def _label(channel: str) -> str:
    return _s("channels")[channel]


def _num(value: float, decimals: int = 1) -> str:
    text = f"{value:.{decimals}f}"
    return text.replace(".", ",") if _current["lang"] == "es" else text


def _money(v: float, _=None) -> str:
    a = abs(v)
    sep = " " if _current["lang"] == "es" else ""
    if a >= 1e6:
        return f"${sep}{_num(v / 1e6)} M" if sep else f"${_num(v / 1e6)}M"
    if a >= 1e3:
        return f"${sep}{v / 1e3:.0f} k" if sep else f"${v / 1e3:.0f}k"
    return f"${sep}{v:.0f}"


MONEY = FuncFormatter(_money)


def _title(ax, title: str, subtitle: str | None = None, pad: int = 24) -> None:
    t = _t()
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=t["ink"], pad=pad)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=t["ink2"], fontsize=10)


def _interval(ax, lo, hi, y, color, median, label=None, size=50) -> None:
    ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
    ax.scatter(median, y, s=size, color=color, edgecolor=_t()["surface"], zorder=3, label=label)


def plot_revenue(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(df["DATE"], df["revenue"], color=_t()["ink"], linewidth=2)
    ax.yaxis.set_major_formatter(MONEY)
    _title(ax, _s("revenue_title"), _s("revenue_sub"))
    fig.tight_layout()
    return fig


def plot_spend(df: pd.DataFrame) -> Figure:
    t = _t()
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(10, 8), sharex=True)
    for ax, c in zip(axes, CHANNELS, strict=True):
        ax.fill_between(df["DATE"], df[c], color=t["series"][c], alpha=0.85, linewidth=0)
        ax.yaxis.set_major_formatter(MONEY)
        ax.text(0, 1.02, _label(c), transform=ax.transAxes, color=t["ink"], fontsize=10)
    fig.suptitle(_s("spend_title"), x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_roi(results: dict) -> Figure:
    t = _t()
    rows = sorted(results["roi"], key=lambda r: r["median"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, r in enumerate(rows):
        _interval(ax, r["p05"], r["p95"], i, t["series"][r["channel"]], r["median"], size=60)
        ax.text(r["p95"], i, f"  {_num(r['median'])}", va="center", color=t["ink"])
    ax.axvline(1, color=t["axis"], linewidth=1)
    ax.set_yticks(range(len(rows)), [_label(r["channel"]) for r in rows])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(_s("roi_axis"))
    _title(ax, _s("roi_title"))
    fig.tight_layout()
    return fig


def plot_contributions(results: dict) -> Figure:
    t = _t()
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
            color=t["series"][c],
            edgecolor=t["surface"],
            linewidth=0.8,
            label=_label(c),
        )
        bottom += wide[c].to_numpy()
    ax.yaxis.set_major_formatter(MONEY)
    ax.legend(ncols=5, loc="upper left", bbox_to_anchor=(0, 1.0))
    _title(ax, _s("contrib_title"), _s("contrib_sub"))
    ax.set_ylim(0, bottom.max() * 1.18)
    fig.tight_layout()
    return fig


def plot_holdout(results: dict) -> Figure:
    t = _t()
    df = pd.DataFrame(results["holdout"])
    df["date"] = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    accent = t["series"]["tv_S"]
    ax.fill_between(df["date"], df["p05"], df["p95"], color=accent, alpha=0.12, linewidth=0)
    ax.plot(df["date"], df["median"], color=accent, linewidth=2, label=_s("predicted"))
    ax.plot(df["date"], df["actual"], color=t["ink"], linewidth=2, marker="o", label=_s("actual"))
    ax.yaxis.set_major_formatter(MONEY)
    ax.legend(loc="upper left")
    mape = _num(results["diagnostics"]["holdout_mape"] * 100)
    _title(ax, _s("holdout_title"), _s("holdout_sub").format(mape=mape))
    fig.tight_layout()
    return fig


def plot_saturation(results: dict) -> Figure:
    t = _t()
    draws, scales = results["draws"], results["scales"]
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(14, 3), sharey=True)
    for ax, c in zip(axes, CHANNELS, strict=True):
        spend = np.linspace(0, 1.5 * results["spend_max_weekly"][c], 60)

        def contribution(s, c=c):
            return band(
                weekly_contribution(
                    s, draws["lam"][c], draws["beta"][c], scales["channel"][c], scales["target"]
                )
            )

        bands = [contribution(s) for s in spend]
        color = t["series"][c]
        ax.fill_between(
            spend,
            [b["p05"] for b in bands],
            [b["p95"] for b in bands],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(spend, [b["median"] for b in bands], color=color, linewidth=2)
        current = results["plans"]["current"][c]
        point = contribution(current)
        ax.scatter(current, point["median"], s=50, color=color, edgecolor=t["surface"], zorder=3)
        ax.set_title(_label(c), loc="left", fontsize=11)
        ax.xaxis.set_major_formatter(MONEY)
        ax.yaxis.set_major_formatter(MONEY)
        ax.locator_params(axis="x", nbins=3)
    fig.suptitle(_s("saturation_title"), x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_meridian(results: dict, meridian: dict) -> Figure:
    t = _t()
    order = [r["channel"] for r in sorted(results["roi"], key=lambda r: r["median"])]
    models = [
        (_s("this_model"), results["roi"], t["series"]["tv_S"], 0.15),
        (_s("meridian"), meridian["roi"], t["series"]["ooh_S"], -0.15),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, rows, color, offset in models:
        by_channel = {r["channel"]: r for r in rows}
        for i, c in enumerate(order):
            r = by_channel[c]
            label = name if i == 0 else None
            _interval(ax, r["p05"], r["p95"], i + offset, color, r["median"], label)
    ax.axvline(1, color=t["axis"], linewidth=1)
    ax.set_yticks(range(len(order)), [_label(c) for c in order])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(_s("roi_axis"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2)
    _title(ax, _s("meridian_title"), pad=40)
    fig.tight_layout()
    return fig


def plot_correlation(validation: dict) -> Figure:
    t = _t()
    corr = validation["correlation"]
    names = {**_s("variables"), **_s("channels")}
    labels = [names[c] for c in corr["columns"]]
    matrix = np.array(corr["matrix"])
    cmap = LinearSegmentedColormap.from_list("diverging", t["diverging"])
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            strong = abs(value) > 0.6
            color = "#fcfcfb" if strong else t["ink"]
            ax.text(j, i, _num(value, 2), ha="center", va="center", color=color, fontsize=9)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label(_s("corr_bar"), color=t["ink2"])
    bar.outline.set_edgecolor(t["axis"])
    _title(ax, _s("corr_title"), _s("corr_sub"))
    fig.tight_layout()
    return fig


def plot_prior_posterior(validation: dict) -> Figure:
    t = _t()
    rows = validation["prior_posterior"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, r in enumerate(rows):
        for key, color, offset in (
            ("prior", t["muted"], 0.15),
            ("posterior", t["series"]["tv_S"], -0.15),
        ):
            b = r[key]
            label = _s(key) if i == 0 else None
            _interval(ax, b["p05"], b["p95"], i + offset, color, b["median"], label, size=45)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _num(v, 0) if v >= 1 else _num(v)))
    ax.set_yticks(range(len(rows)), [_label(r["channel"]) for r in rows])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(_s("pp_axis"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2)
    _title(ax, _s("pp_title"), pad=40)
    fig.tight_layout()
    return fig


def plot_recovery(validation: dict) -> Figure:
    t = _t()
    rows = validation["recovery"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, r in enumerate(rows):
        label = _s("estimated") if i == 0 else None
        _interval(ax, r["p05"], r["p95"], i, t["series"]["tv_S"], r["median"], label)
        ax.scatter(
            r["true"],
            i,
            s=70,
            marker="D",
            color=t["ink"],
            zorder=4,
            label=_s("true") if i == 0 else None,
        )
        note = "  " + _s("covered").format(k=r["covered"], n=r["n"])
        ax.text(max(r["p95"], r["true"]), i, note, va="center", color=t["ink2"], fontsize=9)
    ax.set_yticks(range(len(rows)), [_label(r["channel"]) for r in rows])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(_s("rec_axis"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2)
    _title(ax, _s("rec_title"), pad=40)
    fig.tight_layout()
    return fig


def banner_svg(results: dict, validation: dict | None, lang: str, mode: str) -> str:
    """Banner del README con el título y los números principales, como SVG."""
    t, s = THEMES[mode], TEXT[lang]
    sep = "," if lang == "es" else "."
    uplift = results["headline"]["uplift"]
    kpis = [
        f"+{results['headline']['media_uplift_pct']:.0f}%",
        f"{uplift['prob_positive'] * 100:.0f}%",
        f"{results['diagnostics']['holdout_mape'] * 100:.1f}%".replace(".", sep),
    ]
    if validation is not None:
        covered = sum(r["covered"] for r in validation["recovery"])
        total = sum(r["n"] for r in validation["recovery"])
        kpis.append(f"{covered}/{total}")
    font = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
    width, height = 1200, 400
    column = (width - 120) / len(kpis)
    blocks = []
    for i, (value, label) in enumerate(zip(kpis, s["banner_kpis"], strict=False)):
        x = 60 + i * column
        blocks.append(
            f'<rect x="{x:.0f}" y="214" width="40" height="4" rx="2" fill="{t["series"]["tv_S"]}"/>'
            f'<text x="{x:.0f}" y="282" font-size="56" font-weight="700" fill="{t["ink"]}">'
            f"{escape(value)}</text>"
            f'<text x="{x:.0f}" y="318" font-size="17" fill="{t["ink2"]}">'
            + "".join(
                f'<tspan x="{x:.0f}" dy="{0 if k == 0 else 22}">{escape(line)}</tspan>'
                for k, line in enumerate(textwrap.wrap(label, 24))
            )
            + "</text>"
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="{font}">'
        f'<rect width="{width}" height="{height}" rx="16" fill="{t["surface"]}" '
        f'stroke="{t["grid"]}"/>'
        f'<text x="60" y="78" font-size="15" letter-spacing="2" fill="{t["muted"]}">'
        f"{escape(s['banner_kicker'])}</text>"
        f'<text x="60" y="142" font-size="46" font-weight="700" fill="{t["ink"]}">'
        f"{escape(s['banner_title'])}</text>" + "".join(blocks) + "</svg>"
    )


IMAGE_NAMES = {
    "roi": plot_roi,
    "contributions": plot_contributions,
}
VALIDATION_IMAGES = {
    "correlation": plot_correlation,
    "prior_posterior": plot_prior_posterior,
    "recovery": plot_recovery,
}


def write_readme_images(
    results: dict,
    out_dir: Path,
    meridian: dict | None = None,
    validation: dict | None = None,
) -> None:
    """Escribe cada imagen en out_dir/<idioma>/<tema>/<nombre>.png, más el banner en SVG."""
    for lang in LANGS:
        for mode in MODES:
            folder = out_dir / lang / mode
            folder.mkdir(parents=True, exist_ok=True)
            with style(lang, mode):
                figures = {name: make(results) for name, make in IMAGE_NAMES.items()}
                if meridian is not None:
                    figures["meridian"] = plot_meridian(results, meridian)
                if validation is not None:
                    figures.update({n: make(validation) for n, make in VALIDATION_IMAGES.items()})
                for name, fig in figures.items():
                    fig.savefig(folder / f"{name}.png", dpi=160)
                    plt.close(fig)
            banner = banner_svg(results, validation, lang, mode)
            (folder / "banner.svg").write_text(banner, encoding="utf-8")
