import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from mmm import plots  # noqa: E402
from mmm.data import CHANNELS, load_raw  # noqa: E402


def test_palette_matches_site_css():
    css = (plots.ROOT / "site" / "css" / "styles.css").read_text(encoding="utf-8")
    light = css.split("@media")[0]
    for channel, color in plots.PALETTE.items():
        assert f"--{channel}: {color};" in light


@pytest.mark.parametrize(
    "name", ["plot_roi", "plot_contributions", "plot_holdout", "plot_saturation"]
)
def test_results_plots_return_figures(fake_results, name):
    fig = getattr(plots, name)(fake_results)
    assert isinstance(fig, Figure)


def test_data_plots_return_figures():
    df = load_raw()
    assert isinstance(plots.plot_revenue(df), Figure)
    assert isinstance(plots.plot_spend(df), Figure)
    assert set(CHANNELS) <= set(plots.PALETTE)


def fake_meridian(results):
    roi = [{**r, "median": r["median"] * 1.2} for r in results["roi"]]
    return {"meta": {}, "roi": roi, "diagnostics": {"max_rhat": 1.0, "holdout_mape": 0.03}}


def test_plot_meridian_returns_figure(fake_results):
    assert isinstance(plots.plot_meridian(fake_results, fake_meridian(fake_results)), Figure)


def fake_validation(results):
    cols = ["revenue", *CHANNELS, "newsletter", "competitor_sales_B"]
    n = len(cols)
    matrix = [[1.0 if i == j else 0.1 for j in range(n)] for i in range(n)]
    pp = [
        {
            "channel": c,
            "prior": {"median": 30, "p05": 2, "p95": 900},
            "posterior": {"median": 5, "p05": 3, "p95": 7},
        }
        for c in CHANNELS
    ]
    rec = [
        {"channel": c, "true": 5.0, "median": 5.5, "p05": 3.0, "p95": 8.0, "covered": 9, "n": 10}
        for c in CHANNELS
    ]
    return {
        "correlation": {"columns": cols, "matrix": matrix, "zero_weeks": {}, "vif": {}},
        "prior_posterior": pp,
        "recovery": rec,
    }


@pytest.mark.parametrize("name", ["plot_correlation", "plot_prior_posterior", "plot_recovery"])
def test_validation_plots_return_figures(fake_results, name):
    assert isinstance(getattr(plots, name)(fake_validation(fake_results)), Figure)


def fake_headline(results):
    results["headline"]["uplift"] = {"median": 34.0, "p05": 3.0, "p95": 78.0, "prob_positive": 0.96}
    return results


def test_palette_dark_matches_site_css():
    css = (plots.ROOT / "site" / "css" / "styles.css").read_text(encoding="utf-8")
    dark = css.split(':root[data-theme="dark"]')[1]
    for channel, color in plots.THEMES["dark"]["series"].items():
        assert f"--{channel}: {color};" in dark


def test_readme_images_for_every_language_and_theme(fake_results, tmp_path):
    results = fake_headline(fake_results)
    plots.write_readme_images(
        results, tmp_path, meridian=fake_meridian(results), validation=fake_validation(results)
    )
    names = ["roi", "contributions", "meridian", "correlation", "prior_posterior", "recovery"]
    for lang in ("es", "en"):
        for mode in ("light", "dark"):
            for name in names:
                assert (tmp_path / lang / mode / f"{name}.png").stat().st_size > 0
            assert (tmp_path / lang / mode / "banner.svg").exists()


@pytest.mark.parametrize(
    "lang,title", [("es", "¿Dónde rinde"), ("en", "Where does each ad dollar")]
)
def test_banner_is_valid_svg_with_headline(fake_results, lang, title):
    import xml.etree.ElementTree as ET

    results = fake_headline(fake_results)
    results["headline"]["media_uplift_pct"] = 35.0
    svg = plots.banner_svg(results, fake_validation(results), lang, "dark")
    ET.fromstring(svg)
    assert "+35%" in svg and title in svg and "45/50" in svg


def test_english_labels_and_number_format(fake_results):
    with plots.style("en", "light"):
        fig = plots.plot_roi(fake_results)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert "Out-of-home" in labels
        assert plots._money(1_234_567) == "$1.2M"
    assert plots._money(1_234_567) == "$ 1,2 M"
