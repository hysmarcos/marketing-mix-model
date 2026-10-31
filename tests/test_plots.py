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


def test_write_readme_images(fake_results, tmp_path):
    plots.write_readme_images(fake_results, tmp_path)
    assert (tmp_path / "roi.png").stat().st_size > 0
    assert (tmp_path / "aportes.png").stat().st_size > 0


def fake_meridian(results):
    roi = [{**r, "median": r["median"] * 1.2} for r in results["roi"]]
    return {"meta": {}, "roi": roi, "diagnostics": {"max_rhat": 1.0, "holdout_mape": 0.03}}


def test_plot_meridian_returns_figure(fake_results):
    assert isinstance(plots.plot_meridian(fake_results, fake_meridian(fake_results)), Figure)


def test_readme_images_include_meridian_when_given(fake_results, tmp_path):
    plots.write_readme_images(fake_results, tmp_path, meridian=fake_meridian(fake_results))
    assert (tmp_path / "meridian.png").stat().st_size > 0
