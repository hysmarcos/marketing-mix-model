from pathlib import Path

import pandas as pd
import pytest

from mmm.calibration import LIFT_COLUMNS, TEMPLATE_PATH, CalibrationError, load_lift_tests

# Valores de prueba para los tests. No son resultados de experimentos reales.
GOOD_ROW = {
    "channel": "facebook_S",
    "x": 3000.0,
    "delta_x": 2000.0,
    "delta_y": 9000.0,
    "sigma": 2500.0,
}


def write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "lift.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_template_has_only_the_header():
    df = pd.read_csv(TEMPLATE_PATH)
    assert list(df.columns) == LIFT_COLUMNS
    assert df.empty


def test_load_valid_file(tmp_path):
    df = load_lift_tests(write(tmp_path, [GOOD_ROW]))
    assert list(df.columns) == LIFT_COLUMNS
    assert len(df) == 1


def test_empty_file_raises(tmp_path):
    with pytest.raises(CalibrationError, match="no tiene experimentos"):
        load_lift_tests(TEMPLATE_PATH)


def test_missing_column_raises(tmp_path):
    row = {k: v for k, v in GOOD_ROW.items() if k != "sigma"}
    with pytest.raises(CalibrationError, match="sigma"):
        load_lift_tests(write(tmp_path, [row]))


@pytest.mark.parametrize(
    "change,message",
    [
        ({"channel": "radio_S"}, "radio_S"),
        ({"x": -1.0}, "x"),
        ({"delta_x": 0.0}, "delta_x"),
        ({"sigma": 0.0}, "sigma"),
        ({"delta_y": float("nan")}, "delta_y"),
    ],
)
def test_bad_values_raise_with_row_and_column(tmp_path, change, message):
    with pytest.raises(CalibrationError, match=rf"fila 2.*{message}|{message}.*fila 2"):
        load_lift_tests(write(tmp_path, [GOOD_ROW, {**GOOD_ROW, **change}]))


@pytest.mark.slow
def test_fit_with_lift_tests_adds_the_measurements(prepared_data, tmp_path):
    from mmm.model import build_mmm, fit_mmm

    _, controls, train, _ = prepared_data
    mmm = build_mmm(controls)
    lift = load_lift_tests(write(tmp_path, [GOOD_ROW]))
    idata = fit_mmm(mmm, train, chains=1, draws=50, tune=50, lift_tests=lift)
    assert "lift_measurements" in mmm.model.named_vars
    assert "lift_measurements" in idata["observed_data"]


def test_cli_rejects_bad_lift_tests_before_fitting(capsys):
    from mmm.cli import main

    assert main(["fit", "--lift-tests", str(TEMPLATE_PATH)]) == 1
    assert "no tiene experimentos" in capsys.readouterr().err
