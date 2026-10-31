"""Comandos `mmm fit` y `mmm export`."""

import argparse
import json
import sys
from pathlib import Path

import xarray as xr
from pymc_marketing.mmm import MMM

from mmm import plots
from mmm.data import CHANNELS, load_raw, prepare_features
from mmm.diagnostics import Diagnostics, ModelQualityError, assert_quality, compute_diagnostics
from mmm.export import HORIZON_WEEKS, build_results, write_results, write_simulator_cases
from mmm.model import TARGET, build_mmm, fit_mmm, predict_holdout, split_train_test
from mmm.optimize import optimal_plan

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.nc"
HOLDOUT_PATH = ARTIFACTS / "holdout.nc"
DIAGNOSTICS_PATH = ARTIFACTS / "diagnostics.json"
RESULTS_PATH = ROOT / "site" / "data" / "resultados.json"
CASES_PATH = ROOT / "tests" / "fixtures" / "simulator_cases.json"
IMAGES_DIR = ROOT / "docs" / "img"


def fit(args: argparse.Namespace) -> int:
    df, controls = prepare_features(load_raw())
    train, test = split_train_test(df)
    mmm = build_mmm(controls)
    idata = fit_mmm(mmm, train, draws=args.draws, tune=args.tune, target_accept=args.target_accept)
    holdout = predict_holdout(mmm, test)
    free_vars = [rv.name for rv in mmm.model.free_RVs]
    diagnostics = compute_diagnostics(
        idata, test[TARGET].to_numpy(), holdout.median("sample").to_numpy(), free_vars
    )
    print(json.dumps(diagnostics.to_dict(), indent=2))
    try:
        assert_quality(diagnostics)
    except ModelQualityError as error:
        print(f"El modelo no pasa los controles de calidad: {error}", file=sys.stderr)
        return 1

    ARTIFACTS.mkdir(exist_ok=True)
    mmm.save(str(MODEL_PATH))
    holdout.reset_index("sample", drop=True).to_netcdf(HOLDOUT_PATH)
    DIAGNOSTICS_PATH.write_text(json.dumps(diagnostics.to_dict()), encoding="utf-8")
    print(f"Modelo guardado en {MODEL_PATH}")
    return 0


def export(_: argparse.Namespace) -> int:
    if not MODEL_PATH.exists():
        print("Falta artifacts/model.nc. Ejecutá primero: uv run mmm fit", file=sys.stderr)
        return 1
    df, _ = prepare_features(load_raw())
    train, _ = split_train_test(df)
    mmm = MMM.load(str(MODEL_PATH))
    holdout = xr.load_dataarray(HOLDOUT_PATH)
    diagnostics = Diagnostics(**json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8")))

    current = {c: float(df[c].tail(HORIZON_WEEKS).mean()) for c in CHANNELS}
    spend_max = {c: float(df[c].max()) for c in CHANNELS}
    optimal = optimal_plan(mmm, current, spend_max, df["DATE"].max())

    results = build_results(mmm, mmm.idata, df, train, holdout, diagnostics, optimal)
    write_results(results, RESULTS_PATH)
    write_simulator_cases(results, CASES_PATH)
    plots.write_readme_images(results, IMAGES_DIR)
    print(f"Resultados en {RESULTS_PATH}")
    print(f"Mejora con el reparto óptimo: {results['headline']['media_uplift_pct']:.1f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mmm", description="Marketing Mix Model")
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit", help="Entrena el modelo y controla su calidad")
    fit_parser.add_argument("--draws", type=int, default=2000)
    fit_parser.add_argument("--tune", type=int, default=2000)
    fit_parser.add_argument("--target-accept", type=float, default=0.95)
    fit_parser.set_defaults(func=fit)
    sub.add_parser("export", help="Exporta resultados para la página").set_defaults(func=export)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
