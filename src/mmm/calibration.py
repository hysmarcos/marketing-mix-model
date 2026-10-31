"""Calibración del modelo con experimentos de incrementalidad (lift tests).

Un lift test mide cuántas ventas extra (delta_y) generó un cambio en la inversión
semanal de un canal (de x a x + delta_x). Todos los valores van en pesos por semana.
PyMC-Marketing los pasa a su escala interna.
"""

from pathlib import Path

import pandas as pd

from mmm.data import CHANNELS

LIFT_COLUMNS = ["channel", "x", "delta_x", "delta_y", "sigma"]
TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "data" / "lift_tests_plantilla.csv"


class CalibrationError(ValueError):
    """El archivo de experimentos no cumple el formato."""


def load_lift_tests(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in LIFT_COLUMNS if c not in df.columns]
    if missing:
        raise CalibrationError(f"Faltan columnas en {path}: {', '.join(missing)}")
    if df.empty:
        raise CalibrationError(f"{path} no tiene experimentos. Agregá una fila por experimento.")

    for i, row in df.iterrows():
        line = i + 1  # fila del CSV sin contar el encabezado
        for column in LIFT_COLUMNS:
            if pd.isna(row[column]):
                raise CalibrationError(f"fila {line}: falta el valor de {column}")
        if row["channel"] not in CHANNELS:
            raise CalibrationError(f"fila {line}: el canal {row['channel']} no está en el modelo")
        if row["x"] < 0:
            raise CalibrationError(f"fila {line}: x no puede ser negativo")
        if row["delta_x"] == 0:
            raise CalibrationError(f"fila {line}: delta_x no puede ser 0")
        if row["sigma"] <= 0:
            raise CalibrationError(f"fila {line}: sigma debe ser mayor que 0")

    return df[LIFT_COLUMNS].reset_index(drop=True)
