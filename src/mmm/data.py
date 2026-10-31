"""Carga y control de los datos de demo de Robyn."""

from pathlib import Path

import holidays
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "dt_simulated_weekly.csv"

CHANNELS: list[str] = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]

TARGET = "revenue"
HOLDOUT_WEEKS = 12
SEED = 42

CHANNEL_LABELS: dict[str, str] = {
    "tv_S": "TV",
    "ooh_S": "Vía pública",
    "print_S": "Prensa",
    "facebook_S": "Facebook",
    "search_S": "Búsqueda",
}

REQUIRED_COLUMNS = ["DATE", "revenue", *CHANNELS, "newsletter", "competitor_sales_B", "events"]
NON_NEGATIVE = ["revenue", *CHANNELS]


class DataError(ValueError):
    """Los datos no cumplen una condición necesaria para el modelo."""


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["DATE"])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Faltan columnas: {', '.join(missing)}")

    dates = df["DATE"].sort_values()
    duplicated = dates[dates.duplicated()]
    if not duplicated.empty:
        raise DataError(f"Semana repetida: {duplicated.iloc[0]:%Y-%m-%d}")

    expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="7D")
    absent = expected.difference(dates)
    if not absent.empty:
        raise DataError(f"Falta la semana: {absent[0]:%Y-%m-%d}")

    for column in NON_NEGATIVE:
        negative = df[df[column] < 0]
        if not negative.empty:
            raise DataError(
                f"Valor negativo en {column}, semana {negative['DATE'].iloc[0]:%Y-%m-%d}"
            )

    return df.sort_values("DATE").reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()

    event_columns = []
    for value in sorted(v for v in out["events"].unique() if v != "na"):
        column = f"event_{value}"
        out[column] = (out["events"] == value).astype(int)
        event_columns.append(column)

    years = range(out["DATE"].dt.year.min(), out["DATE"].dt.year.max() + 2)
    german_holidays = {pd.Timestamp(d) for d in holidays.Germany(years=years)}
    out["holiday"] = [
        int(any(start + pd.Timedelta(days=i) in german_holidays for i in range(7)))
        for start in out["DATE"]
    ]

    # PyMC-Marketing escala ventas y canales, pero no los controles.
    for column in ("newsletter", "competitor_sales_B"):
        out[column] = out[column] / out[column].max()

    out["trend"] = range(len(out))
    out["trend"] = out["trend"] / (len(out) - 1)

    controls = ["newsletter", "competitor_sales_B", "holiday", "trend", *event_columns]
    return out, controls
