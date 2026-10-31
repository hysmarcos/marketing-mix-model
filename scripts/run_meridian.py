# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["google-meridian==2.1.0", "holidays", "pandas"]
# ///
"""Entrena Meridian y guarda site/data/meridian.json.

Uso: uv run --script scripts/run_meridian.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmm.meridian_compare import MERIDIAN_PATH, run  # noqa: E402

if __name__ == "__main__":
    results = run()
    print(f"Resultados de Meridian en {MERIDIAN_PATH}")
    print(results["diagnostics"])
