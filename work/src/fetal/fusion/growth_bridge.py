# FIELD: Fusion
# MODEL: Hadlock 4-Parameter EFW + CDC Live-Birth Percentile Bridge
# STEP: growth_bridge
# USED BY: work/tests/test_growth_bridge.py, work/docs/FUSION_STATUS.md

"""Growth-bridge fusion module mapping ultrasound biometrics (HC, AC, FL, optional BPD)
to Hadlock 4-parameter Estimated Fetal Weight (g) and CDC live-birth gestational
percentile bands (below_p10, p10_to_p90, above_p90) for GA weeks 20..42.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from fetal.explain.hadlock import calculate_efw

DEFAULT_PERCENTILES_CSV = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "results"
    / "fusion"
    / "gestational_percentiles.csv"
)


@lru_cache(maxsize=4)
def load_cdc_percentiles(csv_path: str | Path | None = None) -> dict[int, dict[str, float]]:
    """Load CDC live-birth gestational percentiles keyed by integer GA week (20..42)."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_PERCENTILES_CSV
    if not path.is_file():
        raise FileNotFoundError(f"CDC gestational percentiles CSV not found: {path}")

    table: dict[int, dict[str, float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            week = int(row["ga_weeks"])
            table[week] = {
                "n": float(row["n"]),
                "p10": float(row["p10"]),
                "p50": float(row["p50"]),
                "p90": float(row["p90"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }
    return table


def classify_percentile_band(efw_g: float, p10: float, p90: float) -> str:
    """Classify EFW (g) against CDC live-birth p10 and p90 thresholds."""
    if efw_g < p10:
        return "below_p10"
    if efw_g > p90:
        return "above_p90"
    return "p10_to_p90"


def assess_fetal_growth(
    hc_mm: float,
    ac_mm: float,
    fl_mm: float,
    ga_weeks: int,
    bpd_mm: float | None = None,
    percentiles_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute Hadlock 4-parameter EFW (g) and CDC live-birth percentile band.

    Parameters
    ----------
    hc_mm, ac_mm, fl_mm : float
        Head circumference, abdominal circumference, and femur length in mm.
    ga_weeks : int
        Completed gestational age in weeks (valid CDC range: 20 to 42 inclusive).
    bpd_mm : float | None
        Biparietal diameter in mm. If omitted, estimated via standard HC/3.5 ratio
        (matching the 280 mm HC -> 80 mm BPD reference vector) for the 4-parameter formula.
    percentiles_path : str | Path | None
        Optional override path to gestational_percentiles.csv.

    Returns
    -------
    dict
        efw_g, bpd_mm_used, ga_weeks, band ('below_p10' | 'p10_to_p90' | 'above_p90'),
        sga (bool), z_score, p10, p50, p90.
    """
    week = int(round(ga_weeks))
    if abs(ga_weeks - week) > 1e-6 or week < 20 or week > 42:
        raise ValueError(f"ga_weeks must be an integer week in [20, 42], got {ga_weeks}")

    bpd_used = float(bpd_mm) if bpd_mm is not None else float(hc_mm) / 3.5
    efw_g = float(calculate_efw(hc_mm=hc_mm, bpd_mm=bpd_used, ac_mm=ac_mm, fl_mm=fl_mm))

    table = load_cdc_percentiles(str(percentiles_path) if percentiles_path is not None else None)
    if week not in table:
        raise KeyError(f"Gestational week {week} missing from CDC percentiles table")
    ref = table[week]

    band = classify_percentile_band(efw_g, ref["p10"], ref["p90"])
    z_score = (efw_g - ref["mean"]) / ref["std"] if ref["std"] > 0 else 0.0

    return {
        "efw_g": efw_g,
        "bpd_mm_used": bpd_used,
        "ga_weeks": week,
        "band": band,
        "sga": band == "below_p10",
        "z_score": float(z_score),
        "p10": ref["p10"],
        "p50": ref["p50"],
        "p90": ref["p90"],
    }
