"""Utilita' di formattazione e piccoli helper sui dati.

Nessuna dipendenza da matplotlib: la dashboard usa Plotly.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Formattazione
# ----------------------------------------------------------------------------
def fmt_currency(value: Any, decimali: int = 0, valuta: str = "$") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n.d."
    return f"{valuta}{value:,.{decimali}f}"


def fmt_currency_compact(value: Any, valuta: str = "$") -> str:
    """Formato compatto per i KPI: $1,2M / $340k / $980."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n.d."
    segno = "-" if value < 0 else ""
    a = abs(float(value))
    if a >= 1e9:
        return f"{segno}{valuta}{a / 1e9:,.2f}G"
    if a >= 1e6:
        return f"{segno}{valuta}{a / 1e6:,.2f}M"
    if a >= 10_000:
        return f"{segno}{valuta}{a / 1e3:,.0f}k"
    return f"{segno}{valuta}{a:,.0f}"


def fmt_pct(value: Any, decimali: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n.d."
    return f"{value * 100:,.{decimali}f}%"


def fmt_num(value: Any, decimali: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n.d."
    return f"{value:,.{decimali}f}"


def fmt_int(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n.d."
    return f"{int(value):,d}"


# ----------------------------------------------------------------------------
# Serie
# ----------------------------------------------------------------------------
def has_data(series: Optional[pd.Series]) -> bool:
    return series is not None and not series.empty and series.notna().any()


def drawdown_monetario(equity: pd.Series) -> pd.Series:
    if not has_data(equity):
        return pd.Series(dtype=float)
    return (equity - equity.cummax()).clip(upper=0.0)


def drawdown_percentuale(series: pd.Series) -> pd.Series:
    if not has_data(series):
        return pd.Series(dtype=float)
    s = series.astype(float)
    cm = s.cummax().replace(0, np.nan)
    return ((s / cm) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=0.0)


def json_safe(obj: Any) -> Any:
    """Rende un oggetto serializzabile in JSON (NaN/inf -> None, Timestamp -> ISO)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(obj)
        return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.ndarray,)):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {json_safe(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return dataframe_to_records(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    if pd.isna(obj):
        return None
    return str(obj)


def dataframe_to_records(df: pd.DataFrame, index_name: str = "data") -> list:
    """DataFrame -> lista di dizionari JSON-safe, con l'indice come prima chiave."""
    if df is None or df.empty:
        return []
    out = df.reset_index()
    if out.columns[0] in ("index", None) or str(out.columns[0]).startswith("level_"):
        out = out.rename(columns={out.columns[0]: index_name})
    return [json_safe(rec) for rec in out.to_dict(orient="records")]


def safe_div(a: float, b: float, default: Optional[float] = None) -> Optional[float]:
    try:
        if b in (0, None) or not np.isfinite(b):
            return default
        v = a / b
        return v if np.isfinite(v) else default
    except (TypeError, ZeroDivisionError):
        return default
