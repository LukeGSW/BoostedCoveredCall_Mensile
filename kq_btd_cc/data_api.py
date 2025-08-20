"""Download e preparazione dati da EODHD.
- Ricostruzione OHLC aggiustati usando `adjusted_close` come baseline.
- Compatibile con azioni e crypto (es. BTC-USD.CC).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import os
import requests
import pandas as pd
import numpy as np

try:
    import streamlit as st  # per secrets in ambiente Streamlit
except Exception:  # pragma: no cover
    st = None


EODHD_BASE_URL = "https://eodhd.com/api/eod/"


def _get_api_key(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    # Streamlit secrets
    if st is not None:
        try:
            if "EODHD_API_KEY" in st.secrets:
                return st.secrets["EODHD_API_KEY"]
        except Exception:
            pass
    # env var
    return os.getenv("EODHD_API_KEY")


def fetch_eodhd_ohlc(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    period: str = "m",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Scarica OHLCV da EODHD e ricostruisce gli OHLC **aggiustati**.

    Args:
        ticker: es. 'AAPL.US' o 'BTC-USD.CC'
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD' (default: ieri)
        period: 'd','w','m' (qui usiamo 'w' e 'm')
        api_key: override esplicito della chiave
    Returns:
        DataFrame indicizzato per data con colonne: Open, High, Low, Close, Volume
    """
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    key = _get_api_key(api_key)
    if not key:
        raise RuntimeError("EODHD_API_KEY mancante: inseriscila nei secrets o nelle env.")

    url = f"{EODHD_BASE_URL}{ticker}"
    params = {
        "api_token": key,
        "from": start_date,
        "to": end_date,
        "period": period,
        "fmt": "json",
        "order": "a",
    }

    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]).astype({"Volume": "Int64"})

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])  # eodhd key name
    df.set_index("date", inplace=True)

    # Rinominazioni
    rename = {
        "open": "Open_Nominal_EODHD",
        "high": "High_Nominal_EODHD",
        "low": "Low_Nominal_EODHD",
        "adjusted_close": "Close",
        "close": "Close_Nominal_EODHD",
        "volume": "Volume",
    }
    df = df.rename(columns=rename)

    # Converti numerici
    for col in [
        "Open_Nominal_EODHD",
        "High_Nominal_EODHD",
        "Low_Nominal_EODHD",
        "Close",
        "Close_Nominal_EODHD",
        "Volume",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].astype("Int64")

    # Se manca adjusted_close, fallback al close nominale
    if "Close" not in df.columns or not df["Close"].notna().any():
        if "Close_Nominal_EODHD" in df.columns and df["Close_Nominal_EODHD"].notna().any():
            df["Close"] = df["Close_Nominal_EODHD"]
        else:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]).astype({"Volume": "Int64"})

    # Calcolo fattore di aggiustamento
    if (
        "Close_Nominal_EODHD" in df.columns
        and df["Close_Nominal_EODHD"].replace(0, np.nan).notna().any()
    ):
        factor = (df["Close"] / df["Close_Nominal_EODHD"].replace(0, np.nan)).ffill().bfill()
        if "Open_Nominal_EODHD" in df.columns:
            df["Open"] = df["Open_Nominal_EODHD"] * factor
        if "High_Nominal_EODHD" in df.columns:
            df["High"] = df["High_Nominal_EODHD"] * factor
        if "Low_Nominal_EODHD" in df.columns:
            df["Low"] = df["Low_Nominal_EODHD"] * factor
    else:
        # fallback: usa valori nominali
        if "Open_Nominal_EODHD" in df.columns:
            df["Open"] = df["Open_Nominal_EODHD"]
        if "High_Nominal_EODHD" in df.columns:
            df["High"] = df["High_Nominal_EODHD"]
        if "Low_Nominal_EODHD" in df.columns:
            df["Low"] = df["Low_Nominal_EODHD"]

    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].round(6)

    out = df[["Open", "High", "Low", "Close", "Volume"]].sort_index().ffill().bfill()
    return out
