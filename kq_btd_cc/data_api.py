# kq_btd_cc/data_api.py
from __future__ import annotations

import os
from typing import Optional
import numpy as np
import pandas as pd
import requests


def _get_api_key() -> Optional[str]:
    """
    Ordine di ricerca:
    1) st.secrets["EODHD_API_KEY"]
    2) os.environ["EODHD_API_KEY"]
    3) (solo per sviluppo locale) .streamlit/secrets.toml o secrets.toml
    """
    # 1) Streamlit secrets (top-level o nidificato)
    try:
        import streamlit as st
        if "EODHD_API_KEY" in st.secrets:
            return st.secrets["EODHD_API_KEY"]
        # supporto opzionale per formati nidificati:
        for sec in ("eodhd", "EODHD", "EOD"):
            if sec in st.secrets and isinstance(st.secrets[sec], dict):
                d = st.secrets[sec]
                if "api_key" in d:
                    return d["api_key"]
                if "API_KEY" in d:
                    return d["API_KEY"]
    except Exception:
        pass

    # 2) Variabile d'ambiente
    key = os.getenv("EODHD_API_KEY")
    if key:
        return key

    # 3) File TOML locale (non usato su Streamlit Cloud, utile in locale)
    for path in (".streamlit/secrets.toml", "secrets.toml"):
        try:
            import toml  # facoltativo; se non installato si passa oltre
            data = toml.load(path)
            if "EODHD_API_KEY" in data:
                return data["EODHD_API_KEY"]
            for sec in ("eodhd", "EODHD", "EOD"):
                if sec in data and isinstance(data[sec], dict):
                    d = data[sec]
                    if "api_key" in d:
                        return d["api_key"]
                    if "API_KEY" in d:
                        return d["API_KEY"]
        except Exception:
            pass

    return None


def fetch_eodhd_ohlc(ticker: str, start_date: str, end_date: str, period: str = "m") -> pd.DataFrame:
    """
    Scarica OHLCV da EODHD e ricostruisce Open/High/Low aggiustati usando adjusted_close/close.
    period: 'm' mensile, 'w' settimanale.
    Ritorna DataFrame con colonne: Open, High, Low, Close, Volume indicizzate per data.
    """
    api_key = _get_api_key()
    if not api_key:
        # Proviamo a dare un messaggio chiaro dentro Streamlit, se disponibile
        try:
            import streamlit as st
            st.error("API key EODHD mancante. Imposta `EODHD_API_KEY` nei Secrets dell'app Streamlit.")
            st.info('Esempio secrets:  EODHD_API_KEY = "la-tua-api-key"')
            st.stop()
        except Exception:
            raise RuntimeError("API key EODHD mancante. Imposta EODHD_API_KEY nei secrets o come variabile d'ambiente.")

    url = f"https://eodhd.com/api/eod/{ticker}"
    params = {
        "api_token": api_key,
        "from": start_date,
        "to": end_date,
        "period": period,
        "fmt": "json",
        "order": "a",
    }

    try:
        r = requests.get(url, params=params, timeout=45)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        # Se fallisce il download, ritorna DF vuoto
        return pd.DataFrame()

    if not js or not isinstance(js, list):
        return pd.DataFrame()

    df = pd.DataFrame(js)
    if df.empty:
        return pd.DataFrame()

    # parsing e rinomina
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    rename = {
        "open": "Open_Nominal",
        "high": "High_Nominal",
        "low": "Low_Nominal",
        "close": "Close_Nominal",
        "adjusted_close": "Close",
        "volume": "Volume",
    }
    df.rename(columns=rename, inplace=True)

    # numerici
    for col in ["Open_Nominal", "High_Nominal", "Low_Nominal", "Close_Nominal", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fallback Close se adjusted_close mancante
    if "Close" not in df.columns or not df["Close"].notna().any():
        if "Close_Nominal" in df.columns and df["Close_Nominal"].notna().any():
            df["Close"] = df["Close_Nominal"]
        else:
            return pd.DataFrame()

    # ricostruzione OHLC aggiustati
    if "Close_Nominal" in df.columns and df["Close_Nominal"].replace(0, np.nan).notna().any():
        adj_factor = (df["Close"] / df["Close_Nominal"].replace(0, np.nan)).ffill().bfill()
        df["Open"] = (df.get("Open_Nominal") * adj_factor).round(4)
        if "High_Nominal" in df.columns:
            df["High"] = (df["High_Nominal"] * adj_factor).round(4)
        if "Low_Nominal" in df.columns:
            df["Low"] = (df["Low_Nominal"] * adj_factor).round(4)
    else:
        # se non possibile, usa nominali dove presenti
        if "Open_Nominal" in df.columns:
            df["Open"] = df["Open_Nominal"]
        if "High_Nominal" in df.columns:
            df["High"] = df["High_Nominal"]
        if "Low_Nominal" in df.columns:
            df["Low"] = df["Low_Nominal"]

    # colonne finali
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan

    out = df[["Open", "High", "Low", "Close", "Volume"]].sort_index().ffill().bfill()
    # rimuovi righe completamente vuote eventualmente rimaste
    if out[["Open", "Close"]].isna().any().any():
        return pd.DataFrame()

    return out
