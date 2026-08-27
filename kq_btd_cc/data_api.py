"""Accesso ai dati EODHD, con cache e ricostruzione degli OHLC aggiustati.

Le chiamate sono memorizzate in cache da Streamlit: cambiare un parametro che
non tocca i dati (premio, boost, tema) non riscarica nulla.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://eodhd.com/api/eod/{ticker}"
TIMEOUT = 45
COLONNE = ["Open", "High", "Low", "Close", "Volume"]

# Giorni di storico extra scaricati prima della data di inizio, per dare allo
# stimatore di volatilita' una finestra di riscaldamento gia' piena al mese 1.
# Dimensionato sulla finestra lunga di default (504 giorni di borsa ~ 2 anni).
WARMUP_GIORNI = 900


class DatiNonDisponibili(RuntimeError):
    """Il download e' andato a buon fine ma non ci sono dati utilizzabili."""


class ChiaveMancante(RuntimeError):
    """Nessuna API key EODHD configurata."""


# ----------------------------------------------------------------------------
# API key
# ----------------------------------------------------------------------------
def get_api_key() -> Optional[str]:
    """Cerca la chiave in: secrets Streamlit, variabile d'ambiente, file TOML locale."""
    try:
        import streamlit as st
        if "EODHD_API_KEY" in st.secrets:
            return str(st.secrets["EODHD_API_KEY"])
        for sez in ("eodhd", "EODHD", "EOD"):
            if sez in st.secrets:
                d = st.secrets[sez]
                for k in ("api_key", "API_KEY", "key"):
                    if k in d:
                        return str(d[k])
    except Exception:
        pass

    key = os.getenv("EODHD_API_KEY")
    if key:
        return key

    for path in (".streamlit/secrets.toml", "secrets.toml"):
        try:
            import toml
            data = toml.load(path)
            if "EODHD_API_KEY" in data:
                return str(data["EODHD_API_KEY"])
            for sez in ("eodhd", "EODHD", "EOD"):
                if isinstance(data.get(sez), dict):
                    for k in ("api_key", "API_KEY", "key"):
                        if k in data[sez]:
                            return str(data[sez][k])
        except Exception:
            continue
    return None


def ha_api_key() -> bool:
    return bool(get_api_key())


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
def _normalizza(js: list) -> pd.DataFrame:
    """Da JSON EODHD a DataFrame con OHLC aggiustati per split e dividendi."""
    df = pd.DataFrame(js)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={
        "open": "Open_Nominal", "high": "High_Nominal", "low": "Low_Nominal",
        "close": "Close_Nominal", "adjusted_close": "Close", "volume": "Volume",
    })
    for c in ("Open_Nominal", "High_Nominal", "Low_Nominal", "Close_Nominal", "Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Close" not in df.columns or not df["Close"].notna().any():
        if "Close_Nominal" in df.columns and df["Close_Nominal"].notna().any():
            df["Close"] = df["Close_Nominal"]
        else:
            return pd.DataFrame()

    # Il fattore di aggiustamento riporta anche Open/High/Low sulla stessa scala
    # di adjusted_close, altrimenti il confronto open/close di un mese e' falsato.
    if "Close_Nominal" in df.columns and df["Close_Nominal"].replace(0, np.nan).notna().any():
        fattore = (df["Close"] / df["Close_Nominal"].replace(0, np.nan)).ffill().bfill()
        for src, dst in (("Open_Nominal", "Open"), ("High_Nominal", "High"), ("Low_Nominal", "Low")):
            if src in df.columns:
                df[dst] = df[src] * fattore
    else:
        for src, dst in (("Open_Nominal", "Open"), ("High_Nominal", "High"), ("Low_Nominal", "Low")):
            if src in df.columns:
                df[dst] = df[src]

    for c in COLONNE:
        if c not in df.columns:
            df[c] = np.nan

    out = df[COLONNE].copy()
    # High/Low mancanti (tipico su alcuni indici): si ripiega su Open/Close
    out["High"] = out["High"].fillna(out[["Open", "Close"]].max(axis=1))
    out["Low"] = out["Low"].fillna(out[["Open", "Close"]].min(axis=1))
    out["Open"] = out["Open"].fillna(out["Close"])
    out["Volume"] = out["Volume"].fillna(0.0)
    out = out.dropna(subset=["Open", "Close"])
    return out[out["Close"] > 0]


def _scarica(ticker: str, start: str, end: str, period: str, api_key: str) -> pd.DataFrame:
    r = requests.get(
        BASE_URL.format(ticker=ticker),
        params={"api_token": api_key, "from": start, "to": end,
                "period": period, "fmt": "json", "order": "a"},
        timeout=TIMEOUT,
    )
    if r.status_code in (401, 403):
        raise ChiaveMancante("API key EODHD rifiutata dal server (401/403).")
    if r.status_code == 404:
        raise DatiNonDisponibili(f"Ticker '{ticker}' non trovato su EODHD.")
    r.raise_for_status()
    js = r.json()
    if not isinstance(js, list) or not js:
        raise DatiNonDisponibili(f"Nessun dato restituito per '{ticker}' (periodo '{period}').")
    return _normalizza(js)


def _cache_wrapper():
    """Decoratore di cache Streamlit, se disponibile; altrimenti nessuna cache."""
    try:
        import streamlit as st
        return st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
    except Exception:
        def passthrough(fn):
            return fn
        return passthrough


@_cache_wrapper()
def _fetch_cached(ticker: str, start: str, end: str, period: str, api_key: str) -> pd.DataFrame:
    return _scarica(ticker, start, end, period, api_key)


def fetch_eodhd_ohlc(ticker: str, start_date: str, end_date: str,
                     period: str = "m") -> pd.DataFrame:
    """OHLCV aggiustati da EODHD. period: 'd' giornaliero, 'w' settimanale, 'm' mensile."""
    api_key = get_api_key()
    if not api_key:
        raise ChiaveMancante(
            "API key EODHD mancante: impostala nei Secrets di Streamlit "
            "(EODHD_API_KEY) oppure come variabile d'ambiente."
        )
    return _fetch_cached(ticker, start_date, end_date, period, api_key)


def carica_serie(ticker: str, start_date: str, end_date: Optional[str] = None,
                 con_giornalieri: bool = True) -> Dict[str, object]:
    """Scarica in un colpo solo mensile, settimanale e giornaliero.

    I dati giornalieri partono prima della data di inizio, cosi' la volatilita'
    e' gia' stimabile al primo mese del backtest invece che dopo un trimestre.
    """
    fine = end_date or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    inizio = start_date
    warmup = (datetime.strptime(start_date, "%Y-%m-%d").date()
              - timedelta(days=WARMUP_GIORNI)).strftime("%Y-%m-%d")

    avvisi = []
    mensile = fetch_eodhd_ohlc(ticker, inizio, fine, "m")
    if mensile.empty or len(mensile) < 2:
        raise DatiNonDisponibili(
            f"Servono almeno due mesi di storico per '{ticker}' nel periodo richiesto."
        )

    try:
        settimanale = fetch_eodhd_ohlc(ticker, warmup, fine, "w")
    except (DatiNonDisponibili, requests.RequestException) as e:
        settimanale = pd.DataFrame()
        avvisi.append(f"Dati settimanali non scaricati: {e}")

    giornaliero = pd.DataFrame()
    if con_giornalieri:
        try:
            giornaliero = fetch_eodhd_ohlc(ticker, warmup, fine, "d")
        except (DatiNonDisponibili, requests.RequestException) as e:
            avvisi.append(f"Dati giornalieri non scaricati: {e}")

    return {"mensile": mensile, "settimanale": settimanale,
            "giornaliero": giornaliero, "avvisi": avvisi,
            "periodo": (str(mensile.index[0].date()), str(mensile.index[-1].date()))}
