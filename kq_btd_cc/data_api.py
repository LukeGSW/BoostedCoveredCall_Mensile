"""Accesso ai dati EODHD, con cache, tetto giornaliero e OHLC aggiustati.

Le chiamate sono memorizzate in cache da Streamlit: cambiare un parametro che
non tocca i dati (premio, boost, tema) non riscarica nulla. Su Streamlit Cloud
la cache e' condivisa fra tutte le sessioni, quindi il costo cresce col numero
di combinazioni ticker/periodo distinte, non col numero di utenti o di backtest.

A protezione della chiave c'e' comunque un TETTO GIORNALIERO sul numero di
download effettivi, condiviso da tutta l'app e azzerato a mezzanotte UTC. Conta
solo le chiamate vere: una richiesta servita dalla cache non consuma budget.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://eodhd.com/api/eod/{ticker}"
TIMEOUT = 45
COLONNE = ["Open", "High", "Low", "Close", "Volume"]

# Tetto giornaliero di download, sovrascrivibile dai secrets con
# EODHD_LIMITE_GIORNALIERO. Ogni backtest su una combinazione ticker/periodo mai
# vista costa tre chiamate (mensile, settimanale, giornaliero); tutto il resto
# arriva dalla cache e non conta.
LIMITE_GIORNALIERO_DEFAULT = 5000
CHIAMATE_PER_SERIE = 3

# Giorni di storico extra scaricati prima della data di inizio, per dare allo
# stimatore di volatilita' una finestra di riscaldamento gia' piena al mese 1.
# Dimensionato sulla finestra lunga di default (504 giorni di borsa ~ 2 anni).
WARMUP_GIORNI = 900


class DatiNonDisponibili(RuntimeError):
    """Il download e' andato a buon fine ma non ci sono dati utilizzabili."""


class ChiaveMancante(RuntimeError):
    """Nessuna API key EODHD configurata."""


class LimiteGiornaliero(RuntimeError):
    """Esaurito il tetto giornaliero di chiamate condiviso da tutta l'app."""


# ----------------------------------------------------------------------------
# Tetto giornaliero condiviso
# ----------------------------------------------------------------------------
def limite_giornaliero() -> int:
    """Tetto configurato, dai secrets o dall'ambiente, altrimenti il default."""
    for fonte in (_da_secrets, os.getenv):
        try:
            v = fonte("EODHD_LIMITE_GIORNALIERO")
        except Exception:
            v = None
        if v is not None:
            try:
                return max(0, int(v))
            except (TypeError, ValueError):
                pass
    return LIMITE_GIORNALIERO_DEFAULT


def _da_secrets(chiave: str):
    import streamlit as st
    return st.secrets[chiave] if chiave in st.secrets else None


def _contatore() -> Dict[str, object]:
    """Contatore unico per tutta l'app, non per sessione.

    `st.cache_resource` restituisce lo stesso oggetto a ogni sessione e a ogni
    rerun, quindi il conteggio e' davvero globale finche' il processo vive. Fuori
    da Streamlit (test, script) si ripiega su un dizionario di modulo.
    """
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _singleton() -> Dict[str, object]:
            return {"giorno": None, "usate": 0}

        return _singleton()
    except Exception:
        return _CONTATORE_LOCALE


_CONTATORE_LOCALE: Dict[str, object] = {"giorno": None, "usate": 0}


def _oggi_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def stato_budget() -> Dict[str, object]:
    """Quante chiamate sono state usate oggi e quante ne restano."""
    c = _contatore()
    oggi = _oggi_utc()
    if c.get("giorno") != oggi:          # nuovo giorno: si riparte da zero
        c["giorno"], c["usate"] = oggi, 0
    tetto = limite_giornaliero()
    usate = int(c["usate"])
    return {"giorno": oggi, "usate": usate, "tetto": tetto,
            "residue": max(0, tetto - usate), "esaurito": usate >= tetto}


def _consuma(quante: int = 1) -> None:
    """Scala il budget, o solleva LimiteGiornaliero se non basta."""
    st_ = stato_budget()
    if st_["residue"] < quante:
        raise LimiteGiornaliero(messaggio_limite())
    _contatore()["usate"] = int(_contatore()["usate"]) + quante


def messaggio_limite() -> str:
    st_ = stato_budget()
    return (
        f"**Limite giornaliero raggiunto.** Questa dashboard scarica i dati con "
        f"una sola chiave, condivisa da tutti quelli che la usano, e ha un tetto "
        f"di {st_['tetto']:,} scaricamenti al giorno per non esaurirla. Oggi sono "
        f"finiti tutti ({st_['usate']:,}). **Il limite si azzera domani.** "
        f"Nel frattempo puoi continuare a lanciare backtest sui ticker e sui "
        f"periodi gia' scaricati oggi: quelli arrivano dalla cache e non "
        f"consumano nulla."
    )


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
    # Questo corpo gira SOLO quando la cache non ha gia' la risposta: e' il punto
    # esatto in cui parte una chiamata vera, ed e' quindi dove va scalato il budget.
    _consuma(1)
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

    # Nessun controllo a monte: il budget va scalato solo dove parte una chiamata
    # vera, cioe' dentro `_fetch_cached` quando la cache non ha la risposta. Un
    # controllo qui bloccherebbe anche i ticker gia' scaricati oggi, che invece
    # devono continuare a funzionare: e' quello che promette il messaggio.
    avvisi = []
    mensile = fetch_eodhd_ohlc(ticker, inizio, fine, "m")
    if mensile.empty or len(mensile) < 2:
        raise DatiNonDisponibili(
            f"Servono almeno due mesi di storico per '{ticker}' nel periodo richiesto."
        )

    try:
        settimanale = fetch_eodhd_ohlc(ticker, warmup, fine, "w")
    except LimiteGiornaliero:
        # Il mensile c'era gia' in cache ma il resto no: si va avanti con quello
        # che c'e', come per qualunque altro dato mancante.
        settimanale = pd.DataFrame()
        avvisi.append("Dati settimanali non scaricati: limite giornaliero di "
                      "scaricamenti raggiunto, si azzera domani.")
    except (DatiNonDisponibili, requests.RequestException) as e:
        settimanale = pd.DataFrame()
        avvisi.append(f"Dati settimanali non scaricati: {e}")

    giornaliero = pd.DataFrame()
    if con_giornalieri:
        try:
            giornaliero = fetch_eodhd_ohlc(ticker, warmup, fine, "d")
        except LimiteGiornaliero:
            avvisi.append("Dati giornalieri non scaricati: limite giornaliero di "
                          "scaricamenti raggiunto, si azzera domani. Il conto restera' "
                          "valorizzato solo a fine periodo.")
        except (DatiNonDisponibili, requests.RequestException) as e:
            avvisi.append(f"Dati giornalieri non scaricati: {e}")

    return {"mensile": mensile, "settimanale": settimanale,
            "giornaliero": giornaliero, "avvisi": avvisi,
            "periodo": (str(mensile.index[0].date()), str(mensile.index[-1].date()))}
