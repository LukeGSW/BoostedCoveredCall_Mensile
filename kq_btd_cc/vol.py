"""Stimatori di volatilita' realizzata da OHLC giornalieri.

Servono a ricostruire una vol "di mercato" quando non si hanno dati di
volatilita' implicita. Tutti gli stimatori restituiscono una serie di vol
ANNUALIZZATA, allineata alle date dei dati giornalieri.

Regola anti-look-ahead: la vol usata per prezzare il premio del mese M deve
essere quella disponibile all'ultimo giorno PRIMA dell'inizio di M. Se ne
occupa `sigma_at()`.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0

VOL_MODELS = {
    "close_to_close": "Close-to-close (log return)",
    "ewma": "EWMA (RiskMetrics)",
    "parkinson": "Parkinson (High-Low)",
    "garman_klass": "Garman-Klass (OHLC)",
    "rogers_satchell": "Rogers-Satchell (OHLC, drift-robusto)",
    "yang_zhang": "Yang-Zhang (OHLC + gap overnight)",
}


def _log(a: pd.Series, b: pd.Series) -> pd.Series:
    ratio = (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return np.log(ratio.where(ratio > 0))


def vol_close_to_close(df: pd.DataFrame, window: int = 63) -> pd.Series:
    r = _log(df["Close"], df["Close"].shift(1))
    return r.rolling(window, min_periods=max(5, window // 3)).std(ddof=1) * np.sqrt(TRADING_DAYS)


def vol_ewma(df: pd.DataFrame, lam: float = 0.94, window: int = 63) -> pd.Series:
    r = _log(df["Close"], df["Close"].shift(1))
    var = (r ** 2).ewm(alpha=1.0 - lam, min_periods=max(5, window // 3)).mean()
    return np.sqrt(var * TRADING_DAYS)


def vol_parkinson(df: pd.DataFrame, window: int = 63) -> pd.Series:
    hl = _log(df["High"], df["Low"]) ** 2
    var = hl.rolling(window, min_periods=max(5, window // 3)).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * TRADING_DAYS)


def vol_garman_klass(df: pd.DataFrame, window: int = 63) -> pd.Series:
    hl = _log(df["High"], df["Low"]) ** 2
    co = _log(df["Close"], df["Open"]) ** 2
    daily = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    var = daily.rolling(window, min_periods=max(5, window // 3)).mean()
    return np.sqrt(var.clip(lower=0) * TRADING_DAYS)


def vol_rogers_satchell(df: pd.DataFrame, window: int = 63) -> pd.Series:
    ho, lo = _log(df["High"], df["Open"]), _log(df["Low"], df["Open"])
    hc, lc = _log(df["High"], df["Close"]), _log(df["Low"], df["Close"])
    daily = ho * hc + lo * lc
    var = daily.rolling(window, min_periods=max(5, window // 3)).mean()
    return np.sqrt(var.clip(lower=0) * TRADING_DAYS)


def vol_yang_zhang(df: pd.DataFrame, window: int = 63) -> pd.Series:
    """Yang-Zhang: somma pesata di overnight, open-to-close e Rogers-Satchell.

    E' lo stimatore piu' efficiente quando il sottostante ha gap overnight
    significativi (azioni con earnings, crypto sui weekend).
    """
    mp = max(5, window // 3)
    o = _log(df["Open"], df["Close"].shift(1))          # gap overnight
    c = _log(df["Close"], df["Open"])                   # sessione
    ho, lo = _log(df["High"], df["Open"]), _log(df["Low"], df["Open"])
    hc, lc = _log(df["High"], df["Close"]), _log(df["Low"], df["Close"])
    rs = (ho * hc + lo * lc).rolling(window, min_periods=mp).mean()
    vo = o.rolling(window, min_periods=mp).var(ddof=1)
    vc = c.rolling(window, min_periods=mp).var(ddof=1)
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    var = vo + k * vc + (1.0 - k) * rs
    return np.sqrt(var.clip(lower=0) * TRADING_DAYS)


_DISPATCH = {
    "close_to_close": vol_close_to_close,
    "parkinson": vol_parkinson,
    "garman_klass": vol_garman_klass,
    "rogers_satchell": vol_rogers_satchell,
    "yang_zhang": vol_yang_zhang,
}


def realized_vol(
    daily: pd.DataFrame,
    model: str = "yang_zhang",
    window: int = 63,
    long_window: int = 252,
    blend: float = 1.0,
    ewma_lambda: float = 0.94,
) -> pd.Series:
    """Serie di vol annualizzata.

    blend: peso della finestra corta. Con blend < 1 si mescola alla finestra
    lunga (`long_window`), che cattura la mean-reversion della volatilita':
        sigma = blend * sigma_corta + (1 - blend) * sigma_lunga
    """
    if daily is None or daily.empty or "Close" not in daily.columns:
        return pd.Series(dtype=float)

    df = daily.copy()
    for c in ("Open", "High", "Low"):
        if c not in df.columns or df[c].isna().all():
            df[c] = df["Close"]
    df = df[["Open", "High", "Low", "Close"]].astype(float)

    if model == "ewma":
        short = vol_ewma(df, lam=ewma_lambda, window=window)
        long_ = vol_ewma(df, lam=min(0.995, 1.0 - (1.0 - ewma_lambda) / 4.0), window=long_window)
    else:
        fn = _DISPATCH.get(model, vol_yang_zhang)
        short = fn(df, window=window)
        long_ = fn(df, window=long_window)

    b = float(np.clip(blend, 0.0, 1.0))
    out = b * short + (1.0 - b) * long_.reindex(short.index)
    out = out.where(np.isfinite(out))
    # dove la finestra lunga non e' ancora disponibile, usa solo la corta
    return out.fillna(short).rename("realized_vol")


def sigma_at(vol_series: pd.Series, when: pd.Timestamp, fallback: float = np.nan) -> float:
    """Ultima vol disponibile STRETTAMENTE prima di `when` (niente look-ahead)."""
    if vol_series is None or vol_series.empty:
        return fallback
    s = vol_series.dropna()
    if s.empty:
        return fallback
    prior = s.loc[s.index < pd.Timestamp(when)]
    if prior.empty:
        return fallback
    v = float(prior.iloc[-1])
    return v if np.isfinite(v) and v > 0 else fallback


def vol_from_monthly(monthly: pd.DataFrame, window: int = 12) -> pd.Series:
    """Fallback: vol annualizzata stimata dai soli rendimenti mensili.

    Usato quando i dati giornalieri non sono disponibili per il ticker.
    """
    if monthly is None or monthly.empty or "Close" not in monthly.columns:
        return pd.Series(dtype=float)
    r = np.log((monthly["Close"] / monthly["Close"].shift(1)).where(lambda x: x > 0))
    return (r.rolling(window, min_periods=max(3, window // 2)).std(ddof=1)
            * np.sqrt(12.0)).rename("realized_vol")


def vol_summary(vol_series: pd.Series) -> Dict[str, Optional[float]]:
    if vol_series is None or vol_series.dropna().empty:
        return {"media": None, "mediana": None, "min": None, "max": None, "ultima": None}
    s = vol_series.dropna()
    return {
        "media": float(s.mean()), "mediana": float(s.median()),
        "min": float(s.min()), "max": float(s.max()), "ultima": float(s.iloc[-1]),
    }
