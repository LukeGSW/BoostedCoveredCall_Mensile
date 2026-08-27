"""Metriche di performance e rischio.

Punto chiave: la strategia riceve capitale nuovo durante l'anno (i BTD) e a
inizio anno puo' richiedere un rabbocco. Calcolare i rendimenti come
`equity.pct_change()` su una curva gonfiata dai versamenti restituisce numeri
privi di significato. Qui tutte le metriche di rendimento e rischio sono
costruite sul RENDIMENTO TIME-WEIGHTED, che neutralizza i flussi:

    r_t = valore_t / (valore_{t-1} + versamenti_t) - 1

I drawdown sono riportati in tre versioni, perche' misurano cose diverse:
  * `dd_twr_pct`   -> quanto ha perso la strategia in percentuale (pulito dai flussi)
  * `dd_valore`    -> quanto e' sceso il valore del conto in valuta
  * `pnl_dd`       -> quanto e' sceso l'utile netto dei versamenti
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MESI_ANNO = 12.0


# ----------------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------------
def _f(x: Any) -> Optional[float]:
    """Converte in float JSON-safe (NaN/inf -> None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def drawdown_durations(dd: pd.Series) -> List[int]:
    """Durate in mesi degli episodi di drawdown (valori < 0)."""
    if dd is None or dd.empty:
        return []
    out, cur = [], 0
    for v in dd.values:
        if v < -1e-12:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _annualize(returns: pd.Series) -> Dict[str, Optional[float]]:
    r = returns.dropna()
    r = r[np.isfinite(r)]
    if r.empty:
        return {"cagr": None, "vol": None, "sharpe": None, "sortino": None}
    n = len(r)
    tot = float(np.prod(1.0 + r.values))
    anni = n / MESI_ANNO
    cagr = tot ** (1.0 / anni) - 1.0 if (anni > 0 and tot > 0) else None
    vol = float(r.std(ddof=1) * np.sqrt(MESI_ANNO)) if n > 1 else None
    down = r[r < 0]
    dvol = float(np.sqrt((down.values ** 2).mean()) * np.sqrt(MESI_ANNO)) if len(down) else None
    sharpe = (cagr / vol) if (cagr is not None and vol) else None
    sortino = (cagr / dvol) if (cagr is not None and dvol) else None
    return {"cagr": _f(cagr), "vol": _f(vol), "sharpe": _f(sharpe), "sortino": _f(sortino)}


def var_cvar(returns: pd.Series, confidence: float = 0.99) -> Dict[str, Optional[float]]:
    """VaR e CVaR storici mensili (valori negativi = perdita)."""
    r = returns.dropna()
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return {"var": None, "cvar": None}
    q = float(np.quantile(r.values, 1.0 - confidence))
    coda = r[r <= q]
    return {"var": _f(q), "cvar": _f(coda.mean()) if len(coda) else _f(q)}


# ----------------------------------------------------------------------------
# Metriche complete su un risultato di variante
# ----------------------------------------------------------------------------
def compute_metrics(df: pd.DataFrame, confidence: float = 0.99) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}

    r = df["twr_mese"]
    ann = _annualize(r)
    vc = var_cvar(r, confidence)

    valore_fin = float(df["valore_portafoglio"].iloc[-1])
    versamenti = float(df["versamenti_cum"].iloc[-1])
    pnl = valore_fin - versamenti
    capitale_impiegato = (df["quote_coperte"] + df["quote_extra"]) * df["close"]

    dd_pct = df["dd_twr_pct"]
    maxdd_pct = _f(dd_pct.min())
    durate = drawdown_durations(dd_pct)
    calmar = (ann["cagr"] / abs(maxdd_pct)) if (ann["cagr"] is not None and maxdd_pct) else None

    twr_tot = float(np.prod(1.0 + r.dropna().values)) - 1.0 if len(r.dropna()) else None
    mesi = int(len(df))
    positivi = int((r > 0).sum())

    # Buy & Hold a parita' di versamenti: il confronto che conta davvero
    bh = _bh_block(df, confidence)
    riduzione_dd = None
    if maxdd_pct is not None and bh["bh_max_dd_pct"]:
        riduzione_dd = 1.0 - abs(maxdd_pct) / abs(bh["bh_max_dd_pct"])
    extra_cagr = None
    if ann["cagr"] is not None and bh["bh_cagr"] is not None:
        extra_cagr = ann["cagr"] - bh["bh_cagr"]

    return {
        "periodo_inizio": str(df.index[0].date()),
        "periodo_fine": str(df.index[-1].date()),
        "mesi": mesi,
        "anni": _f(mesi / MESI_ANNO),

        # Denaro
        "valore_finale": _f(valore_fin),
        "versamenti_totali": _f(versamenti),
        "pnl_netto": _f(pnl),
        "roi_su_versamenti": _f(pnl / versamenti) if versamenti else None,
        "capitale_medio_impiegato": _f(capitale_impiegato.mean()),
        "capitale_max_impiegato": _f(capitale_impiegato.max()),
        "cassa_finale": _f(df["cassa"].iloc[-1]),
        "finanziamento_massimo": _f(-min(0.0, float(df["cassa"].min()))),
        "mesi_a_debito": int((df["cassa"] < -1e-9).sum()),
        "interessi_netti": _f(df["interessi"].sum()) if "interessi" in df.columns else None,

        # Rendimento (time-weighted)
        "twr_totale": _f(twr_tot),
        "cagr": ann["cagr"],
        "volatilita_annua": ann["vol"],
        "sharpe": ann["sharpe"],
        "sortino": ann["sortino"],
        "calmar": _f(calmar),

        # Rischio
        "max_dd_pct": maxdd_pct,
        "max_dd_valore": _f(df["dd_valore"].min()),
        "max_dd_pnl": _f(df["pnl_dd"].min()),
        "dd_durata_max_mesi": int(max(durate)) if durate else 0,
        "dd_durata_media_mesi": _f(np.mean(durate)) if durate else None,
        "var_mensile": vc["var"],
        "cvar_mensile": vc["cvar"],
        "var_confidence": _f(confidence),

        # Distribuzione
        "mesi_positivi": positivi,
        "hit_rate": _f(positivi / mesi) if mesi else None,
        "miglior_mese": _f(r.max()),
        "peggior_mese": _f(r.min()),

        # Opzioni e BTD
        "premi_totali": _f(df["premio"].sum()),
        "intrinseco_totale": _f(df["intrinseco_pagato"].sum()),
        "netto_opzioni": _f(df["netto_opzione"].sum()),
        "premio_pct_medio": _f(df.loc[df["premio_pct"] > 0, "premio_pct"].mean()),
        "mesi_call_assegnata": int((df["intrinseco_pagato"] > 0).sum()),
        "btd_numero": int((df["btd_importo"] > 0).sum()),
        "btd_totale": _f(df["btd_importo"].sum()),
        "btd_medio": _f(df.loc[df["btd_importo"] > 0, "btd_importo"].mean()),

        # Benchmark a parita' di flussi
        "bh_stessi_flussi_finale": _f(df["bh_stessi_flussi"].iloc[-1]),
        "bh_stessi_flussi_pnl": _f(df["bh_stessi_flussi"].iloc[-1] - versamenti),
        **bh,

        # Confronto diretto con il benchmark
        "riduzione_dd_vs_bh": _f(riduzione_dd),
        "extra_cagr_vs_bh": _f(extra_cagr),
        "extra_pnl_vs_bh": _f(pnl - (df["bh_stessi_flussi"].iloc[-1] - versamenti)),
    }


def _bh_block(df: pd.DataFrame, confidence: float) -> Dict[str, Any]:
    """Metriche del Buy & Hold che riceve gli stessi versamenti della strategia."""
    if "bh_twr_mese" not in df.columns:
        return {"bh_twr_totale": None, "bh_cagr": None, "bh_volatilita_annua": None,
                "bh_sharpe": None, "bh_sortino": None, "bh_max_dd_pct": None,
                "bh_max_dd_valore": None, "bh_var_mensile": None, "bh_calmar": None}
    rb = df["bh_twr_mese"]
    ab = _annualize(rb)
    dd = _f(df["bh_dd_twr_pct"].min())
    tot = float(np.prod(1.0 + rb.dropna().values)) - 1.0 if len(rb.dropna()) else None
    return {
        "bh_twr_totale": _f(tot),
        "bh_cagr": ab["cagr"],
        "bh_volatilita_annua": ab["vol"],
        "bh_sharpe": ab["sharpe"],
        "bh_sortino": ab["sortino"],
        "bh_max_dd_pct": dd,
        "bh_max_dd_valore": _f(df["bh_dd_valore"].min()),
        "bh_calmar": _f(ab["cagr"] / abs(dd)) if (ab["cagr"] is not None and dd) else None,
        "bh_var_mensile": var_cvar(rb, confidence)["var"],
    }


def metrics_table(risultati: Dict[str, Any]) -> pd.DataFrame:
    """Tabella comparativa delle varianti, pronta per la dashboard."""
    righe = {}
    for _, res in risultati.items():
        mt = res.get("metrics") or {}
        if mt:
            righe[res["label"]] = mt
    if not righe:
        return pd.DataFrame()
    ordine = [
        "valore_finale", "versamenti_totali", "pnl_netto", "roi_su_versamenti",
        "twr_totale", "cagr", "volatilita_annua", "sharpe", "sortino", "calmar",
        "max_dd_pct", "max_dd_valore", "dd_durata_max_mesi",
        "var_mensile", "cvar_mensile", "hit_rate", "miglior_mese", "peggior_mese",
        "premi_totali", "intrinseco_totale", "netto_opzioni", "premio_pct_medio",
        "mesi_call_assegnata", "btd_numero", "btd_totale",
        "capitale_medio_impiegato", "finanziamento_massimo", "mesi_a_debito",
        "bh_cagr", "bh_volatilita_annua", "bh_sharpe", "bh_max_dd_pct",
        "bh_stessi_flussi_pnl", "riduzione_dd_vs_bh", "extra_cagr_vs_bh", "extra_pnl_vs_bh",
    ]
    df = pd.DataFrame(righe)
    return df.reindex([k for k in ordine if k in df.index])


ETICHETTE = {
    "valore_finale": "Valore finale conto",
    "versamenti_totali": "Capitale versato (totale)",
    "pnl_netto": "Utile netto dei versamenti",
    "roi_su_versamenti": "ROI sul capitale versato",
    "twr_totale": "Rendimento totale (time-weighted)",
    "cagr": "CAGR",
    "volatilita_annua": "Volatilita annua",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "max_dd_pct": "Max drawdown (%)",
    "max_dd_valore": "Max drawdown (valuta)",
    "dd_durata_max_mesi": "Drawdown piu lungo (mesi)",
    "var_mensile": "VaR mensile",
    "cvar_mensile": "CVaR mensile",
    "hit_rate": "Mesi positivi",
    "miglior_mese": "Miglior mese",
    "peggior_mese": "Peggior mese",
    "premi_totali": "Premi incassati",
    "intrinseco_totale": "Intrinseco pagato sulle call",
    "netto_opzioni": "Risultato netto opzioni",
    "premio_pct_medio": "Premio medio (% dello spot)",
    "mesi_call_assegnata": "Mesi con call in-the-money",
    "btd_numero": "Numero di acquisti BTD",
    "btd_totale": "Capitale investito in BTD",
    "capitale_medio_impiegato": "Capitale medio impiegato",
    "finanziamento_massimo": "Massimo saldo a debito",
    "mesi_a_debito": "Mesi con saldo a debito",
    "bh_stessi_flussi_pnl": "Utile del B&H a parita di flussi",
    "bh_cagr": "CAGR del B&H a parita di flussi",
    "bh_volatilita_annua": "Volatilita annua del B&H",
    "bh_sharpe": "Sharpe del B&H",
    "bh_max_dd_pct": "Max drawdown del B&H (%)",
    "riduzione_dd_vs_bh": "Riduzione del drawdown vs B&H",
    "extra_cagr_vs_bh": "CAGR in piu rispetto al B&H",
    "extra_pnl_vs_bh": "Utile in piu rispetto al B&H",
}

FORMATI = {
    "roi_su_versamenti": "pct", "twr_totale": "pct", "cagr": "pct",
    "volatilita_annua": "pct", "max_dd_pct": "pct", "var_mensile": "pct",
    "cvar_mensile": "pct", "hit_rate": "pct", "miglior_mese": "pct",
    "peggior_mese": "pct", "premio_pct_medio": "pct",
    "bh_cagr": "pct", "bh_volatilita_annua": "pct", "bh_max_dd_pct": "pct",
    "riduzione_dd_vs_bh": "pct", "extra_cagr_vs_bh": "pct",
    "sharpe": "num", "sortino": "num", "calmar": "num", "bh_sharpe": "num",
    "dd_durata_max_mesi": "int", "mesi_call_assegnata": "int", "btd_numero": "int",
    "mesi_a_debito": "int",
}


def format_value(key: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n.d."
    kind = FORMATI.get(key, "cur")
    if kind == "pct":
        return f"{value * 100:,.2f}%"
    if kind == "num":
        return f"{value:,.2f}"
    if kind == "int":
        return f"{int(value):,d}"
    return f"${value:,.0f}"
