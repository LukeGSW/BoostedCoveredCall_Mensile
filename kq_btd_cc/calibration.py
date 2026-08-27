"""Calibrazione dello stimatore di premio sui prezzi reali delle opzioni.

Flusso d'uso:
  1. si scarica da OptionLAB (o da qualunque altra fonte) un file con i prezzi
     reali delle call vicine a delta 0.50 sul sottostante di interesse;
  2. per ogni osservazione il modello ricostruisce il premio partendo dalla sola
     volatilita' realizzata nota PRIMA di quella data, senza mai vedere la IV;
  3. si cerca il VRP che minimizza l'errore quadratico fra premio stimato e
     premio reale, entrambi espressi in frazione dello spot.

Il coefficiente calibrato e' l'unico numero che serve per rendere realistico il
premio su quel sottostante, e finisce nel JSON di export.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .pricing import PremiumModel
from . import vol as volmod

# Sinonimi accettati per l'auto-riconoscimento delle colonne.
ALIAS: Dict[str, List[str]] = {
    "data": ["data", "date", "quote_date", "quotedate", "trade_date", "tradedate",
             "datetime", "day", "giorno", "data_quotazione"],
    "spot": ["spot", "underlying", "underlying_price", "underlyingprice", "sottostante",
             "prezzo_sottostante", "stock_price", "s", "close", "und_price", "prezzo",
             "under_op", "underop", "under_open"],
    "strike": ["strike", "k", "strike_price", "strikeprice", "prezzo_esercizio"],
    "scadenza": ["expiry", "expiration", "exp_date", "expirationdate", "expiry_date",
                 "scadenza", "data_scadenza", "maturity"],
    "dte": ["dte", "days_to_expiry", "days", "giorni", "giorni_a_scadenza", "tenor",
            "days_to_expiration", "t_days"],
    "delta": ["delta", "call_delta", "delta_call"],
    "bid": ["bid", "denaro", "bid_price"],
    "ask": ["ask", "lettera", "ask_price", "offer"],
    "mid": ["mid", "mid_price", "price", "premio", "premium", "last", "close_option",
            "option_price", "prezzo_opzione", "mark", "theo"],
    "tipo": ["type", "tipo", "cp", "call_put", "right", "option_type", "putcall"],
    "iv": ["iv", "implied_vol", "impliedvolatility", "implied_volatility", "volatilita_implicita"],
}


# ----------------------------------------------------------------------------
# Normalizzazione del file caricato
# ----------------------------------------------------------------------------
def _norm(nome: Any) -> str:
    return str(nome).strip().lower().replace(" ", "_").replace("-", "_")


# ----------------------------------------------------------------------------
# Preset OptionLAB
# ----------------------------------------------------------------------------
# L'export di OptionLAB affianca ai trade una seconda serie (Date, DailyEquity)
# molto piu' lunga: se si lasciasse indovinare la colonna della data,
# l'abbinamento automatico prenderebbe 'Date' invece di 'Open'. Meglio
# riconoscere il formato e mappare a mano.
FIRMA_OPTIONLAB = {"ticket", "symbol", "side", "maturity", "right", "under_op", "open_price"}

MAPPATURA_OPTIONLAB: Dict[str, str] = {
    "data": "Open",             # data di apertura della posizione
    "spot": "Under Op",         # sottostante all'apertura
    "mid": "Open Price",        # premio incassato, per unita' di sottostante
    "strike": "Strike",
    "scadenza": "Maturity",
    "tipo": "Right",
}


def e_optionlab(df: pd.DataFrame) -> bool:
    return FIRMA_OPTIONLAB.issubset({_norm(c) for c in df.columns})


def carica_file_opzioni(sorgente: Any, nome: str = "") -> Tuple[pd.DataFrame, bool]:
    """Legge un CSV/Excel di prezzi opzioni. Ritorna (dataframe, e_optionlab)."""
    nome = (nome or getattr(sorgente, "name", "") or "").lower()
    if nome.endswith((".xlsx", ".xls")):
        df = pd.read_excel(sorgente)
    else:
        df = pd.read_csv(sorgente, sep=None, engine="python", decimal=".")
        if df.shape[1] == 1:                      # separatore non riconosciuto
            if hasattr(sorgente, "seek"):
                sorgente.seek(0)
            df = pd.read_csv(sorgente, sep=";", decimal=".")

    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    optionlab = e_optionlab(df)
    if optionlab:
        # tiene solo le righe che sono davvero operazioni
        df = df.dropna(subset=[c for c in ("Ticket", "Open", "Under Op", "Open Price")
                               if c in df.columns])
        for col in ("Side", "Right"):
            if col in df.columns:
                v = df[col].astype(str).str.strip().str.lower()
                atteso = "sell" if col == "Side" else "call"
                if v.str.startswith(atteso).any():
                    df = df[v.str.startswith(atteso)]
    return df.reset_index(drop=True), optionlab


def suggerisci_mappatura(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Prova ad associare ogni campo richiesto a una colonna del file.

    Due passate: prima gli abbinamenti esatti, che si prendono la colonna in
    esclusiva; poi quelli parziali, ma solo sulle colonne rimaste libere e a
    livello di parola intera. Senza questa disciplina 'price' finirebbe per
    agganciare 'underlying_price' e il premio verrebbe letto come lo spot.
    """
    if e_optionlab(df):
        base: Dict[str, Optional[str]] = {campo: None for campo in ALIAS}
        reale = {_norm(c): c for c in df.columns}
        for campo, colonna in MAPPATURA_OPTIONLAB.items():
            base[campo] = reale.get(_norm(colonna))
        return base

    cols = {_norm(c): c for c in df.columns}
    out: Dict[str, Optional[str]] = {campo: None for campo in ALIAS}
    usate: set = set()

    for campo, alias in ALIAS.items():
        for a in alias:
            if a in cols and cols[a] not in usate:
                out[campo] = cols[a]
                usate.add(cols[a])
                break

    for campo, alias in ALIAS.items():
        if out[campo] is not None:
            continue
        for norm_c, orig in cols.items():
            if orig in usate:
                continue
            parole = set(norm_c.split("_"))
            if any(a in parole or a == norm_c for a in alias):
                out[campo] = orig
                usate.add(orig)
                break
    return out


def prepara_osservazioni(
    df: pd.DataFrame,
    mappatura: Dict[str, Optional[str]],
    solo_call: bool = True,
    delta_target: float = 0.50,
    delta_tolleranza: float = 0.10,
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Estrae le colonne utili e costruisce `premio_pct_reale` e `dte`."""
    if df is None or df.empty:
        return pd.DataFrame()

    def col(nome: str) -> Optional[pd.Series]:
        c = mappatura.get(nome)
        return df[c] if c and c in df.columns else None

    out = pd.DataFrame(index=df.index)

    data = col("data")
    if data is None:
        raise ValueError("Manca la colonna con la data della quotazione.")
    out["data"] = pd.to_datetime(data, errors="coerce", dayfirst=dayfirst)

    spot = col("spot")
    if spot is None:
        raise ValueError("Manca la colonna con il prezzo del sottostante.")
    out["spot"] = pd.to_numeric(spot, errors="coerce")

    # Premio: mid esplicito oppure media bid/ask
    mid, bid, ask = col("mid"), col("bid"), col("ask")
    if mid is not None:
        out["premio"] = pd.to_numeric(mid, errors="coerce")
    elif bid is not None and ask is not None:
        b = pd.to_numeric(bid, errors="coerce")
        a = pd.to_numeric(ask, errors="coerce")
        out["premio"] = (b + a) / 2.0
    else:
        raise ValueError("Manca il prezzo dell'opzione (mid oppure bid e ask).")

    # Giorni a scadenza: espliciti oppure dalla data di scadenza
    dte, scad = col("dte"), col("scadenza")
    if dte is not None:
        out["dte"] = pd.to_numeric(dte, errors="coerce")
    elif scad is not None:
        out["dte"] = (pd.to_datetime(scad, errors="coerce", dayfirst=dayfirst)
                      - out["data"]).dt.days
    else:
        raise ValueError("Mancano i giorni a scadenza (dte oppure data di scadenza).")

    for opz in ("strike", "delta", "iv"):
        s = col(opz)
        out[opz] = pd.to_numeric(s, errors="coerce") if s is not None else np.nan

    tipo = col("tipo")
    if tipo is not None and solo_call:
        t = tipo.astype(str).str.strip().str.lower()
        out = out[t.str.startswith(("c", "call"))]

    out = out.dropna(subset=["data", "spot", "premio", "dte"])
    out = out[(out["spot"] > 0) & (out["premio"] > 0) & (out["dte"] > 0)]

    # Se il delta c'e', si tengono solo le opzioni vicine al delta obiettivo
    if out["delta"].notna().any():
        d = out["delta"].abs()
        out = out[(d - delta_target).abs() <= delta_tolleranza]

    out["premio_pct_reale"] = out["premio"] / out["spot"]
    out["T"] = out["dte"] / 365.0

    # Guardia contro una mappatura sbagliata: una call vicina a delta 0.50 con
    # scadenza breve non vale meta' del sottostante ne' un valore identico su
    # ogni riga. Meglio fermarsi qui che restituire una calibrazione insensata.
    if not out.empty:
        mediana = float(out["premio_pct_reale"].median())
        if mediana > 0.50:
            raise ValueError(
                f"Il premio risulta pari al {mediana:.0%} dello spot: la colonna del prezzo "
                "dell'opzione sembra puntare al sottostante. Controlla la mappatura."
            )
        if len(out) > 3 and float(out["premio_pct_reale"].std(ddof=0)) < 1e-9:
            raise ValueError(
                "Il premio in frazione dello spot e' identico su tutte le righe: "
                "controlla quale colonna e' stata associata al prezzo dell'opzione."
            )
    return out.sort_values("data").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Aggancio alla volatilita' realizzata
# ----------------------------------------------------------------------------
def aggancia_volatilita(
    oss: pd.DataFrame,
    vol_series: pd.Series,
) -> pd.DataFrame:
    """Aggiunge a ogni osservazione la vol realizzata nota PRIMA di quella data."""
    if oss.empty:
        return oss
    out = oss.copy()
    out["sigma_realizzata"] = [
        volmod.sigma_at(vol_series, d, fallback=np.nan) for d in out["data"]
    ]
    return out.dropna(subset=["sigma_realizzata"])


# ----------------------------------------------------------------------------
# Stima e fit
# ----------------------------------------------------------------------------
def _premi_stimati(oss: pd.DataFrame, modello: PremiumModel) -> np.ndarray:
    return np.array([
        modello.quote(float(r.spot), float(r.sigma_realizzata), float(r.T))["premium_pct"]
        for r in oss.itertuples()
    ])


OBIETTIVI = {
    "livello": "Allinea il livello medio dei premi (consigliato per il backtest)",
    "assoluto": "Minimizza l'errore quadratico in punti di spot",
    "relativo": "Minimizza l'errore quadratico in percentuale del premio",
}


def _errore(oss: pd.DataFrame, modello: PremiumModel, obiettivo: str = "livello") -> float:
    """Funzione da minimizzare.

    Misurato sui premi reali di sei sottostanti (1.666 operazioni), i tre
    obiettivi si comportano cosi':

      * "livello"  azzera lo scarto sul premio medio incassato. E' quello che
        conta in un backtest, dove a fare il risultato e' il totale dei premi.
      * "assoluto" lascia una sottostima del 7-14% perche' il fit insegue le
        osservazioni a premio alto, che pesano di piu' in valore assoluto.
      * "relativo" pesa tutte le osservazioni allo stesso modo ma, con un
        predittore rumoroso, spinge la stima verso il basso: la sottostima
        misurata arriva al 20-48%.
    """
    stimati = _premi_stimati(oss, modello)
    reali = oss["premio_pct_reale"].values
    if obiettivo == "livello":
        m_s, m_r = float(np.mean(stimati)), float(np.mean(reali))
        return float((m_s - m_r) ** 2)
    if obiettivo == "relativo":
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(reali > 0, stimati / reali - 1.0, np.nan)
        rel = rel[np.isfinite(rel)]
        return float(np.mean(rel ** 2)) if rel.size else float("inf")
    return float(np.mean((stimati - reali) ** 2))


def calibra_vrp(
    oss: pd.DataFrame,
    base: Optional[PremiumModel] = None,
    vrp_min: float = 0.50,
    vrp_max: float = 2.50,
    fit_addendo: bool = False,
    obiettivo: str = "livello",
) -> Dict[str, Any]:
    """Cerca il VRP (e opzionalmente l'addendo) che minimizza l'errore.

    Ricerca su griglia seguita da raffinamento locale: niente scipy, e la
    funzione obiettivo e' monotona a tratti quindi la griglia basta.
    """
    if oss is None or oss.empty or len(oss) < 3:
        return {"ok": False, "errore": "Servono almeno 3 osservazioni valide."}

    base = base or PremiumModel()
    kwargs = {k: v for k, v in base.to_dict().items() if k not in ("vrp", "vrp_add")}

    def prova(vrp: float, add: float) -> float:
        return _errore(oss, PremiumModel(vrp=vrp, vrp_add=add, **kwargs), obiettivo)

    migliore_vrp, migliore_add = base.vrp, base.vrp_add
    passi_add = np.linspace(-0.15, 0.15, 13) if fit_addendo else np.array([base.vrp_add])

    lo, hi = vrp_min, vrp_max
    for giro in range(4):                       # griglia via via piu' fitta
        griglia = np.linspace(lo, hi, 41)
        best = np.inf
        for add in passi_add:
            for v in griglia:
                e = prova(float(v), float(add))
                if e < best:
                    best, migliore_vrp, migliore_add = e, float(v), float(add)
        passo = (hi - lo) / 40.0
        lo, hi = max(vrp_min, migliore_vrp - passo * 2), min(vrp_max, migliore_vrp + passo * 2)
        if hi - lo < 1e-4:
            break

    modello = PremiumModel(vrp=migliore_vrp, vrp_add=migliore_add, **kwargs)
    out = {"ok": True, "modello": modello, **valuta(oss, modello)}
    out["metriche"]["obiettivo"] = obiettivo
    return out


def valuta(oss: pd.DataFrame, modello: PremiumModel) -> Dict[str, Any]:
    """Metriche di bonta' del fit, in frazione dello spot."""
    stimati = _premi_stimati(oss, modello)
    reali = oss["premio_pct_reale"].values
    err = stimati - reali
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((reali - reali.mean()) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs(err / np.where(reali == 0, np.nan, reali))))

    dettaglio = oss.copy()
    dettaglio["premio_pct_stimato"] = stimati
    dettaglio["errore"] = err
    dettaglio["sigma_implicita_stimata"] = [modello.implied_sigma(s)
                                            for s in oss["sigma_realizzata"].values]
    if oss["iv"].notna().any():
        dettaglio["errore_vol"] = dettaglio["sigma_implicita_stimata"] - oss["iv"]

    metriche = {
        "n": int(len(oss)),
        "vrp_calibrato": float(modello.vrp),
        "vrp_addendo": float(modello.vrp_add),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "mape": mape if np.isfinite(mape) else None,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "premio_medio_reale": float(reali.mean()),
        "premio_medio_stimato": float(stimati.mean()),
        "periodo_inizio": str(pd.Timestamp(oss["data"].min()).date()),
        "periodo_fine": str(pd.Timestamp(oss["data"].max()).date()),
        "dte_medio": float(oss["dte"].mean()),
    }
    if "errore_vol" in dettaglio.columns:
        ev = dettaglio["errore_vol"].dropna()
        if not ev.empty:
            metriche["vol_bias"] = float(ev.mean())
            metriche["vol_mae"] = float(ev.abs().mean())
            metriche["iv_media_reale"] = float(oss["iv"].dropna().mean())

    return {"metriche": metriche, "dettaglio": dettaglio}


def confronta_modelli_vol(
    oss_base: pd.DataFrame,
    daily: pd.DataFrame,
    cfg_vol: Dict[str, Any],
    modelli: Optional[Iterable[str]] = None,
    base: Optional[PremiumModel] = None,
    obiettivo: str = "livello",
) -> pd.DataFrame:
    """Calibra e valuta ogni stimatore di volatilita': quale descrive meglio i prezzi reali."""
    modelli = list(modelli or volmod.VOL_MODELS.keys())
    righe = []
    for nome in modelli:
        try:
            vs = volmod.realized_vol(
                daily, model=nome,
                window=int(cfg_vol.get("vol_window", 63)),
                long_window=int(cfg_vol.get("vol_long_window", 252)),
                blend=float(cfg_vol.get("vol_blend", 1.0)),
                ewma_lambda=float(cfg_vol.get("ewma_lambda", 0.94)),
            )
            oss = aggancia_volatilita(oss_base, vs)
            if oss.empty or len(oss) < 3:
                continue
            fit = calibra_vrp(oss, base=base, obiettivo=obiettivo)
            if not fit.get("ok"):
                continue
            m = fit["metriche"]
            righe.append({
                "modello": volmod.VOL_MODELS.get(nome, nome),
                "chiave": nome,
                "vrp": m["vrp_calibrato"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "bias": m["bias"],
                "mape": m["mape"],
                "r2": m["r2"],
                "n": m["n"],
            })
        except Exception:
            continue
    if not righe:
        return pd.DataFrame()
    ordine = "mape" if obiettivo == "relativo" else "rmse"
    return pd.DataFrame(righe).sort_values(ordine).reset_index(drop=True)


def pacchetto_export(fit: Dict[str, Any], nome_file: Optional[str] = None,
                     ticker: Optional[str] = None) -> Dict[str, Any]:
    """Riduce il risultato della calibrazione a un blocco JSON-safe."""
    if not fit or not fit.get("ok"):
        return {}
    det = fit.get("dettaglio")
    colonne = ["data", "spot", "strike", "dte", "delta", "premio",
               "premio_pct_reale", "premio_pct_stimato", "errore",
               "sigma_realizzata", "sigma_implicita_stimata"]
    campione = []
    if isinstance(det, pd.DataFrame) and not det.empty:
        cols = [c for c in colonne if c in det.columns]
        campione = det[cols].assign(data=det["data"].dt.strftime("%Y-%m-%d")).to_dict("records")
    return {
        "file_sorgente": nome_file,
        "ticker": ticker,
        "modello_premio": fit["modello"].to_dict() if fit.get("modello") else None,
        "metriche": fit.get("metriche", {}),
        "osservazioni": campione,
    }
