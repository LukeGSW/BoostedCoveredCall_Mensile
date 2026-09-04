"""Metriche di performance e rischio.

La misura di rendimento e' il RENDIMENTO SEMPLICE ANNUO SUL CAPITALE INVESTITO,
non un tasso composto. La strategia liquida tutto a dicembre e riparte a
gennaio: ogni anno e' un ciclo chiuso a se' stante, e il numero naturale e'
quanto ha reso il capitale impiegato in quel ciclo.

    rendimento_anno = risultato dell'anno / (capitale di gennaio + BTD dell'anno)

Al denominatore c'e' solo il denaro che l'investitore tira davvero fuori: i
premi reinvestiti non entrano, perche' arrivano dal mercato. Un CAGR calcolato
sul conto intero direbbe altro, perche' il conto include la cassa ferma che non
lavora e che dopo qualche anno puo' essere meta' del totale.

Il rischio resta misurato sul rendimento time-weighted di ogni periodo, che
neutralizza i flussi di cassa:

    r_t = valore_t / (valore_{t-1} + versamenti_t) - 1

ATTENZIONE ALLA FREQUENZA DI VALORIZZAZIONE. Le metriche di questo modulo si
calcolano sulla serie di PERIODO, cioe' su un valore per barra. Un crollo
rientrato prima della chiusura della barra non compare: il drawdown misurato
cosi' e' sistematicamente piu' tenero di quello vero. Il drawdown misurato sui
prezzi di ogni giorno sta in `giornaliero.py` e arriva nelle stesse metriche con
le chiavi `max_dd_giornaliero_pct` e `max_dd_intraday_pct`; `dd_nascosto_dal_periodo`
dice quanti punti si perdevano guardando solo le chiusure di periodo. Per
giudicare il rischio si guarda quello giornaliero.

I drawdown sono riportati in tre versioni, perche' misurano cose diverse:
  * `dd_twr_pct`   -> quanto ha perso la strategia in percentuale (pulito dai flussi)
  * `dd_valore`    -> quanto e' sceso il valore del conto in valuta
  * `pnl_dd`       -> quanto e' sceso l'utile netto dei versamenti

Per CONFRONTARE due strategie va usato solo `dd_twr_pct`. Quello in valuta
dipende da quanto capitale ciascuna sta facendo lavorare: i Buy-The-Dip portano
la strategia a impiegare piu' denaro del benchmark, quindi in una discesa perde
piu' dollari pur perdendo una percentuale sensibilmente minore. Su S&P 500
2000-2026 la variante Cash perde 53.272 dollari contro i 48.338 del benchmark,
ma il 33,8% contro il 46,3%.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .cadenza import PERIODI_ANNO, adatta_dizionario, normalizza

# Quanti passi elementari entrano in un anno. Resta 12 come default perche' la
# cadenza mensile e' quella originale, ma ogni funzione accetta il valore vero.
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
    """Durate in periodi (mesi o settimane) degli episodi di drawdown (valori < 0)."""
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


def rendimenti_annuali(df: pd.DataFrame) -> pd.DataFrame:
    """Rendimento semplice di ogni anno sul capitale davvero investito.

    La strategia liquida tutto a dicembre e riparte a gennaio, quindi ogni anno
    e' un ciclo chiuso: il rendimento naturale e' quello semplice sul capitale
    impiegato in quel ciclo, non un CAGR calcolato sul conto intero. Il conto
    include la cassa ferma, che non lavora, e diluisce qualunque tasso composto.

    Il capitale al denominatore e' quello che l'investitore tira davvero fuori:
    capitale di gennaio piu' gli acquisti sui cali. I premi reinvestiti non
    entrano, perche' arrivano dal mercato.
    """
    if df is None or df.empty or "capitale_impiegato_anno" not in df.columns:
        return pd.DataFrame()
    righe, valore_prec = [], 0.0
    for anno, g in df.groupby("anno", sort=True):
        versato = float(g["versamento_mese"].sum())
        valore_fine = float(g["valore_portafoglio"].iloc[-1])
        risultato = valore_fine - valore_prec - versato
        capitale = float(g["capitale_impiegato_anno"].iloc[-1])
        righe.append({
            "anno": int(anno), "periodi": int(len(g)),
            "capitale_investito": capitale,
            "risultato": risultato,
            "rendimento": risultato / capitale if capitale > 0 else np.nan,
        })
        valore_prec = valore_fine
    return pd.DataFrame(righe).set_index("anno")


def _blocco_rendimento(df: pd.DataFrame, prefisso: str = "") -> Dict[str, Any]:
    """Statistiche sui rendimenti annuali: media, dispersione, rapporto fra i due."""
    ann = rendimenti_annuali(df)
    p = prefisso
    if ann.empty:
        return {f"{p}rendimento_medio": None, f"{p}rendimento_mediano": None,
                f"{p}rendimento_volatilita": None, f"{p}rendimento_su_rischio": None,
                f"{p}anni_positivi": 0, f"{p}anni_totali": 0,
                f"{p}miglior_anno": None, f"{p}peggior_anno": None,
                f"{p}capitale_investito_medio": None}
    r = ann["rendimento"].dropna()
    media = float(r.mean()) if len(r) else None
    dev = float(r.std(ddof=1)) if len(r) > 1 else None
    return {
        f"{p}rendimento_medio": _f(media),
        f"{p}rendimento_mediano": _f(r.median()) if len(r) else None,
        f"{p}rendimento_volatilita": _f(dev),
        f"{p}rendimento_su_rischio": _f(media / dev) if (media is not None and dev) else None,
        f"{p}anni_positivi": int((r > 0).sum()),
        f"{p}anni_totali": int(len(r)),
        f"{p}miglior_anno": _f(r.max()) if len(r) else None,
        f"{p}peggior_anno": _f(r.min()) if len(r) else None,
        f"{p}capitale_investito_medio": _f(ann["capitale_investito"].mean()),
    }


def var_cvar(returns: pd.Series, confidence: float = 0.99) -> Dict[str, Optional[float]]:
    """VaR e CVaR storici sul rendimento di un periodo (valori negativi = perdita)."""
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
def compute_metrics(df: pd.DataFrame, confidence: float = 0.99,
                    periodi_anno: float = MESI_ANNO) -> Dict[str, Any]:
    """Metriche di una variante. `periodi_anno` e' 12 sul mensile, 52 sul settimanale.

    Serve solo a convertire il numero di barre in anni: tutto il resto (il
    rendimento annuo sul capitale investito, i drawdown, il VaR) e' gia'
    indipendente dalla cadenza perche' si appoggia agli anni di calendario o
    alla singola barra.
    """
    if df is None or df.empty:
        return {}
    ppa = float(periodi_anno) or MESI_ANNO

    r = df["twr_mese"]
    vc = var_cvar(r, confidence)
    rend = _blocco_rendimento(df)

    valore_fin = float(df["valore_portafoglio"].iloc[-1])
    versamenti = float(df["versamenti_cum"].iloc[-1])
    pnl = valore_fin - versamenti
    capitale_impiegato = (df["quote_coperte"] + df["quote_extra"]) * df["close"]

    dd_pct = df["dd_twr_pct"]
    maxdd_pct = _f(dd_pct.min())
    durate = drawdown_durations(dd_pct)
    medio = rend["rendimento_medio"]
    rend_su_dd = (medio / abs(maxdd_pct)) if (medio is not None and maxdd_pct) else None

    mesi = int(len(df))
    positivi = int((r > 0).sum())

    # Buy & Hold a parita' di versamenti, e il solo sottostante con lo stesso
    # ciclo annuale: quest'ultimo e' il confronto a parita' di mandato.
    bh = _bh_block(df, confidence)
    ciclo = _ciclo_block(df)
    riduzione_dd = None
    if maxdd_pct is not None and bh["bh_max_dd_pct"]:
        riduzione_dd = 1.0 - abs(maxdd_pct) / abs(bh["bh_max_dd_pct"])

    return {
        "periodo_inizio": str(df.index[0].date()),
        "periodo_fine": str(df.index[-1].date()),
        "mesi": mesi,
        "periodi_anno": _f(ppa),
        "anni": _f(mesi / ppa),

        # Denaro
        "valore_finale": _f(valore_fin),
        "versamenti_totali": _f(versamenti),
        "pnl_netto": _f(pnl),
        "roi_su_versamenti": _f(pnl / versamenti) if versamenti else None,
        "capitale_medio_impiegato": _f(capitale_impiegato.mean()),
        "capitale_max_impiegato": _f(capitale_impiegato.max()),
        **_capitale_al_lavoro(df, capitale_impiegato, pnl),
        "cassa_finale": _f(df["cassa"].iloc[-1]),
        "finanziamento_massimo": _f(-min(0.0, float(df["cassa"].min()))),
        "mesi_a_debito": int((df["cassa"] < -1e-9).sum()),
        "interessi_netti": _f(df["interessi"].sum()) if "interessi" in df.columns else None,

        # Rendimento: semplice, annuo, sul capitale davvero investito
        **rend,
        "rendimento_su_drawdown": _f(rend_su_dd),

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
        # Decomposizione esatta: quanto viene dal movimento delle quote e quanto
        # dalle opzioni. Le tre voci sommate danno l'utile netto.
        "contributo_prezzo": _f(
            pnl - float(df["netto_opzione"].sum())
            - (float(df["interessi"].sum()) if "interessi" in df.columns else 0.0)),
        "premio_pct_medio": _f(df.loc[df["premio_pct"] > 0, "premio_pct"].mean()),
        "mesi_call_assegnata": int((df["intrinseco_pagato"] > 0).sum()),
        "mesi_con_call": (int(df["call_venduta"].sum()) if "call_venduta" in df.columns
                          else None),
        "quota_periodi_con_call": (_f(df["call_venduta"].mean())
                                   if "call_venduta" in df.columns else None),
        "btd_numero": int((df["btd_importo"] > 0).sum()),
        "btd_totale": _f(df["btd_importo"].sum()),
        "btd_medio": _f(df.loc[df["btd_importo"] > 0, "btd_importo"].mean()),
        **_btd_tetto(df),
        **_reinvestimento(df),

        # Benchmark a parita' di flussi
        "bh_stessi_flussi_finale": _f(df["bh_stessi_flussi"].iloc[-1]),
        "bh_stessi_flussi_pnl": _f(df["bh_stessi_flussi"].iloc[-1] - versamenti),
        **bh,
        **ciclo,

        # Confronto diretto con il benchmark
        "riduzione_dd_vs_bh": _f(riduzione_dd),
        "extra_pnl_vs_bh": _f(pnl - (df["bh_stessi_flussi"].iloc[-1] - versamenti)),
        "extra_pnl_vs_ciclo": _f(pnl - ciclo["ciclo_pnl"]) if ciclo["ciclo_pnl"] is not None else None,
        "riduzione_dd_vs_ciclo": (
            _f(1.0 - abs(maxdd_pct) / abs(ciclo["ciclo_max_dd_pct"]))
            if (maxdd_pct is not None and ciclo["ciclo_max_dd_pct"]) else None),
    }


def _ciclo_block(df: pd.DataFrame) -> Dict[str, Any]:
    """Solo sottostante con lo stesso ciclo annuale: nessuna opzione, nessun BTD."""
    vuoto = {"ciclo_valore_finale": None, "ciclo_pnl": None, "ciclo_max_dd_pct": None}
    if "ciclo_annuale_dd" not in df.columns or df["ciclo_annuale_dd"].isna().all():
        return vuoto
    return {
        "ciclo_valore_finale": _f(df["ciclo_annuale"].iloc[-1]),
        "ciclo_pnl": _f(df["ciclo_annuale_pnl"].iloc[-1]),
        "ciclo_max_dd_pct": _f(df["ciclo_annuale_dd"].min()),
    }


def _capitale_al_lavoro(df: pd.DataFrame, impiegato: pd.Series,
                        pnl: float) -> Dict[str, Any]:
    """Quanto del conto e' davvero investito e quanto resta fermo in cassa.

    Il reset annuale reimpiega solo il capitale deciso e lascia in cassa i
    profitti accumulati: dopo qualche anno una fetta grossa del conto sta ferma.
    Per questo il rendimento si misura sul capitale investito, non sul conto.
    """
    valore = df["valore_portafoglio"].replace(0, np.nan)
    quota = (impiegato / valore).replace([np.inf, -np.inf], np.nan)
    return {
        "quota_conto_investita": _f(quota.mean()),
        "quota_conto_investita_finale": _f(quota.iloc[-1]),
        "cassa_media": _f(df["cassa"].mean()),
    }


def _reinvestimento(df: pd.DataFrame) -> Dict[str, Any]:
    """Quanto del risultato delle opzioni e' tornato al lavoro, e dopo quanto.

    L'attesa media si ricava dalla legge di Little: la somma dei premi fermi a
    fine di ogni periodo, divisa per i premi incassati, da' il numero medio di
    periodi che un dollaro di premio passa nel salvadanaio prima di rientrare.

    Il denominatore sono i premi LORDI, perche' e' quello che entra nel
    salvadanaio: l'intrinseco viene pagato dopo e puo' mandarlo a debito.
    """
    if "reinvestito" not in df.columns:
        return {}
    reinvestito = float(df["reinvestito"].sum())
    fuori = {
        "premi_reinvestiti_totali": _f(reinvestito),
        "reinvestimenti_numero": int((df["reinvestito"] > 1e-9).sum()),
    }
    if "premi_pendenti" not in df.columns:
        return fuori

    fermi = df["premi_pendenti"].clip(lower=0.0)
    maturato = float(df["premio"].sum()) if "premio" in df.columns else 0.0
    # Quello che a dicembre e' ancora nel salvadanaio non viene mai reinvestito:
    # la liquidazione di fine anno se lo porta via insieme a tutto il resto.
    mai = float(fermi.groupby(df["anno"]).last().sum()) if "anno" in df.columns else 0.0
    fuori.update({
        "premi_in_attesa_medi": _f(fermi.mean()),
        "premi_in_attesa_max": _f(fermi.max()),
        "premi_mai_reinvestiti": _f(mai),
        "attesa_media_periodi": _f(fermi.sum() / maturato) if maturato > 0 else None,
        "quota_premi_reinvestiti": (_f(reinvestito / maturato) if maturato > 0 else None),
        # Quanto in basso e' andato il conto delle opzioni: e' il prezzo di aver
        # speso il premio prima di sapere quanto sarebbe costato il riacquisto.
        "premi_saldo_minimo": (_f(min(0.0, float(df["cassa_opzioni"].min())))
                               if "cassa_opzioni" in df.columns else None),
    })
    return fuori


def _btd_tetto(df: pd.DataFrame) -> Dict[str, Any]:
    """Quanto il tetto annuo ha vincolato gli acquisti sui cali.

    Quando il tetto e' il vincolo che decide, alzare il boost non fa comprare di
    piu': fa solo esaurire il budget prima, su cali piu' superficiali, e lascia
    scoperti quelli profondi che arrivano dopo. E' un effetto controintuitivo che
    resta invisibile se non lo si misura.
    """
    if "btd_tagliato_dal_tetto" not in df.columns:
        return {}
    acquisti = df[df["btd_importo"] > 0]
    quote = ((acquisti["btd_importo"] / acquisti["btd_prezzo"]).sum()
             if not acquisti.empty else 0.0)
    residui = df.groupby("anno")["btd_residuo_anno"].last()
    anni_pieni = int((residui.fillna(np.inf) < 1.0).sum())
    saltati = df[df["btd_saltato_dal_tetto"].astype(bool)]
    cali_persi = df["rendimento_mese"].shift(1).reindex(saltati.index)
    return {
        "btd_prezzo_medio": _f(acquisti["btd_importo"].sum() / quote) if quote else None,
        "btd_quote_comprate": _f(quote),
        "btd_tagliato_dal_tetto": _f(df["btd_tagliato_dal_tetto"].sum()),
        "btd_segnali_saltati": int(df["btd_saltato_dal_tetto"].astype(bool).sum()),
        "btd_calo_peggiore_saltato": _f(cali_persi.min()) if len(cali_persi) else None,
        "anni_con_tetto_esaurito": anni_pieni,
        "anni_totali": int(df["anno"].nunique()),
    }


def _bh_block(df: pd.DataFrame, confidence: float) -> Dict[str, Any]:
    """Metriche del Buy & Hold che riceve gli stessi versamenti della strategia."""
    if "bh_twr_mese" not in df.columns:
        return {"bh_max_dd_pct": None, "bh_max_dd_valore": None, "bh_var_mensile": None}
    rb = df["bh_twr_mese"]
    return {
        "bh_max_dd_pct": _f(df["bh_dd_twr_pct"].min()),
        "bh_max_dd_valore": _f(df["bh_dd_valore"].min()),
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
        "rendimento_medio", "rendimento_mediano", "rendimento_volatilita",
        "rendimento_su_rischio", "rendimento_su_drawdown",
        "anni_positivi", "anni_totali", "miglior_anno", "peggior_anno",
        "capitale_investito_medio",
        "max_dd_giornaliero_pct", "max_dd_intraday_pct", "dd_nascosto_dal_periodo",
        "ciclo_max_dd_giornaliero_pct", "riduzione_dd_giornaliera_vs_ciclo",
        "dd_giornaliero_durata_max",
        "max_dd_pct", "max_dd_valore", "dd_durata_max_mesi",
        "var_giornaliero", "peggior_giorno",
        "var_mensile", "cvar_mensile", "hit_rate", "miglior_mese", "peggior_mese",
        "contributo_prezzo", "premi_totali", "intrinseco_totale", "netto_opzioni",
        "interessi_netti", "premio_pct_medio",
        "mesi_con_call", "quota_periodi_con_call",
        "mesi_call_assegnata", "btd_numero", "btd_totale",
        "premi_reinvestiti_totali", "reinvestimenti_numero", "attesa_media_periodi",
        "premi_in_attesa_medi", "premi_mai_reinvestiti", "quota_premi_reinvestiti",
        "premi_saldo_minimo",
        "capitale_medio_impiegato", "quota_conto_investita",
        "cassa_media",
        "finanziamento_massimo", "mesi_a_debito",
        "btd_prezzo_medio", "btd_quote_comprate", "btd_tagliato_dal_tetto",
        "btd_segnali_saltati", "btd_calo_peggiore_saltato", "anni_con_tetto_esaurito",
        "ciclo_rendimento_medio", "ciclo_rendimento_volatilita",
        "ciclo_rendimento_su_rischio", "extra_rendimento_vs_ciclo",
        "ciclo_pnl", "ciclo_max_dd_pct", "extra_pnl_vs_ciclo", "riduzione_dd_vs_ciclo",
        "bh_max_dd_pct", "bh_stessi_flussi_pnl", "riduzione_dd_vs_bh", "extra_pnl_vs_bh",
    ]
    df = pd.DataFrame(righe)
    return df.reindex([k for k in ordine if k in df.index])


ETICHETTE = {
    "valore_finale": "Valore finale conto",
    "versamenti_totali": "Capitale versato (totale)",
    "pnl_netto": "Utile netto dei versamenti",
    "roi_su_versamenti": "ROI sul capitale versato",
    "max_dd_giornaliero_pct": "Max drawdown VERO (%) — valorizzato ogni giorno",
    "max_dd_intraday_pct": "Peggio visto in giornata (%) — sui minimi",
    "dd_nascosto_dal_periodo": "Drawdown che la chiusura di periodo nascondeva",
    "ciclo_max_dd_giornaliero_pct": "Max drawdown vero del solo sottostante (%)",
    "riduzione_dd_giornaliera_vs_ciclo": "Riduzione del drawdown VERO vs solo sottostante",
    "dd_giornaliero_durata_max": "Drawdown piu lungo (giorni di borsa)",
    "dd_giornaliero_durata_media": "Durata media dei drawdown (giorni di borsa)",
    "max_dd_giornaliero_valore": "Max drawdown giornaliero in valuta",
    "var_giornaliero": "VaR giornaliero",
    "peggior_giorno": "Peggior giorno",
    "miglior_giorno": "Miglior giorno",
    "giorni": "Giorni di borsa valorizzati",
    "riconciliazione_scarto": "Scarto fra serie giornaliera e serie di periodo",
    "max_dd_pct": "Max drawdown alla chiusura di periodo (%)",
    "max_dd_valore": "Max drawdown in valuta — dipende dal capitale impiegato",
    "dd_durata_max_mesi": "Drawdown piu lungo (mesi)",
    "var_mensile": "VaR mensile",
    "cvar_mensile": "CVaR mensile",
    "hit_rate": "Mesi positivi",
    "miglior_mese": "Miglior mese",
    "peggior_mese": "Peggior mese",
    "interessi_netti": "Interessi netti su cassa e debito",
    "premi_totali": "Premi incassati",
    "intrinseco_totale": "Intrinseco pagato sulle call",
    "netto_opzioni": "Risultato netto opzioni (premi meno intrinseco)",
    "contributo_prezzo": "Contributo del movimento delle quote",
    "premio_pct_medio": "Premio medio (% dello spot)",
    "mesi_call_assegnata": "Mesi con call in-the-money",
    "mesi_con_call": "Mesi in cui la call e stata venduta",
    "quota_periodi_con_call": "Quota dei mesi con la call venduta",
    "btd_numero": "Numero di acquisti BTD",
    "premi_reinvestiti_totali": "Premi rimessi al lavoro",
    "reinvestimenti_numero": "Numero di reinvestimenti",
    "premi_in_attesa_medi": "Premi fermi in attesa, in media",
    "premi_in_attesa_max": "Premi fermi in attesa, al massimo",
    "premi_mai_reinvestiti": "Premi liquidati a dicembre senza essere reinvestiti",
    "attesa_media_periodi": "Attesa media prima del reinvestimento (periodi)",
    "quota_premi_reinvestiti": "Quota dei premi incassati tornata al lavoro",
    "premi_saldo_minimo": "Saldo piu basso toccato dal conto delle opzioni",
    "btd_totale": "Capitale investito in BTD",
    "capitale_medio_impiegato": "Capitale medio impiegato",
    "quota_conto_investita": "Quota media del conto investita",
    "quota_conto_investita_finale": "Quota del conto investita a fine periodo",
    "cassa_media": "Cassa media ferma sul conto",
    "btd_prezzo_medio": "Prezzo medio pagato nei BTD",
    "btd_quote_comprate": "Quote comprate coi BTD",
    "btd_tagliato_dal_tetto": "Acquisti BTD tagliati dal tetto annuo",
    "btd_segnali_saltati": "Segnali BTD saltati per tetto esaurito",
    "btd_calo_peggiore_saltato": "Il calo piu profondo lasciato scoperto",
    "anni_con_tetto_esaurito": "Anni in cui il tetto si e esaurito",
    "finanziamento_massimo": "Massimo saldo a debito",
    "mesi_a_debito": "Mesi con saldo a debito",
    "bh_stessi_flussi_pnl": "Utile del B&H a parita di flussi",
    "ciclo_pnl": "Utile del solo sottostante, stesso ciclo annuale",
    "ciclo_rendimento_medio": "Rendimento medio annuo del solo sottostante",
    "ciclo_rendimento_volatilita": "Oscillazione dei rendimenti del solo sottostante",
    "ciclo_rendimento_su_rischio": "Rendimento diviso oscillazione del solo sottostante",
    "extra_rendimento_vs_ciclo": "Rendimento in piu rispetto al solo sottostante",
    "ciclo_max_dd_pct": "Max drawdown del solo sottostante (%)",
    "extra_pnl_vs_ciclo": "Utile in piu rispetto al solo sottostante",
    "riduzione_dd_vs_ciclo": "Riduzione del drawdown % vs solo sottostante",
    "bh_max_dd_pct": "Max drawdown del B&H (%)",
    "riduzione_dd_vs_bh": "Riduzione del drawdown vs B&H",
    "extra_pnl_vs_bh": "Utile in piu rispetto al B&H",
}

def etichette(cadenza: str = "mensile") -> Dict[str, str]:
    """Le etichette adattate alla cadenza scelta.

    Le chiavi delle metriche restano quelle storiche (`miglior_mese`,
    `mesi_a_debito`): a cambiare e' solo la parola che legge l'utente, che sulla
    cadenza settimanale diventa "settimana".
    """
    return adatta_dizionario(ETICHETTE, cadenza)


FORMATI = {
    "roi_su_versamenti": "pct", "max_dd_pct": "pct", "var_mensile": "pct",
    "max_dd_giornaliero_pct": "pct", "max_dd_intraday_pct": "pct",
    "dd_nascosto_dal_periodo": "pct", "ciclo_max_dd_giornaliero_pct": "pct",
    "riduzione_dd_giornaliera_vs_ciclo": "pct",
    "var_giornaliero": "pct", "peggior_giorno": "pct", "miglior_giorno": "pct",
    "riconciliazione_scarto": "pct",
    "dd_giornaliero_durata_max": "int", "giorni": "int",
    "reinvestimenti_numero": "int", "attesa_media_periodi": "num",
    "mesi_con_call": "int", "quota_periodi_con_call": "pct",
    "quota_premi_reinvestiti": "pct",
    "rendimento_medio": "pct", "rendimento_mediano": "pct",
    "rendimento_volatilita": "pct", "miglior_anno": "pct", "peggior_anno": "pct",
    "rendimento_su_rischio": "num", "rendimento_su_drawdown": "num",
    "anni_positivi": "int", "anni_totali": "int",
    "cvar_mensile": "pct", "hit_rate": "pct", "miglior_mese": "pct",
    "peggior_mese": "pct", "premio_pct_medio": "pct",
    "bh_max_dd_pct": "pct", "riduzione_dd_vs_bh": "pct",
    "ciclo_rendimento_medio": "pct", "ciclo_rendimento_volatilita": "pct",
    "extra_rendimento_vs_ciclo": "pct", "ciclo_rendimento_su_rischio": "num",
    "ciclo_max_dd_pct": "pct", "riduzione_dd_vs_ciclo": "pct",
    "quota_conto_investita": "pct", "quota_conto_investita_finale": "pct",
    "dd_durata_max_mesi": "int", "mesi_call_assegnata": "int", "btd_numero": "int",
    "mesi_a_debito": "int", "btd_segnali_saltati": "int",
    "anni_con_tetto_esaurito": "int", "btd_quote_comprate": "num",
    "btd_calo_peggiore_saltato": "pct",
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
