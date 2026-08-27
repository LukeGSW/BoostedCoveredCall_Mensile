"""Motore di backtest della Boosted Covered Call mensile.

Modello (una riga = un mese di calendario):

  * A inizio anno si impiega un CAPITALE FISSO (`capitale_iniziale`) comprando
    quote all'open di gennaio: sono le quote COPERTE dalla covered call e
    restano costanti per tutto l'anno.
  * Ogni mese si vende una call a delta ~0.50 con scadenza a fine mese e si
    incassa un premio pari a una percentuale del valore corrente del
    sottostante (quindi diverso ogni mese: N * open_mese * premio_pct).
  * A scadenza, se la call e' in-the-money la si riacquista al valore
    intrinseco pagando in contanti: le quote restano le stesse. E' qui che si
    paga il costo del cap sull'upside, e il costo si accumula davvero.
  * Quando il sottostante ha un mese negativo scatta il Buy-The-Dip: si
    acquista |rendimento del mese precedente| * capitale_iniziale, piu' il
    BOOST, una percentuale fissa del capitale iniziale che si aggiunge a ogni
    acquisto. Le due quote sono tracciate separatamente in `btd_quota_calo` e
    `btd_quota_boost`. Non sono coperte dalla call, sono soggette allo stesso
    tetto annuo e vengono liquidate a fine anno come tutto il resto.
  * Il CAPITALE ADDIZIONALE ANNUALE e' un'altra cosa: entra una volta sola
    all'apertura di gennaio insieme al capitale fisso, compra sottostante e
    resta li' per tutto l'anno. La call venduta NON lo cappa, quindi si tiene
    tutto il rialzo. Sul capitale fisso si incassa il premio e si paga il cap;
    su questo no.
  * A fine anno si liquida tutto e si ricomincia con lo stesso capitale fisso.
    L'eventuale eccedenza resta come cassa non impiegata; se manca capitale,
    si versa la differenza (ed e' un versamento, non un utile).

Contabilita': ogni euro che entra dall'esterno e' tracciato in
`versamenti_cum`, cosi' `pnl_netto = valore_portafoglio - versamenti_cum` e'
il vero risultato della strategia, e i rendimenti sono time-weighted.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .pricing import PremiumModel, bs_call_price, strike_for_delta
from . import vol as volmod


# ----------------------------------------------------------------------------
# Configurazione
# ----------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    ticker: str = "BTC-USD.CC"
    start_date: str = "2018-01-01"
    end_date: Optional[str] = None

    # Capitale
    capitale_iniziale: float = 25_000.0        # base coperta, fissata a inizio anno
    capitale_addizionale: float = 0.0          # capitale extra annuo, non coperto

    # Buy-The-Dip
    # BOOST: percentuale del capitale iniziale che si aggiunge a OGNI acquisto
    # BTD, oltre alla quota legata all'entita' del calo. Ne eredita tutte le
    # caratteristiche: stesso momento e stesso prezzo di acquisto, quote non
    # coperte dalla call, stesso tetto annuo, liquidazione a fine anno.
    # Da non confondere con `capitale_addizionale`, che entra una volta sola.
    boost_pct: float = 0.05
    btd_cap_annuo_pct: float = 1.00            # tetto annuo ai BTD, % del capitale iniziale
    btd_dd_weekly_limit: float = -0.90         # blocca il BTD se il DD weekly e' sotto
    btd_execution: str = "open"                # "open" (mese del segnale) | "close" (legacy)

    # Opzione
    strike_mode: str = "delta"                 # "delta" (0.50) | "atm_spot"
    applica_cap: bool = True                   # riacquisto a intrinseco a scadenza

    # Stima della volatilita'.
    # I default vengono da uno sweep su 1.581 premi reali di call ATM mensili
    # (SPX, SPY, AAPL, PG, WMT, TSLA, 2000-2025): su tutti e sei i sottostanti
    # ha vinto la finestra corta di circa sei mesi, smorzata a meta' verso la
    # media di lungo periodo. La finestra corta cattura il raggruppamento della
    # volatilita', lo smorzamento ne toglie l'errore di stima.
    vol_model: str = "yang_zhang"
    vol_window: int = 126                      # ~6 mesi
    vol_long_window: int = 504                 # ~2 anni
    vol_blend: float = 0.60
    ewma_lambda: float = 0.94

    # Mercato
    premium_model: PremiumModel = field(default_factory=PremiumModel)
    idle_cash_rate: float = 0.0                # remunerazione della cassa non impiegata
    debit_cash_rate: float = 0.06              # costo del saldo a debito (vedi sotto)
    var_confidence: float = 0.99

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["premium_model"] = self.premium_model.to_dict()
        return d


VARIANTS = {
    "no_premi":       {"label": "BTD No Premi",             "vende_call": False, "reinveste": False},
    "premi_cash":     {"label": "BTD + Premi (Cash)",       "vende_call": True,  "reinveste": False},
    "premi_reinvest": {"label": "BTD + Premi (Reinvest)",   "vende_call": True,  "reinveste": True},
}


# ----------------------------------------------------------------------------
# Helper temporali
# ----------------------------------------------------------------------------
def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def _month_days(ts: pd.Timestamp) -> int:
    return int(ts.days_in_month)


# ----------------------------------------------------------------------------
# Preparazione dati
# ----------------------------------------------------------------------------
def prepare_market_data(
    monthly: pd.DataFrame,
    weekly: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    cfg: BacktestConfig,
) -> Dict[str, Any]:
    """Normalizza i dati e calcola segnali e volatilita' realizzata."""
    warnings: List[str] = []

    m = monthly.copy()
    m = m[~m.index.duplicated(keep="last")].sort_index()
    m["rendimento_mese"] = m["Close"].pct_change()
    # Il segnale del mese i guarda il mese i-1: nessun look-ahead.
    m["segnale_btd"] = (m["rendimento_mese"] < 0).shift(1).fillna(False).astype(bool)

    dd_weekly = pd.Series(dtype=float)
    if weekly is not None and not weekly.empty and "Close" in weekly.columns:
        w = weekly.copy().sort_index()
        cm = w["Close"].cummax()
        dd_weekly = ((w["Close"] - cm) / cm.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dd_weekly.name = "dd_weekly"
    else:
        warnings.append("Dati settimanali non disponibili: il filtro sul drawdown weekly e' disattivato.")

    if daily is not None and not daily.empty and len(daily) > cfg.vol_window:
        vol_series = volmod.realized_vol(
            daily, model=cfg.vol_model, window=cfg.vol_window,
            long_window=cfg.vol_long_window, blend=cfg.vol_blend,
            ewma_lambda=cfg.ewma_lambda,
        )
        vol_source = f"giornaliero / {volmod.VOL_MODELS.get(cfg.vol_model, cfg.vol_model)}"
    else:
        vol_series = volmod.vol_from_monthly(m)
        vol_source = "mensile (fallback)"
        warnings.append(
            "Dati giornalieri non disponibili: la volatilita' e' stimata dai rendimenti "
            "mensili ed e' molto meno precisa."
        )

    return {"monthly": m, "dd_weekly": dd_weekly, "vol": vol_series,
            "vol_source": vol_source, "warnings": warnings}


# ----------------------------------------------------------------------------
# Simulazione di una variante
# ----------------------------------------------------------------------------
def run_variant(market: Dict[str, Any], cfg: BacktestConfig, variant: str) -> Dict[str, Any]:
    spec = VARIANTS[variant]
    vende_call = bool(spec["vende_call"])
    reinveste = bool(spec["reinveste"])

    m: pd.DataFrame = market["monthly"]
    dd_weekly: pd.Series = market["dd_weekly"]
    vol_series: pd.Series = market["vol"]

    cap0 = float(cfg.capitale_iniziale)
    cap_add = float(cfg.capitale_addizionale)
    capitale_annuo = cap0 + cap_add
    tetto_btd = cap0 * float(cfg.btd_cap_annuo_pct)
    boost_per_acquisto = cap0 * float(cfg.boost_pct)
    pm = cfg.premium_model
    idle_m = float(cfg.idle_cash_rate) / 12.0
    debito_m = float(cfg.debit_cash_rate) / 12.0

    # Stato del conto
    cassa = 0.0
    quote_coperte = 0.0
    quote_extra = 0.0
    versamenti = 0.0
    btd_usato_anno = 0.0
    anno_corrente: Optional[int] = None

    # Benchmark con gli STESSI flussi di cassa (compra e tiene, mai liquida)
    bh_quote = 0.0

    rows: List[Dict[str, Any]] = []

    for i in range(len(m)):
        bar = m.iloc[i]
        data = bar.name
        O, C = float(bar["Open"]), float(bar["Close"])
        if not (np.isfinite(O) and np.isfinite(C) and O > 0 and C > 0):
            continue

        ms = _month_start(data)
        versato_mese = 0.0
        liquidazione = 0.0
        nuovo_anno = (anno_corrente is None) or (data.year != anno_corrente)

        # ---------------- Reset annuale ----------------
        if nuovo_anno:
            if anno_corrente is not None:
                # liquida al close del mese precedente
                prezzo_liq = float(m.iloc[i - 1]["Close"])
                liquidazione = (quote_coperte + quote_extra) * prezzo_liq
                cassa += liquidazione
                quote_coperte = quote_extra = 0.0
            manca = capitale_annuo - cassa
            if manca > 0:
                versamenti += manca
                versato_mese += manca
                cassa += manca
                bh_quote += manca / O            # stesso flusso sul benchmark
            cassa -= capitale_annuo
            quote_coperte = cap0 / O
            quote_extra = cap_add / O
            btd_usato_anno = 0.0
            anno_corrente = data.year

        # Interessi sulla liquidita'. Il saldo puo' andare a debito quando il
        # riacquisto della call a intrinseco supera la cassa disponibile: e' un
        # finanziamento garantito dalle azioni in portafoglio, e come tale costa.
        interessi = 0.0
        if cassa > 0 and idle_m:
            interessi = cassa * idle_m
        elif cassa < 0 and debito_m:
            interessi = cassa * debito_m
        cassa += interessi

        # ---------------- Premio della call ----------------
        sigma = volmod.sigma_at(vol_series, ms, fallback=np.nan)
        T = _month_days(data) / 365.0
        strike = np.nan
        premio_pct = 0.0
        premio = 0.0
        vrp_applicato = np.nan

        if vende_call and quote_coperte > 0 and np.isfinite(sigma) and sigma > 0:
            q = pm.quote(O, sigma, T)
            strike, premio_pct, vrp_applicato = q["strike"], q["premium_pct"], q["vrp"]
            if cfg.strike_mode == "atm_spot":
                strike = O
                premio_pct = bs_call_price(O, O, T, q["sigma"], pm.r, pm.q) / O
            # il premio e' sempre una % del valore CORRENTE del sottostante
            premio = quote_coperte * O * premio_pct
            cassa += premio

        # ---------------- Buy-The-Dip ----------------
        segnale = bool(bar["segnale_btd"])
        rend_trigger = float(m["rendimento_mese"].iloc[i - 1]) if i > 0 else np.nan
        # decisione presa alla fine del mese precedente: nessun look-ahead
        ts_decisione = ms - pd.Timedelta(days=1)
        dd_w = np.nan
        if not dd_weekly.empty:
            try:
                v = dd_weekly.asof(ts_decisione)
                dd_w = float(v) if pd.notna(v) else np.nan
            except Exception:
                dd_w = np.nan
        bloccato = bool(np.isfinite(dd_w) and dd_w < float(cfg.btd_dd_weekly_limit))

        btd_importo = 0.0
        quota_calo = 0.0
        quota_boost = 0.0
        prezzo_btd = O if cfg.btd_execution == "open" else C
        if segnale and not bloccato and np.isfinite(rend_trigger) and rend_trigger < 0:
            # quota legata all'entita' del calo del mese precedente
            quota_calo = abs(rend_trigger) * cap0
            # boost: si aggiunge per intero a ogni acquisto
            quota_boost = boost_per_acquisto

            potenziale = quota_calo + quota_boost
            btd_importo = max(0.0, min(potenziale, tetto_btd - btd_usato_anno))
            if btd_importo > 1e-9:
                # se il tetto annuo taglia l'acquisto, taglia entrambe le quote
                if potenziale > 0:
                    fattore = btd_importo / potenziale
                    quota_calo *= fattore
                    quota_boost *= fattore
                if cassa < btd_importo:                     # serve denaro fresco
                    manca = btd_importo - cassa
                    versamenti += manca
                    versato_mese += manca
                    cassa += manca
                    bh_quote += manca / prezzo_btd
                cassa -= btd_importo
                quote_extra += btd_importo / prezzo_btd
                btd_usato_anno += btd_importo
            else:
                btd_importo = quota_calo = quota_boost = 0.0

        # ---------------- Scadenza della call ----------------
        intrinseco = 0.0
        if vende_call and cfg.applica_cap and quote_coperte > 0 and np.isfinite(strike):
            intrinseco = quote_coperte * max(0.0, C - strike)
            cassa -= intrinseco

        netto_opzione = premio - intrinseco

        # ---------------- Reinvestimento ----------------
        reinvestito = 0.0
        if reinveste and netto_opzione > 0 and C > 0:
            reinvestito = min(netto_opzione, max(0.0, cassa))
            if reinvestito > 0:
                cassa -= reinvestito
                quote_extra += reinvestito / C

        # ---------------- Mark to market ----------------
        valore = (quote_coperte + quote_extra) * C + cassa
        rows.append({
            "data": data, "anno": int(data.year),
            "open": O, "close": C,
            "rendimento_mese": float(bar["rendimento_mese"]) if pd.notna(bar["rendimento_mese"]) else np.nan,
            "segnale_btd": segnale, "btd_bloccato": bloccato, "dd_weekly": dd_w,
            "btd_importo": btd_importo,
            "btd_quota_calo": quota_calo, "btd_quota_boost": quota_boost,
            "btd_residuo_anno": max(0.0, tetto_btd - btd_usato_anno),
            "btd_prezzo": prezzo_btd if btd_importo > 0 else np.nan,
            "sigma_stimata": float(sigma) if np.isfinite(sigma) else np.nan,
            "sigma_implicita": float(pm.implied_sigma(sigma)) if np.isfinite(sigma) else np.nan,
            "vrp_applicato": float(vrp_applicato) if np.isfinite(vrp_applicato) else np.nan,
            "strike": float(strike) if np.isfinite(strike) else np.nan,
            "premio_pct": premio_pct, "premio": premio,
            "intrinseco_pagato": intrinseco, "netto_opzione": netto_opzione,
            "reinvestito": reinvestito,
            "quote_coperte": quote_coperte, "quote_extra": quote_extra,
            "cassa": cassa, "interessi": interessi, "liquidazione": liquidazione,
            "valore_portafoglio": valore,
            "versamento_mese": versato_mese, "versamenti_cum": versamenti,
            "pnl_netto": valore - versamenti,
            "bh_stessi_flussi": bh_quote * C,
        })

    df = pd.DataFrame(rows).set_index("data") if rows else pd.DataFrame()
    if df.empty:
        return {"variant": variant, "label": spec["label"], "monthly": df,
                "yearly": pd.DataFrame(), "metrics": {}}

    # Rendimento time-weighted: i versamenti del mese entrano a inizio periodo
    base = df["valore_portafoglio"].shift(1).fillna(0.0) + df["versamento_mese"]
    df["twr_mese"] = np.where(base > 0, df["valore_portafoglio"] / base - 1.0, 0.0)
    df["indice_twr"] = (1.0 + df["twr_mese"]).cumprod()
    df["dd_valore"] = (df["valore_portafoglio"] - df["valore_portafoglio"].cummax()).clip(upper=0.0)
    df["dd_twr_pct"] = (df["indice_twr"] / df["indice_twr"].cummax() - 1.0).clip(upper=0.0)
    df["pnl_dd"] = (df["pnl_netto"] - df["pnl_netto"].cummax()).clip(upper=0.0)

    # Stesse metriche per il Buy & Hold che riceve i medesimi versamenti:
    # e' il termine di paragone corretto per dire se la strategia riduce il
    # drawdown e se la variante reinvestita tiene il passo sui rendimenti.
    base_bh = df["bh_stessi_flussi"].shift(1).fillna(0.0) + df["versamento_mese"]
    df["bh_twr_mese"] = np.where(base_bh > 0, df["bh_stessi_flussi"] / base_bh - 1.0, 0.0)
    df["bh_indice_twr"] = (1.0 + df["bh_twr_mese"]).cumprod()
    df["bh_dd_twr_pct"] = (df["bh_indice_twr"] / df["bh_indice_twr"].cummax() - 1.0).clip(upper=0.0)
    df["bh_dd_valore"] = (df["bh_stessi_flussi"] - df["bh_stessi_flussi"].cummax()).clip(upper=0.0)
    df["bh_pnl_netto"] = df["bh_stessi_flussi"] - df["versamenti_cum"]

    return {"variant": variant, "label": spec["label"], "monthly": df,
            "yearly": _yearly_table(df), "metrics": {}}


# ----------------------------------------------------------------------------
# Tabella annuale
# ----------------------------------------------------------------------------
def _yearly_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = []
    valore_fine_prec = 0.0
    for anno, g in df.groupby("anno", sort=True):
        versato = float(g["versamento_mese"].sum())
        valore_fine = float(g["valore_portafoglio"].iloc[-1])
        risultato = valore_fine - valore_fine_prec - versato
        capitale_medio = float((g["quote_coperte"] + g["quote_extra"]).mul(g["close"]).mean())
        twr_anno = float((1.0 + g["twr_mese"]).prod() - 1.0)
        out.append({
            "anno": int(anno),
            "mesi": int(len(g)),
            "rendimento_sottostante": float(g["close"].iloc[-1] / g["open"].iloc[0] - 1.0),
            "premi_incassati": float(g["premio"].sum()),
            "intrinseco_pagato": float(g["intrinseco_pagato"].sum()),
            "netto_opzioni": float(g["netto_opzione"].sum()),
            "btd_numero": int((g["btd_importo"] > 0).sum()),
            "btd_investito": float(g["btd_importo"].sum()),
            "btd_da_calo": float(g["btd_quota_calo"].sum()),
            "btd_da_boost": float(g["btd_quota_boost"].sum()),
            "versamenti": versato,
            "capitale_medio_impiegato": capitale_medio,
            "valore_fine_anno": valore_fine,
            "risultato_anno": risultato,
            "twr_anno": twr_anno,
        })
        valore_fine_prec = valore_fine
    return pd.DataFrame(out).set_index("anno")


# ----------------------------------------------------------------------------
# Monitoraggio a mercato
# ----------------------------------------------------------------------------
def dettaglio_anno(risultato: Dict[str, Any], variante: str = "premi_cash",
                   anno: Optional[int] = None) -> Dict[str, Any]:
    """Fotografia mese per mese di un singolo anno, per seguire la strategia dal vivo.

    Senza `anno` restituisce l'ultimo presente nel backtest, cioe' quello in
    corso quando il backtest arriva a oggi.
    """
    res = (risultato.get("varianti") or {}).get(variante)
    if not res or res["monthly"].empty:
        return {}
    df: pd.DataFrame = res["monthly"]
    anno = int(anno if anno is not None else df["anno"].iloc[-1])
    g = df[df["anno"] == anno]
    if g.empty:
        return {}

    cfg = risultato.get("config", {})
    cap0 = float(cfg.get("capitale_iniziale", 0.0))
    tetto = cap0 * float(cfg.get("btd_cap_annuo_pct", 1.0))
    cap_add = float(cfg.get("capitale_addizionale", 0.0))

    btd_usato = float(g["btd_importo"].sum())
    quote_tot = (g["quote_coperte"] + g["quote_extra"]).iloc[-1]
    valore_iniziale = float(g["valore_portafoglio"].iloc[0])
    valore_corrente = float(g["valore_portafoglio"].iloc[-1])

    return {
        "anno": anno,
        "mesi": g,
        "riepilogo": {
            "mesi_trascorsi": int(len(g)),
            "capitale_fisso": cap0,
            "quote_coperte": float(g["quote_coperte"].iloc[-1]),
            "quote_extra": float(g["quote_extra"].iloc[-1]),
            "valore_posizione": float(quote_tot * g["close"].iloc[-1]),
            "cassa": float(g["cassa"].iloc[-1]),
            "valore_conto": valore_corrente,
            "premi_incassati": float(g["premio"].sum()),
            "intrinseco_pagato": float(g["intrinseco_pagato"].sum()),
            "netto_opzioni": float(g["netto_opzione"].sum()),
            "mesi_call_itm": int((g["intrinseco_pagato"] > 0).sum()),
            "btd_numero": int((g["btd_importo"] > 0).sum()),
            "btd_investito": btd_usato,
            "btd_da_calo": float(g["btd_quota_calo"].sum()),
            "btd_da_boost": float(g["btd_quota_boost"].sum()),
            "btd_residuo": max(0.0, tetto - btd_usato),
            "btd_tetto": tetto,
            "boost_per_acquisto": cap0 * float(cfg.get("boost_pct", 0.0)),
            "capitale_addizionale": cap_add,
            "versamenti": float(g["versamento_mese"].sum()),
            "risultato_anno": valore_corrente - valore_iniziale - float(g["versamento_mese"].iloc[1:].sum()),
            "twr_anno": float((1.0 + g["twr_mese"]).prod() - 1.0),
            "dd_valore": float(g["dd_valore"].min()),
            "segnali_bloccati": int((g["btd_bloccato"] & g["segnale_btd"]).sum()),
        },
    }


def piano_prossimo_mese(risultato: Dict[str, Any], variante: str = "premi_cash") -> Dict[str, Any]:
    """Cosa fara' la strategia il mese prossimo, con i dati disponibili oggi.

    Tutte le grandezze sono gia' determinate dalla chiusura dell'ultimo mese:
    il segnale guarda indietro, la volatilita' e' quella nota, e l'unica cosa
    che manca e' il prezzo di apertura del mese, quindi gli importi in quote
    sono indicativi mentre quelli in valuta sono esatti.
    """
    res = (risultato.get("varianti") or {}).get(variante)
    if not res or res["monthly"].empty:
        return {}
    df: pd.DataFrame = res["monthly"]
    cfg = risultato.get("config", {})
    ultima = df.iloc[-1]
    prossimo = (pd.Timestamp(df.index[-1]) + pd.offsets.MonthBegin(1))
    nuovo_anno = int(prossimo.year) != int(ultima["anno"])

    cap0 = float(cfg.get("capitale_iniziale", 0.0))
    cap_add = float(cfg.get("capitale_addizionale", 0.0))
    tetto = cap0 * float(cfg.get("btd_cap_annuo_pct", 1.0))
    boost_per_acquisto = cap0 * float(cfg.get("boost_pct", 0.0))

    corrente = df[df["anno"] == int(ultima["anno"])]
    btd_usato = 0.0 if nuovo_anno else float(corrente["btd_importo"].sum())

    # Segnale: il mese appena chiuso e' stato negativo?
    rend = float(ultima["rendimento_mese"]) if pd.notna(ultima["rendimento_mese"]) else np.nan
    segnale = bool(np.isfinite(rend) and rend < 0)

    dd_weekly = risultato.get("mercato", {}).get("dd_weekly")
    dd = np.nan
    if isinstance(dd_weekly, pd.Series) and not dd_weekly.empty:
        dd = float(dd_weekly.iloc[-1])
    bloccato = bool(np.isfinite(dd) and dd < float(cfg.get("btd_dd_weekly_limit", -0.90)))

    quota_calo = quota_boost = importo = 0.0
    if segnale and not bloccato:
        quota_calo = abs(rend) * cap0
        quota_boost = boost_per_acquisto
        potenziale = quota_calo + quota_boost
        importo = max(0.0, min(potenziale, tetto - btd_usato))
        if potenziale > 0 and importo < potenziale:
            fattore = importo / potenziale
            quota_calo *= fattore
            quota_boost *= fattore

    # Premio stimato sul prossimo mese, allo spot di oggi
    pm_cfg = cfg.get("premium_model", {}) or {}
    pm = PremiumModel(**{k: v for k, v in pm_cfg.items()
                         if k in PremiumModel.__dataclass_fields__})
    sigma = float(ultima["sigma_stimata"]) if pd.notna(ultima["sigma_stimata"]) else np.nan
    spot = float(ultima["close"])
    T = int(prossimo.days_in_month) / 365.0
    quote_coperte = 0.0 if nuovo_anno else float(ultima["quote_coperte"])
    quota_prem: Dict[str, float] = {}
    if np.isfinite(sigma) and sigma > 0:
        quota_prem = pm.quote(spot, sigma, T)

    return {
        "mese": prossimo.strftime("%Y-%m"),
        "reset_annuale": nuovo_anno,
        "capitale_da_impiegare": (cap0 + cap_add) if nuovo_anno else 0.0,
        "prezzo_riferimento": spot,
        "rendimento_ultimo_mese": rend,
        "segnale_btd": segnale,
        "btd_bloccato": bloccato,
        "dd_weekly": dd,
        "btd_importo": importo,
        "btd_quota_calo": quota_calo,
        "btd_quota_boost": quota_boost,
        "btd_usato_anno": btd_usato,
        "btd_residuo_anno": max(0.0, tetto - btd_usato),
        "quote_coperte": quote_coperte,
        "sigma_stimata": sigma if np.isfinite(sigma) else None,
        "strike_indicativo": quota_prem.get("strike"),
        "premio_pct": quota_prem.get("premium_pct"),
        "premio_atteso": (quote_coperte * spot * quota_prem["premium_pct"]
                          if quota_prem and quote_coperte > 0 else 0.0),
        "vrp_applicato": quota_prem.get("vrp"),
    }


# ----------------------------------------------------------------------------
# Orchestrazione
# ----------------------------------------------------------------------------
def run_backtest(
    monthly: pd.DataFrame,
    weekly: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    cfg: BacktestConfig,
) -> Dict[str, Any]:
    """Esegue le tre varianti piu' i benchmark. Non produce grafici."""
    from .metrics import compute_metrics    # import locale: evita cicli

    if monthly is None or monthly.empty or len(monthly) < 2:
        return {"ok": False, "errore": "Serie mensile vuota o troppo corta.",
                "config": cfg.to_dict(), "varianti": {}, "warnings": []}

    market = prepare_market_data(monthly, weekly, daily, cfg)
    m = market["monthly"]

    risultati: Dict[str, Any] = {}
    for name in VARIANTS:
        res = run_variant(market, cfg, name)
        if not res["monthly"].empty:
            res["metrics"] = compute_metrics(res["monthly"], cfg.var_confidence)
        risultati[name] = res

    # Buy & Hold semplice: solo il capitale iniziale, mai piu' toccato
    capitale_annuo = cfg.capitale_iniziale + cfg.capitale_addizionale
    o0 = float(m["Open"].iloc[0])
    bh_semplice = (m["Close"] / o0 * capitale_annuo).rename("bh_semplice") if o0 > 0 else pd.Series(dtype=float)

    return {
        "ok": True,
        "config": cfg.to_dict(),
        "mercato": {
            "prezzi": m[["Open", "Close", "rendimento_mese", "segnale_btd"]],
            "vol": market["vol"],
            "vol_source": market["vol_source"],
            "dd_weekly": market["dd_weekly"],
            "bh_semplice": bh_semplice,
        },
        "varianti": risultati,
        "warnings": market["warnings"],
        # Serie giornaliera grezza: serve a ricalibrare la volatilita' con altri
        # stimatori senza riscaricare i dati. Il prefisso la tiene fuori dall'export.
        "_giornaliero": daily if isinstance(daily, pd.DataFrame) else None,
    }
