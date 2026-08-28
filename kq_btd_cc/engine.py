"""Motore di backtest della Boosted Covered Call, mensile o settimanale.

La cadenza decide solo quanto e' lungo il passo elementare: `cadenza="mensile"`
lavora su barre mensili e vende call a scadenza mensile, `cadenza="settimanale"`
lavora su barre settimanali e vende call a scadenza settimanale. Tutto il resto
del modello e' identico, e in particolare TUTTO CIO CHE E ANNUALE resta invariato:
il capitale si decide a gennaio, si liquida a dicembre, si ricomincia. Cambia il
ritmo, non la struttura.

Modello (una riga = un periodo, mese o settimana):

  * A inizio anno si impiega un CAPITALE FISSO (`capitale_iniziale`) comprando
    quote all'open di gennaio: sono le quote COPERTE dalla covered call e
    restano costanti per tutto l'anno.
  * Ogni periodo si vende una call a delta ~0.50 con scadenza a fine periodo e
    si incassa un premio pari a una percentuale del valore corrente del
    sottostante (quindi diverso ogni volta: N * open_periodo * premio_pct).
    Sulla cadenza settimanale la call dura sette giorni invece di trenta: il
    singolo premio vale circa la meta', ma se ne incassano 52 invece di 12.
  * A scadenza, se la call e' in-the-money la si riacquista al valore
    intrinseco pagando in contanti: le quote restano le stesse. E' qui che si
    paga il costo del cap sull'upside, e il costo si accumula davvero.
  * Quando il sottostante ha un periodo negativo scatta il Buy-The-Dip: si
    acquista |rendimento del periodo precedente| * capitale_iniziale, piu' il
    BOOST, una percentuale fissa del capitale iniziale che si aggiunge a ogni
    acquisto. Le due quote sono tracciate separatamente in `btd_quota_calo` e
    `btd_quota_boost`. Non sono coperte dalla call e vengono liquidate a fine
    anno come tutto il resto. Di default non c'e' un tetto agli acquisti
    dell'anno: se lo si imposta, il budget si esaurisce sui cali superficiali
    di inizio anno e i piu' profondi restano scoperti.
  * Il CAPITALE ADDIZIONALE ANNUALE e' un'altra cosa: entra una volta sola
    all'apertura di gennaio insieme al capitale fisso, compra sottostante e
    resta li' per tutto l'anno. La call venduta NON lo cappa, quindi si tiene
    tutto il rialzo. Sul capitale fisso si incassa il premio e si paga il cap;
    su questo no.
  * A fine anno si liquida tutto e si ricomincia. Con `capitale_modo="fisso"`
    si rimette al lavoro sempre lo stesso importo e l'eccedenza resta in cassa:
    la strategia non compone, e il conto cresce in linea retta perche' il
    capitale che lavora non aumenta mai. Con `capitale_modo="composto"` torna
    al lavoro tutto il conto, mantenendo la proporzione fra parte coperta e
    parte scoperta, e i profitti si capitalizzano.

Contabilita': due grandezze diverse, da non confondere.
  * `versamenti_cum` e' il denaro entrato dall'esterno dall'inizio del backtest.
    NON si azzera a gennaio, perche' e' il metro con cui si misura l'utile:
    `pnl_netto = valore_portafoglio - versamenti_cum`. Azzerarlo farebbe
    ricomparire ogni anno come "guadagno" del denaro che invece hai versato tu.
  * `capitale_impiegato_anno` e' quanto la strategia sta facendo lavorare nel
    ciclo in corso: capitale fisso piu' i Buy-The-Dip accumulati nell'anno.
    Questo SI azzera a ogni gennaio, insieme a tutto il resto.
I rendimenti sono time-weighted, quindi ripuliti dai flussi.

QUANDO SI REINVESTONO I PREMI (solo variante Reinvest). Due modi, scelti da
`reinvesto_modo`:
  * "subito"  ogni periodo, alla chiusura, si comprano quote con il risultato
              netto delle opzioni appena incassato. E' il comportamento storico.
  * "al_btd"  i premi si accumulano in un salvadanaio e ci restano finche' non
              scatta un acquisto sui cali: a quel punto entra tutto insieme, allo
              stesso prezzo del Buy-The-Dip. Si comprano premi arretrati sul
              ribasso invece che al prezzo corrente, qualunque esso sia.

Nel BTD il premio entra LORDO, compreso quello del periodo in corso, che e' stato
accreditato all'apertura poche righe prima. E' quello che succede su un conto
vero: quando vendi la call il premio e' cassa disponibile subito, e se il ribasso
arriva prima della scadenza lo spendi senza aspettare di sapere quanto ti costera'
il riacquisto. L'intrinseco si paga dopo, a scadenza, e si scala dal conto delle
opzioni, che puo' andare a debito e in quel caso paga interessi come qualunque
altro saldo negativo. Il vantaggio e' che nel frattempo quel premio ha comprato
quote e ha lavorato; il costo e' che si e' speso denaro prima di sapere quanto
sarebbe rimasto. Se le scadenze precedenti hanno gia' prosciugato il conto delle
opzioni non si compra a debito: si versa quello che c'e'.

Il salvadanaio si azzera al reset di gennaio insieme a tutto il resto: i premi
incassati a novembre senza piu' un calo davanti restano in cassa e vengono
liquidati, esattamente come nella variante Cash.

VALORIZZAZIONE. Il motore decide e opera sulla griglia del periodo, ma il conto
viene poi rivalutato GIORNO PER GIORNO da `giornaliero.py`, che ricostruisce la
posizione dentro il periodo e segna a mercato anche la call venduta. Serve
perche' un crollo rientrato prima della chiusura della barra non lascerebbe
traccia nella serie di periodo, e il drawdown misurato risulterebbe molto piu'
tenero di quello vero. Le due serie coincidono al centesimo a ogni fine periodo.

Nomi delle colonne: restano quelli storici (`rendimento_mese`, `twr_mese`,
`versamento_mese`) anche in cadenza settimanale, dove vanno letti come "del
periodo". Cambiarli avrebbe rotto export, grafici e file gia' salvati senza
aggiungere nulla: e' la UI a chiamarli col nome giusto.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .cadenza import CADENZE, PERIODI_ANNO
from .giornaliero import metriche_giornaliere, valorizza_giornaliero
from .cadenza import normalizza as CADENZA_VALIDA
from .cadenza import periodi_anno
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

    # Passo elementare del backtest.
    #   "mensile"     barre mensili, call a scadenza mensile: 12 cicli l'anno.
    #   "settimanale" barre settimanali, call a scadenza settimanale: 52 cicli
    #                 l'anno. Piu' premi incassati e molti piu' acquisti sui
    #                 cali, quindi piu' aggressiva. Il ciclo ANNUALE non cambia.
    cadenza: str = "mensile"

    # Capitale
    capitale_iniziale: float = 25_000.0        # base coperta, fissata a inizio anno
    capitale_addizionale: float = 0.0          # capitale extra annuo, non coperto

    # Come si decide, ogni gennaio, quanto rimettere al lavoro.
    #   "fisso"    sempre lo stesso importo: i profitti restano in cassa e la
    #              strategia NON compone. La curva cresce in linea retta perche'
    #              il capitale che lavora non aumenta mai.
    #   "composto" il capitale di gennaio cresce col conto, mantenendo la
    #              proporzione fra parte coperta e parte scoperta: i profitti
    #              tornano al lavoro e il rendimento si capitalizza.
    capitale_modo: str = "fisso"

    # Solo in modalita' composta: quota del conto tenuta liquida a gennaio per
    # finanziare gli acquisti sui cali durante l'anno, espressa in frazione del
    # capitale impiegato. Senza riserva ogni Buy-The-Dip richiederebbe denaro
    # fresco e il capitale versato crescerebbe insieme al conto, il che rende
    # illeggibile qualunque confronto. In modalita' composta gli acquisti sui
    # cali si finanziano solo da qui: se la riserva finisce, si comprano meno.
    riserva_btd_pct: float = 0.75

    # Buy-The-Dip
    # BOOST: percentuale del capitale iniziale che si aggiunge a OGNI acquisto
    # BTD, oltre alla quota legata all'entita' del calo. Ne eredita tutte le
    # caratteristiche: stesso momento e stesso prezzo di acquisto, quote non
    # coperte dalla call, stesso tetto annuo, liquidazione a fine anno.
    # Da non confondere con `capitale_addizionale`, che entra una volta sola.
    boost_pct: float = 0.05
    # Tetto annuo agli acquisti BTD, in percentuale del capitale iniziale.
    # None = nessun limite, ed e' il default: un tetto stringente cambia la
    # natura della strategia, perche' il budget si esaurisce sui cali
    # superficiali di inizio anno e lascia scoperti quelli profondi che
    # arrivano dopo. Resta configurabile per chi lo vuole simulare.
    btd_cap_annuo_pct: Optional[float] = None
    btd_dd_weekly_limit: float = -0.90         # blocca il BTD se il DD weekly e' sotto
    btd_execution: str = "open"                # "open" (mese del segnale) | "close" (legacy)

    # Quando la variante Reinvest rimette al lavoro il risultato netto delle
    # opzioni. "subito" alla chiusura di ogni periodo, "al_btd" solo insieme al
    # prossimo acquisto sui cali, allo stesso prezzo. Sulle altre due varianti
    # non ha alcun effetto: una non vende call, l'altra tiene i premi in cassa.
    reinvesto_modo: str = "subito"             # "subito" | "al_btd"

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

    @property
    def settimanale(self) -> bool:
        return str(self.cadenza) == "settimanale"

    @property
    def periodi_anno(self) -> int:
        """Quanti passi elementari entrano in un anno: divisore di interessi e metriche."""
        return PERIODI_ANNO[CADENZA_VALIDA(self.cadenza)]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["premium_model"] = self.premium_model.to_dict()
        return d


VARIANTS = {
    "no_premi":       {"label": "BTD No Premi",             "vende_call": False, "reinveste": False},
    "premi_cash":     {"label": "BTD + Premi (Cash)",       "vende_call": True,  "reinveste": False,
                       "premi_separati": True},
    # Anche qui i premi stanno in un conto separato: cosi' non finanziano gli
    # acquisti sui cali (che restano capitale proprio) e l'unica differenza fra
    # "subito" e "al_btd" e' il momento in cui tornano al lavoro, non la
    # contabilita'. Prima i premi finivano nella cassa generale e riducevano di
    # nascosto il capitale versato per il BTD.
    "premi_reinvest": {"label": "BTD + Premi (Reinvest)",   "vende_call": True,  "reinveste": True,
                       "premi_separati": True},
}

# Il termine di paragone: stesso identico ciclo annuale della strategia (capitale
# fisso impiegato a gennaio, tutto liquidato a dicembre, si ricomincia), ma senza
# vendere opzioni e senza comprare sui cali. Gira attraverso lo stesso motore
# delle varianti, quindi ha le stesse colonne, le stesse metriche e la stessa
# tabella annuale, e puo' comparire in ogni grafico accanto alle altre curve.
BENCHMARK = "benchmark"
SPEC_BENCHMARK = {"label": "Buy & Hold (stesso ciclo annuale)",
                  "vende_call": False, "reinveste": False, "usa_btd": False}
TUTTE_LE_SPEC = {**VARIANTS, BENCHMARK: SPEC_BENCHMARK}


# ----------------------------------------------------------------------------
# Helper temporali
# ----------------------------------------------------------------------------
def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def _month_days(ts: pd.Timestamp) -> int:
    return int(ts.days_in_month)


def _inizio_periodo(ts: pd.Timestamp, cadenza: str) -> pd.Timestamp:
    """Primo giorno del periodo a cui appartiene la barra.

    E' il momento in cui si impiega il capitale e si fissa lo strike, ed e'
    anche il taglio anti-look-ahead: la volatilita' usata per prezzare il premio
    deve essere quella nota STRETTAMENTE prima di questo istante.

    Sulla cadenza settimanale si risale al lunedi della settimana della barra,
    che e' corretto sia che il fornitore dati indicizzi la barra al primo giorno
    della settimana sia che la indicizzi all'ultimo.
    """
    ts = pd.Timestamp(ts)
    if CADENZA_VALIDA(cadenza) == "settimanale":
        return (ts - pd.Timedelta(days=int(ts.weekday()))).normalize()
    return _month_start(ts)


def _giorni_periodo(ts: pd.Timestamp, cadenza: str) -> int:
    """Durata in giorni di calendario del periodo: e' la vita della call venduta."""
    if CADENZA_VALIDA(cadenza) == "settimanale":
        return 7
    return _month_days(ts)


# ----------------------------------------------------------------------------
# Preparazione dati
# ----------------------------------------------------------------------------
def prepare_market_data(
    monthly: pd.DataFrame,
    weekly: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    cfg: BacktestConfig,
) -> Dict[str, Any]:
    """Normalizza i dati e calcola segnali e volatilita' realizzata.

    La cadenza sceglie quale serie guida il backtest: quella mensile o quella
    settimanale. La serie settimanale resta comunque usata per intero, warm-up
    compreso, dal filtro sul drawdown.
    """
    warnings: List[str] = []
    cadenza = CADENZA_VALIDA(cfg.cadenza)
    ppa = PERIODI_ANNO[cadenza]

    if cadenza == "settimanale":
        if weekly is None or weekly.empty:
            raise ValueError("Cadenza settimanale richiesta ma la serie settimanale non e' disponibile.")
        m = weekly.copy()
        # La serie settimanale viene scaricata con un warm-up di due anni per il
        # filtro sul drawdown: senza questo taglio il backtest partirebbe molto
        # prima della data chiesta dall'utente.
        if cfg.start_date:
            m = m[m.index >= pd.Timestamp(cfg.start_date)]
        if cfg.end_date:
            m = m[m.index <= pd.Timestamp(cfg.end_date)]
    else:
        m = monthly.copy()

    m = m[~m.index.duplicated(keep="last")].sort_index()
    m["rendimento_mese"] = m["Close"].pct_change()
    # Il segnale del periodo i guarda il periodo i-1: nessun look-ahead.
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
        vol_series = volmod.vol_from_periods(m, periodi_anno=ppa)
        vol_source = f"{cadenza} (fallback)"
        warnings.append(
            f"Dati giornalieri non disponibili: la volatilita' e' stimata dai rendimenti "
            f"{'settimanali' if cadenza == 'settimanale' else 'mensili'} ed e' molto meno precisa."
        )

    return {"monthly": m, "dd_weekly": dd_weekly, "vol": vol_series,
            "vol_source": vol_source, "warnings": warnings,
            "cadenza": cadenza, "periodi_anno": ppa}


# ----------------------------------------------------------------------------
# Simulazione di una variante
# ----------------------------------------------------------------------------
def run_variant(market: Dict[str, Any], cfg: BacktestConfig, variant: str) -> Dict[str, Any]:
    spec = TUTTE_LE_SPEC[variant]
    vende_call = bool(spec["vende_call"])
    reinveste = bool(spec["reinveste"])
    reinvesto_al_btd = reinveste and str(cfg.reinvesto_modo) == "al_btd"
    usa_btd = bool(spec.get("usa_btd", True))
    # I premi incassati restano in un conto a parte e NON finanziano gli acquisti
    # sui cali: quelli si pagano con capitale proprio. E' denaro che arriva dal
    # mercato, non dalle tasche di chi investe, e va tenuto distinto.
    premi_separati = bool(spec.get("premi_separati", False))

    m: pd.DataFrame = market["monthly"]
    dd_weekly: pd.Series = market["dd_weekly"]
    vol_series: pd.Series = market["vol"]
    cadenza = CADENZA_VALIDA(market.get("cadenza", cfg.cadenza))
    ppa = float(market.get("periodi_anno", PERIODI_ANNO[cadenza]))

    cap0_base = float(cfg.capitale_iniziale)
    cap_add_base = float(cfg.capitale_addizionale)
    capitale_base = cap0_base + cap_add_base
    componi = str(cfg.capitale_modo) == "composto"
    # Valori dell'anno in corso: identici alla base in modalita' fissa, scalati
    # sul conto disponibile in modalita' composta.
    cap0 = cap0_base
    cap_add = cap_add_base
    capitale_annuo = capitale_base
    tetto_btd = (cap0 * float(cfg.btd_cap_annuo_pct)
                 if cfg.btd_cap_annuo_pct else float('inf'))
    boost_per_acquisto = cap0 * float(cfg.boost_pct)
    pm = cfg.premium_model
    # I tassi annui vanno ripartiti sul passo effettivo: dodicesimi sulla
    # cadenza mensile, cinquantaduesimi su quella settimanale.
    idle_m = float(cfg.idle_cash_rate) / ppa
    debito_m = float(cfg.debit_cash_rate) / ppa

    # Stato del conto
    cassa = 0.0            # liquidita' operativa: capitale, liquidazioni, BTD
    cassa_opzioni = 0.0    # premi incassati e intrinseco pagato, se separati
    # Salvadanaio del reinvestimento differito: risultato netto delle opzioni
    # gia' maturato e non ancora rimesso al lavoro. In modalita' "al_btd" a fine
    # periodo vale sempre quanto `cassa_opzioni`, ed e' l'invariante che lo
    # tiene onesto. Il denaro sta in `cassa_opzioni`, qui c'e' solo il conto.
    premi_pendenti = 0.0
    premi_investiti = 0.0
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

        ms = _inizio_periodo(data, cadenza)
        versato_mese = 0.0
        liquidazione = 0.0
        nuovo_anno = (anno_corrente is None) or (data.year != anno_corrente)

        # ---------------- Reset annuale ----------------
        if nuovo_anno:
            if anno_corrente is not None:
                # liquida al close del mese precedente e riunisce i due conti
                prezzo_liq = float(m.iloc[i - 1]["Close"])
                liquidazione = (quote_coperte + quote_extra) * prezzo_liq
                cassa += liquidazione + cassa_opzioni
                cassa_opzioni = 0.0
                quote_coperte = quote_extra = 0.0
            # Quanto rimettere al lavoro quest'anno
            if componi and cassa > 0:
                # Una parte resta liquida per comprare sui cali durante l'anno.
                # Il benchmark non compra sui cali, quindi non trattiene nulla:
                # tenergli fermo un terzo del conto sarebbe una zavorra falsa.
                riserva = max(0.0, float(cfg.riserva_btd_pct)) if usa_btd else 0.0
                capitale_annuo = cassa / (1.0 + riserva)
            else:
                capitale_annuo = capitale_base
            fattore = capitale_annuo / capitale_base if capitale_base > 0 else 1.0
            cap0 = cap0_base * fattore
            cap_add = cap_add_base * fattore
            tetto_btd = (cap0 * float(cfg.btd_cap_annuo_pct)
                         if cfg.btd_cap_annuo_pct else float("inf"))
            boost_per_acquisto = cap0 * float(cfg.boost_pct)

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
            # Il salvadanaio e' stato liquidato insieme a tutto il resto.
            premi_pendenti = 0.0
            anno_corrente = data.year

        # Interessi sulla liquidita'. Il saldo puo' andare a debito quando il
        # riacquisto della call a intrinseco supera la cassa disponibile: e' un
        # finanziamento garantito dalle azioni in portafoglio, e come tale costa.
        totale_liquido = cassa + cassa_opzioni
        interessi = 0.0
        if totale_liquido > 0 and idle_m:
            interessi = totale_liquido * idle_m
        elif totale_liquido < 0 and debito_m:
            interessi = totale_liquido * debito_m
        cassa += interessi

        # ---------------- Premio della call ----------------
        sigma = volmod.sigma_at(vol_series, ms, fallback=np.nan)
        T = _giorni_periodo(data, cadenza) / 365.0
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
            if premi_separati:
                cassa_opzioni += premio
            else:
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
        reinvestito_btd = 0.0
        potenziale = 0.0
        tagliato_dal_tetto = 0.0
        prezzo_btd = O if cfg.btd_execution == "open" else C
        if usa_btd and segnale and not bloccato and np.isfinite(rend_trigger) and rend_trigger < 0:
            # quota legata all'entita' del calo del mese precedente
            quota_calo = abs(rend_trigger) * cap0
            # boost: si aggiunge per intero a ogni acquisto
            quota_boost = boost_per_acquisto

            potenziale = quota_calo + quota_boost
            btd_importo = max(0.0, min(potenziale, tetto_btd - btd_usato_anno))
            # Quanto il tetto annuo ha tolto a questo acquisto. Serve a vedere
            # quando il tetto diventa il vincolo che decide la strategia: alzando
            # il boost il budget si esaurisce prima, e i cali piu' profondi, che
            # sul serio arrivano piu' tardi nell'anno, restano fuori.
            tagliato_dal_tetto = max(0.0, potenziale - btd_importo)
            if btd_importo > 1e-9:
                # se il tetto annuo taglia l'acquisto, taglia entrambe le quote
                if potenziale > 0:
                    fattore = btd_importo / potenziale
                    quota_calo *= fattore
                    quota_boost *= fattore
                if componi:
                    # niente denaro fresco: si compra solo con la riserva
                    btd_importo = min(btd_importo, max(0.0, cassa))
                    if potenziale > 0:
                        f = btd_importo / potenziale
                        quota_calo, quota_boost = quota_calo * f, quota_boost * f
                    tagliato_dal_tetto = max(0.0, potenziale - btd_importo)
                elif cassa < btd_importo:                   # serve denaro fresco
                    manca = btd_importo - cassa
                    versamenti += manca
                    versato_mese += manca
                    cassa += manca
                    bh_quote += manca / prezzo_btd
                if btd_importo <= 1e-9:
                    btd_importo = quota_calo = quota_boost = 0.0
                cassa -= btd_importo
                quote_extra += btd_importo / prezzo_btd
                btd_usato_anno += btd_importo
            else:
                btd_importo = quota_calo = quota_boost = 0.0

        # ---------------- Premi arretrati, insieme al Buy-The-Dip ----------------
        # Il salvadanaio si svuota solo se un acquisto sui cali e' avvenuto
        # davvero: un segnale bloccato dal filtro o azzerato dal tetto non conta.
        # Entra al prezzo del BTD, che e' il punto: si comprano i premi arretrati
        # sul ribasso invece che al prezzo corrente qualunque esso sia.
        #
        # Si versa tutto quello che il conto delle opzioni ha in cassa, premio del
        # periodo in corso COMPRESO: e' stato accreditato all'apertura, poche
        # righe piu' su, ed e' denaro disponibile esattamente come su un conto
        # vero. Il suo intrinseco si paghera' a scadenza, piu' avanti, e potra'
        # mandare a debito il conto delle opzioni. Aspettare di conoscerlo
        # significherebbe lasciare fermo un mese di premio proprio nel momento in
        # cui il ribasso lo renderebbe piu' utile.
        if reinvesto_al_btd and btd_importo > 1e-9 and cassa_opzioni > 1e-9:
            arretrati = cassa_opzioni
            if arretrati > 1e-9 and prezzo_btd > 0:
                cassa_opzioni -= arretrati
                quote_extra += arretrati / prezzo_btd
                premi_pendenti -= arretrati
                premi_investiti += arretrati
                reinvestito_btd = arretrati

        # ---------------- Scadenza della call ----------------
        intrinseco = 0.0
        if vende_call and cfg.applica_cap and quote_coperte > 0 and np.isfinite(strike):
            intrinseco = quote_coperte * max(0.0, C - strike)
            if premi_separati:
                cassa_opzioni -= intrinseco
            else:
                cassa -= intrinseco

        netto_opzione = premio - intrinseco

        # ---------------- Reinvestimento ----------------
        reinvestito_chiusura = 0.0
        if reinvesto_al_btd:
            # Il netto di questo periodo entra nel salvadanaio: sara' disponibile
            # dal prossimo acquisto sui cali in poi, mai per quello appena fatto.
            premi_pendenti += netto_opzione
        elif reinveste and netto_opzione > 0 and C > 0:
            fonte = cassa_opzioni if premi_separati else cassa
            reinvestito_chiusura = min(netto_opzione, max(0.0, fonte))
            if reinvestito_chiusura > 0:
                if premi_separati:
                    cassa_opzioni -= reinvestito_chiusura
                else:
                    cassa -= reinvestito_chiusura
                quote_extra += reinvestito_chiusura / C
                premi_investiti += reinvestito_chiusura
        reinvestito = reinvestito_btd + reinvestito_chiusura

        # ---------------- Mark to market ----------------
        valore = (quote_coperte + quote_extra) * C + cassa + cassa_opzioni
        rows.append({
            "data": data, "anno": int(data.year),
            "open": O, "close": C,
            "rendimento_mese": float(bar["rendimento_mese"]) if pd.notna(bar["rendimento_mese"]) else np.nan,
            "segnale_btd": segnale, "btd_bloccato": bloccato, "dd_weekly": dd_w,
            "btd_importo": btd_importo,
            "btd_quota_calo": quota_calo, "btd_quota_boost": quota_boost,
            "btd_residuo_anno": (max(0.0, tetto_btd - btd_usato_anno)
                                 if np.isfinite(tetto_btd) else np.nan),
            "capitale_impiegato_anno": capitale_annuo + btd_usato_anno,
            "btd_potenziale": potenziale,
            "btd_tagliato_dal_tetto": tagliato_dal_tetto,
            "btd_saltato_dal_tetto": bool(tagliato_dal_tetto > 1e-9 and btd_importo <= 1e-9),
            "btd_prezzo": prezzo_btd if btd_importo > 0 else np.nan,
            "sigma_stimata": float(sigma) if np.isfinite(sigma) else np.nan,
            "sigma_implicita": float(pm.implied_sigma(sigma)) if np.isfinite(sigma) else np.nan,
            "vrp_applicato": float(vrp_applicato) if np.isfinite(vrp_applicato) else np.nan,
            "strike": float(strike) if np.isfinite(strike) else np.nan,
            "premio_pct": premio_pct, "premio": premio,
            "intrinseco_pagato": intrinseco, "netto_opzione": netto_opzione,
            "reinvestito": reinvestito,
            # Separati perche' cadono in due momenti diversi del periodo: gli
            # arretrati al prezzo del BTD, il resto alla chiusura. La
            # rivalutazione giornaliera ha bisogno di sapere quale dei due.
            "reinvestito_al_btd": reinvestito_btd,
            "reinvestito_a_chiusura": reinvestito_chiusura,
            "premi_pendenti": premi_pendenti,
            "premi_investiti_cum": premi_investiti,
            "quote_coperte": quote_coperte, "quote_extra": quote_extra,
            "cassa": cassa + cassa_opzioni, "cassa_opzioni": cassa_opzioni,
            "interessi": interessi, "liquidazione": liquidazione,
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
            "yearly": _yearly_table(df), "metrics": {}, "cadenza": cadenza}


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
        # Capitale davvero tirato fuori nell'anno: quello impiegato a gennaio
        # piu' gli acquisti sui cali. NON comprende i premi reinvestiti, che
        # sono denaro arrivato dal mercato e non dalle tasche dell'investitore.
        capitale_investito = float(g["capitale_impiegato_anno"].iloc[-1])
        out.append({
            "anno": int(anno),
            "periodi": int(len(g)),
            "capitale_investito": capitale_investito,
            "rendimento_anno": (risultato / capitale_investito
                                if capitale_investito > 0 else float("nan")),
            "rendimento_sottostante": float(g["close"].iloc[-1] / g["open"].iloc[0] - 1.0),
            "premi_incassati": float(g["premio"].sum()),
            "intrinseco_pagato": float(g["intrinseco_pagato"].sum()),
            "netto_opzioni": float(g["netto_opzione"].sum()),
            "btd_numero": int((g["btd_importo"] > 0).sum()),
            "btd_investito": float(g["btd_importo"].sum()),
            "btd_da_calo": float(g["btd_quota_calo"].sum()),
            "btd_da_boost": float(g["btd_quota_boost"].sum()),
            "btd_tagliato_dal_tetto": float(g["btd_tagliato_dal_tetto"].sum()),
            "btd_segnali_saltati": int(g["btd_saltato_dal_tetto"].sum()),
            "btd_prezzo_medio": (
                float(g.loc[g["btd_importo"] > 0, "btd_importo"].sum()
                      / (g.loc[g["btd_importo"] > 0, "btd_importo"]
                         / g.loc[g["btd_importo"] > 0, "btd_prezzo"]).sum())
                if (g["btd_importo"] > 0).any() else float("nan")),
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
    """Fotografia periodo per periodo di un singolo anno, per seguire la strategia dal vivo.

    Senza `anno` restituisce l'ultimo presente nel backtest, cioe' quello in
    corso quando il backtest arriva a oggi. La chiave `mesi` resta per
    compatibilita' e contiene le barre del periodo, mensili o settimanali.
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
    quota_tetto = cfg.get("btd_cap_annuo_pct")
    tetto = cap0 * float(quota_tetto) if quota_tetto else float("inf")
    cap_add = float(cfg.get("capitale_addizionale", 0.0))

    btd_usato = float(g["btd_importo"].sum())
    quote_tot = (g["quote_coperte"] + g["quote_extra"]).iloc[-1]
    valore_iniziale = float(g["valore_portafoglio"].iloc[0])
    valore_corrente = float(g["valore_portafoglio"].iloc[-1])

    return {
        "anno": anno,
        "mesi": g,
        "periodi": g,
        "cadenza": CADENZA_VALIDA(cfg.get("cadenza", "mensile")),
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
            "btd_tagliato_dal_tetto": float(g["btd_tagliato_dal_tetto"].sum()),
            "btd_segnali_saltati": int(g["btd_saltato_dal_tetto"].sum()),
            "btd_prezzo_medio": (
                float(g.loc[g["btd_importo"] > 0, "btd_importo"].sum()
                      / (g.loc[g["btd_importo"] > 0, "btd_importo"]
                         / g.loc[g["btd_importo"] > 0, "btd_prezzo"]).sum())
                if (g["btd_importo"] > 0).any() else float("nan")),
            "btd_residuo": (max(0.0, tetto - btd_usato) if np.isfinite(tetto) else None),
            "btd_tetto": tetto if np.isfinite(tetto) else None,
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
    """Cosa fara' la strategia il periodo prossimo, con i dati disponibili oggi.

    Tutte le grandezze sono gia' determinate dalla chiusura dell'ultimo periodo:
    il segnale guarda indietro, la volatilita' e' quella nota, e l'unica cosa
    che manca e' il prezzo di apertura, quindi gli importi in quote sono
    indicativi mentre quelli in valuta sono esatti.
    """
    res = (risultato.get("varianti") or {}).get(variante)
    if not res or res["monthly"].empty:
        return {}
    df: pd.DataFrame = res["monthly"]
    cfg = risultato.get("config", {})
    cadenza = CADENZA_VALIDA(cfg.get("cadenza", "mensile"))
    ultima = df.iloc[-1]
    if cadenza == "settimanale":
        prossimo = _inizio_periodo(pd.Timestamp(df.index[-1]), cadenza) + pd.Timedelta(days=7)
    else:
        prossimo = pd.Timestamp(df.index[-1]) + pd.offsets.MonthBegin(1)
    nuovo_anno = int(prossimo.year) != int(ultima["anno"])

    cap0 = float(cfg.get("capitale_iniziale", 0.0))
    cap_add = float(cfg.get("capitale_addizionale", 0.0))
    quota_tetto = cfg.get("btd_cap_annuo_pct")
    tetto = cap0 * float(quota_tetto) if quota_tetto else float("inf")
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
    T = _giorni_periodo(prossimo, cadenza) / 365.0
    quote_coperte = 0.0 if nuovo_anno else float(ultima["quote_coperte"])
    quota_prem: Dict[str, float] = {}
    if np.isfinite(sigma) and sigma > 0:
        quota_prem = pm.quote(spot, sigma, T)

    etichetta = (prossimo.strftime("settimana del %d/%m/%Y") if cadenza == "settimanale"
                 else prossimo.strftime("%Y-%m"))
    return {
        "mese": etichetta,
        "periodo": etichetta,
        "cadenza": cadenza,
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
        "btd_residuo_anno": (max(0.0, tetto - btd_usato) if np.isfinite(tetto) else None),
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

    cadenza = CADENZA_VALIDA(cfg.cadenza)
    ppa = PERIODI_ANNO[cadenza]
    sorgente = weekly if cadenza == "settimanale" else monthly
    if sorgente is None or sorgente.empty or len(sorgente) < 2:
        manca = "settimanale" if cadenza == "settimanale" else "mensile"
        return {"ok": False,
                "errore": f"Serie {manca} vuota o troppo corta per questo ticker.",
                "config": cfg.to_dict(), "varianti": {}, "warnings": []}

    market = prepare_market_data(monthly, weekly, daily, cfg)
    m = market["monthly"]
    if m.empty or len(m) < 2:
        return {"ok": False,
                "errore": (f"Nella finestra richiesta ci sono meno di due barre "
                           f"{'settimanali' if cadenza == 'settimanale' else 'mensili'}."),
                "config": cfg.to_dict(), "varianti": {}, "warnings": market["warnings"]}

    benchmark = run_variant(market, cfg, BENCHMARK)
    bm = benchmark["monthly"]

    # Rivalutazione giornaliera: non cambia una virgola di come si opera, cambia
    # solo con che frequenza si guarda il conto. Senza dati giornalieri si resta
    # alla valorizzazione di fine periodo, dicendolo.
    pm = cfg.premium_model
    giornalieri_disponibili = (isinstance(daily, pd.DataFrame) and not daily.empty)
    if not giornalieri_disponibili:
        market["warnings"].append(
            "Dati giornalieri non disponibili: il conto e' valorizzato solo alla chiusura "
            "di ogni periodo, quindi i drawdown rientrati prima della chiusura non compaiono."
        )

    def _giornaliera(res: Dict[str, Any]) -> None:
        """Aggiunge a una variante la serie giornaliera e le sue metriche di rischio."""
        if not giornalieri_disponibili or res["monthly"].empty:
            res["daily"] = pd.DataFrame()
            return
        res["daily"] = valorizza_giornaliero(res["monthly"], daily, cfg, pm.r, pm.q)
        if res.get("metrics") is not None and not res["daily"].empty:
            res["metrics"].update(metriche_giornaliere(
                res["daily"], res["monthly"], cfg.var_confidence))

    risultati: Dict[str, Any] = {}
    for name in VARIANTS:
        res = run_variant(market, cfg, name)
        if not res["monthly"].empty and not bm.empty:
            idx = res["monthly"].index
            res["monthly"]["ciclo_annuale"] = bm["valore_portafoglio"].reindex(idx)
            res["monthly"]["ciclo_annuale_pnl"] = bm["pnl_netto"].reindex(idx)
            res["monthly"]["ciclo_annuale_twr"] = bm["twr_mese"].reindex(idx)
            res["monthly"]["ciclo_annuale_dd"] = bm["dd_twr_pct"].reindex(idx)
        if not res["monthly"].empty:
            res["metrics"] = compute_metrics(res["monthly"], cfg.var_confidence, ppa)
        _giornaliera(res)
        risultati[name] = res

    if not bm.empty:
        benchmark["metrics"] = compute_metrics(bm, cfg.var_confidence, ppa)
        _giornaliera(benchmark)
        # Confronto col benchmark sul rendimento annuo: si calcola qui, dove il
        # benchmark e' gia' stato girato, invece che dentro compute_metrics.
        rb = benchmark["metrics"].get("rendimento_medio")
        for res in risultati.values():
            mt = res.get("metrics")
            if not mt:
                continue
            suo_dd = benchmark["metrics"].get("max_dd_giornaliero_pct")
            mio_dd = mt.get("max_dd_giornaliero_pct")
            mt["ciclo_max_dd_giornaliero_pct"] = suo_dd
            # La riduzione del drawdown va misurata sul drawdown VERO: farla
            # sulle chiusure di periodo confronterebbe due numeri entrambi
            # sottostimati, e non nello stesso modo.
            if mio_dd is not None and suo_dd:
                mt["riduzione_dd_giornaliera_vs_ciclo"] = 1.0 - abs(mio_dd) / abs(suo_dd)
            mt["ciclo_rendimento_medio"] = rb
            mt["ciclo_rendimento_volatilita"] = benchmark["metrics"].get("rendimento_volatilita")
            mt["ciclo_rendimento_su_rischio"] = benchmark["metrics"].get("rendimento_su_rischio")
            mio = mt.get("rendimento_medio")
            mt["extra_rendimento_vs_ciclo"] = (
                mio - rb if (mio is not None and rb is not None) else None)

    scarti = [r["metrics"].get("riconciliazione_scarto") for r in risultati.values()
              if r.get("metrics")]
    scarti = [x for x in scarti if x is not None]
    if scarti and max(scarti) > 1e-3:
        market["warnings"].append(
            f"La serie giornaliera e quella di periodo non coincidono a fine periodo "
            f"(scarto massimo {max(scarti):.2%} del conto): probabilmente i due download "
            f"hanno aggiustamenti diversi per dividendi o split. I drawdown giornalieri "
            f"restano indicativi."
        )

    # Buy & Hold semplice: solo il capitale iniziale, mai piu' toccato
    capitale_annuo = cfg.capitale_iniziale + cfg.capitale_addizionale
    o0 = float(m["Open"].iloc[0])
    bh_semplice = (m["Close"] / o0 * capitale_annuo).rename("bh_semplice") if o0 > 0 else pd.Series(dtype=float)


    return {
        "ok": True,
        "config": cfg.to_dict(),
        "cadenza": cadenza,
        "periodi_anno": ppa,
        "mercato": {
            "prezzi": m[["Open", "Close", "rendimento_mese", "segnale_btd"]],
            "vol": market["vol"],
            "vol_source": market["vol_source"],
            "dd_weekly": market["dd_weekly"],
            "bh_semplice": bh_semplice,
        },
        "varianti": risultati,
        "benchmark": benchmark,
        "warnings": market["warnings"],
        # Serie giornaliera grezza: serve a ricalibrare la volatilita' con altri
        # stimatori senza riscaricare i dati. Il prefisso la tiene fuori dall'export.
        "_giornaliero": daily if isinstance(daily, pd.DataFrame) else None,
    }
