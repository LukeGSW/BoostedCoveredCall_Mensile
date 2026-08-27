"""Boosted Covered Call — Studio Mensile.

Dashboard Streamlit: capitale fisso annuo coperto da una call mensile venduta a
delta 0.50, Buy-The-Dip potenziato sui mesi negativi, reset a fine anno.

Il premio non e' piu' un numero da indovinare: viene stimato dalla volatilita'
realizzata del sottostante e puo' essere calibrato sui prezzi reali delle
opzioni caricati dall'utente.
"""
from __future__ import annotations

import datetime as dt
import html
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from kq_btd_cc import CSS, __version__ as VERSIONE
from kq_btd_cc import calibration as calib
from kq_btd_cc import charts
from kq_btd_cc.core import PREFERENZE_DEFAULT, costruisci_config, costruisci_figure
from kq_btd_cc.data_api import ChiaveMancante, DatiNonDisponibili, carica_serie, ha_api_key
from kq_btd_cc.engine import (VARIANTS, dettaglio_anno, piano_prossimo_mese,
                              run_backtest)
from kq_btd_cc.export import build_export, export_json_bytes, nome_file_export
from kq_btd_cc.metrics import ETICHETTE, format_value, metrics_table
from kq_btd_cc.pricing import PremiumModel
from kq_btd_cc import riferimenti
from kq_btd_cc.utils import fmt_currency_compact, fmt_num, fmt_pct
from kq_btd_cc.vol import VOL_MODELS

st.set_page_config(
    page_title="Boosted Covered Call — Studio Mensile",
    page_icon="📈", layout="wide", initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "scrollZoom": False,
}


def _kwargs_larghezza() -> Dict[str, Any]:
    """Streamlit ha sostituito use_container_width con width='stretch' dalla 1.49."""
    try:
        maggiore, minore = (int(p) for p in st.__version__.split(".")[:2])
        if (maggiore, minore) >= (1, 49):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


LARGO = _kwargs_larghezza()

# Estremi del periodo selezionabile. 1970 copre tutto lo storico che EODHD puo'
# restituire su indici e azioni, bolla dot-com e crisi del 2008 comprese.
ANNO_MINIMO = 1970
OGGI = dt.date.today()

MESI_IT = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
           "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]


def _ultimo_giorno(anno: int, mese: int) -> dt.date:
    """Ultimo giorno del mese indicato."""
    return (dt.date(anno + (mese == 12), (mese % 12) + 1, 1) - dt.timedelta(days=1))


# ---------------------------------------------------------------------------
# Helper di rendering
# ---------------------------------------------------------------------------
def grafico(fig, key: Optional[str] = None) -> None:
    st.plotly_chart(fig, config=PLOTLY_CONFIG, key=key, **LARGO)


def kpi_cards(voci: List[Tuple[str, str, str, Optional[str]]]) -> None:
    """voci: (etichetta, valore, sottotitolo, segno) con segno in {'pos','neg',None}."""
    blocchi = []
    for etichetta, valore, sotto, segno in voci:
        cls = {"pos": " kq-pos", "neg": " kq-neg"}.get(segno or "", "")
        blocchi.append(
            f'<div class="kq-kpi"><div class="lab">{html.escape(etichetta)}</div>'
            f'<div class="val{cls}">{html.escape(valore)}</div>'
            f'<div class="sub">{html.escape(sotto)}</div></div>'
        )
    st.markdown(f'<div class="kq-kpi-row">{"".join(blocchi)}</div>', unsafe_allow_html=True)


def nota(testo: str) -> None:
    st.markdown(f'<div class="kq-note">{testo}</div>', unsafe_allow_html=True)


def segno_di(valore: Optional[float]) -> Optional[str]:
    if valore is None:
        return None
    return "pos" if valore > 0 else ("neg" if valore < 0 else None)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    with st.sidebar:
        st.markdown("### Parametri")

        with st.expander("Sottostante e periodo", expanded=True):
            ticker = st.text_input(
                "Ticker EODHD", value="BTC-USD.CC",
                help="Formato EODHD: BTC-USD.CC, ETH-USD.CC, SPY.US, AAPL.US...").strip()
            # Niente st.date_input qui: il calendario di Streamlit elenca nel menu
            # degli anni una finestra fissa di vent'anni, e ignora min_value. Con
            # max_value a oggi si ferma al 2007, tagliando fuori la bolla dot-com.
            # Il motore lavora su barre mensili, quindi anno e mese bastano.
            c1, c2 = st.columns([1, 1.4])
            with c1:
                anno_inizio = st.number_input(
                    "Anno di inizio", min_value=ANNO_MINIMO, max_value=OGGI.year,
                    value=2018, step=1, format="%d",
                    help="Puoi scriverlo direttamente. Lo storico effettivo dipende dal "
                         "ticker: EODHD parte dalla prima data che ha.")
            with c2:
                mese_inizio = st.selectbox(
                    "Mese di inizio", options=list(range(1, 13)),
                    format_func=lambda m: MESI_IT[m - 1].capitalize(), index=0)
            data_inizio = dt.date(int(anno_inizio), int(mese_inizio), 1)

            fine_manuale = st.checkbox(
                "Imposta una data di fine", value=False,
                help="Senza questa opzione il backtest arriva all'ultimo dato disponibile.")
            c3, c4 = st.columns([1, 1.4])
            with c3:
                anno_fine = st.number_input(
                    "Anno di fine", min_value=ANNO_MINIMO, max_value=OGGI.year,
                    value=OGGI.year, step=1, format="%d", disabled=not fine_manuale)
            with c4:
                mese_fine = st.selectbox(
                    "Mese di fine", options=list(range(1, 13)),
                    format_func=lambda m: MESI_IT[m - 1].capitalize(),
                    index=OGGI.month - 1, disabled=not fine_manuale)
            data_fine = min(OGGI, _ultimo_giorno(int(anno_fine), int(mese_fine)))

            if fine_manuale and data_fine <= data_inizio:
                st.error("Il periodo di fine deve venire dopo quello di inizio.")
            st.caption(
                f"Periodo: {MESI_IT[data_inizio.month - 1]} {data_inizio.year} → "
                + (f"{MESI_IT[data_fine.month - 1]} {data_fine.year}" if fine_manuale
                   else "ultimo dato disponibile")
            )

        with st.expander("Capitale", expanded=True):
            capitale = st.number_input(
                "Capitale iniziale — coperto dalla call", value=25_000, step=1_000,
                min_value=1_000,
                help="Impiegato all'apertura di gennaio. E' la base su cui si vende la call: "
                     "su questo si incassa il premio e su questo il rialzo viene tagliato. "
                     "E' anche il riferimento percentuale del BTD, del boost e del tetto annuo.")
            capitale_add = st.number_input(
                "Capitale addizionale annuale — non coperto", value=0, step=1_000, min_value=0,
                help="Entra una volta sola all'apertura di gennaio, insieme al capitale "
                     "iniziale, e compra sottostante che resta li' per tutto l'anno. La call "
                     "NON lo cappa: si tiene tutto il rialzo, e su di esso non si incassa "
                     "nessun premio. Viene liquidato a fine anno come il resto.")
            idle = st.slider(
                "Remunerazione della cassa non impiegata (annua)", 0.0, 6.0, 0.0, 0.25,
                help="Interesse riconosciuto sulla liquidita' ferma fra un ciclo e l'altro.") / 100.0
            debito = st.slider(
                "Costo del saldo a debito (annuo)", 0.0, 15.0, 6.0, 0.5,
                help="Quando il riacquisto della call a scadenza supera la liquidita' "
                     "disponibile, il conto va a debito contro le azioni in portafoglio. "
                     "Questo e' il tasso applicato a quel finanziamento.") / 100.0

        with st.expander("Buy-The-Dip", expanded=True):
            boost = st.slider(
                "BTD Boost", 0.0, 15.0, 5.0, 0.5,
                help="Percentuale del capitale iniziale che si aggiunge a OGNI acquisto BTD, "
                     "oltre alla quota legata all'entita' del calo. Eredita tutto dal BTD: "
                     "stesso momento e prezzo di acquisto, quote non coperte dalla call, "
                     "stesso tetto annuo, liquidazione a fine anno. Da non confondere con il "
                     "capitale addizionale annuale, che entra invece una volta sola.") / 100.0
            tetto = st.slider(
                "Tetto annuo agli acquisti BTD", 25, 400, 100, 25,
                help="Massimo cumulato acquistabile in un anno, in percentuale del "
                     "capitale fisso.") / 100.0
            limite_dd = st.slider(
                "Sospendi il BTD sotto questo drawdown settimanale", -95.0, -20.0, -90.0, 5.0,
                help="Se il drawdown settimanale dell'asset e' piu' profondo di questa "
                     "soglia, l'acquisto del mese viene saltato.") / 100.0
            esecuzione = st.radio(
                "Esecuzione dell'acquisto", ["Apertura del mese", "Chiusura del mese"],
                index=0, horizontal=False,
                help="Il segnale e' noto alla chiusura del mese precedente: comprare "
                     "all'apertura e' la scelta eseguibile. La chiusura riproduce il "
                     "comportamento della versione precedente della dashboard.")

        with st.expander("Premio della call", expanded=True):
            st.caption("Il premio viene calcolato dalla volatilita' del sottostante, "
                       "non impostato a mano.")
            delta_target = st.slider(
                "Delta della call venduta", 0.20, 0.80, 0.50, 0.05,
                help="0.50 e' la call at-the-money classica della strategia.")
            vrp = st.slider(
                "Volatility risk premium a volatilita 20%", 0.60, 2.00, 0.96, 0.01,
                help="Di quanto la volatilita' implicita eccede quella realizzata, misurato "
                     "su un sottostante che oscilla del 20% annuo. Il default viene da 1.630 "
                     "vendite reali di call ATM mensili su sei sottostanti. Il livello dipende "
                     "anche dallo stimatore di volatilita' scelto qui sotto: la scheda "
                     "Calibrazione premio lo misura sul tuo sottostante.")
            vrp_slope = st.slider(
                "Pendenza del VRP rispetto alla volatilita", -0.40, 0.10, -0.125, 0.005,
                help="Sui prezzi reali il premio al rischio si riduce quando il sottostante "
                     "e' piu' volatile: circa 1.03 al 12% di volatilita', 0.83 al 60%. "
                     "Modellarlo riduce di un terzo l'errore sui ticker per cui non si hanno "
                     "prezzi reali. A zero il VRP torna costante.")
            st.caption(
                f"VRP applicato: {max(0.3, vrp + vrp_slope * math.log(0.12/0.20)):.2f} "
                f"al 12% di volatilita · {vrp:.2f} al 20% · "
                f"{max(0.3, vrp + vrp_slope * math.log(0.60/0.20)):.2f} al 60%"
            )
            c3, c4 = st.columns(2)
            with c3:
                tasso = st.number_input("Tasso privo di rischio", value=4.0, step=0.25,
                                        min_value=-2.0, max_value=25.0) / 100.0
            with c4:
                dividendo = st.number_input("Dividend yield", value=0.0, step=0.25,
                                            min_value=0.0, max_value=25.0) / 100.0
            strike_atm = st.checkbox(
                "Strike esattamente allo spot invece che a delta target", value=False,
                help="A delta 0.50 lo strike sta appena sopra lo spot, tanto piu' quanto "
                     "e' alta la volatilita'.")
            applica_cap = st.checkbox(
                "Applica il cap della covered call", value=True,
                help="A scadenza la call in-the-money viene riacquistata al valore "
                     "intrinseco. Disattivandolo si incassano i premi senza pagarne il "
                     "costo, ed e' esattamente il difetto della versione precedente.")

        with st.expander("Stima della volatilita", expanded=False):
            modello_vol = st.selectbox(
                "Stimatore", options=list(VOL_MODELS.keys()),
                format_func=lambda k: VOL_MODELS[k], index=list(VOL_MODELS).index("yang_zhang"))
            finestra = st.slider("Finestra corta (giorni)", 20, 180, 63, 1)
            finestra_lunga = st.slider("Finestra lunga (giorni)", 120, 756, 252, 6)
            blend = st.slider(
                "Peso della finestra corta", 0.0, 1.0, 0.70, 0.05,
                help="Sotto 1 si mescola la finestra lunga, che tiene conto del ritorno "
                     "della volatilita' verso la sua media.")
            lam = st.slider("Lambda EWMA", 0.80, 0.99, 0.94, 0.01,
                            disabled=(modello_vol != "ewma"))

        with st.expander("Rischio", expanded=False):
            var_conf = st.slider("Confidenza di VaR e CVaR", 0.90, 0.999, 0.99, 0.005)

        with st.expander("Grafici", expanded=False):
            log_equity = st.checkbox("Equity in scala logaritmica", value=False)
            variante_dettaglio = st.selectbox(
                "Variante nei grafici di dettaglio",
                options=list(VARIANTS.keys()),
                format_func=lambda k: VARIANTS[k]["label"], index=1)
            st.caption("Quali grafici produrre")
            flags = {
                "mostra_grafico_1": st.checkbox("Confronto delle equity", value=True),
                "mostra_grafico_pnl": st.checkbox("Utile netto dei versamenti", value=True),
                "mostra_grafico_verdetto": st.checkbox("Verdetto contro il Buy & Hold", value=True),
                "mostra_grafici_abc": st.checkbox("Valore e drawdown per variante", value=True),
                "mostra_grafico_underwater": st.checkbox("Drawdown a confronto", value=True),
                "mostra_grafico_5": st.checkbox("Acquisti Buy-The-Dip", value=True),
                "mostra_grafico_6": st.checkbox("Drawdown settimanale e filtro", value=True),
                "mostra_grafico_rend_annuali": st.checkbox("Rendimenti annuali", value=True),
                "mostra_grafico_composizione": st.checkbox("Composizione annuale", value=True),
                "mostra_grafico_premio": st.checkbox("Premio stimato e volatilita", value=True),
                "mostra_grafico_strike": st.checkbox("Prezzo e strike", value=True),
                "mostra_grafici_addizionali": st.checkbox(
                    "Heatmap, distribuzioni, rolling, rischio/rendimento", value=True),
            }

        esegui = st.button("Esegui il backtest", type="primary", **LARGO)
        # Marcatore di versione: serve a capire a colpo d'occhio se il deploy ha
        # davvero preso il codice nuovo.
        st.caption(f"kq_btd_cc {VERSIONE} · Streamlit {st.__version__}")

    params: Dict[str, Any] = {
        "ticker": ticker or "BTC-USD.CC",
        "start_date": data_inizio.strftime("%Y-%m-%d"),
        "end_date": data_fine.strftime("%Y-%m-%d") if fine_manuale else None,
        "capitale_iniziale": float(capitale),
        "capitale_addizionale": float(capitale_add),
        "boost_pct": float(boost),
        "btd_cap_annuo_pct": float(tetto),
        "btd_dd_weekly_limit": float(limite_dd),
        "btd_execution": "open" if esecuzione.startswith("Apertura") else "close",
        "strike_mode": "atm_spot" if strike_atm else "delta",
        "applica_cap": bool(applica_cap),
        "vol_model": modello_vol,
        "vol_window": int(finestra),
        "vol_long_window": int(finestra_lunga),
        "vol_blend": float(blend),
        "ewma_lambda": float(lam),
        "idle_cash_rate": float(idle),
        "debit_cash_rate": float(debito),
        "var_confidence": float(var_conf),
        "premium_model": PremiumModel(vrp=float(vrp), vrp_slope=float(vrp_slope),
                                      target_delta=float(delta_target),
                                      r=float(tasso), q=float(dividendo)),
    }
    prefs = {**PREFERENZE_DEFAULT, **flags,
             "variante_dettaglio": variante_dettaglio, "log_equity": bool(log_equity)}
    return params, prefs, esegui


# ---------------------------------------------------------------------------
# Esecuzione
# ---------------------------------------------------------------------------
def esegui_backtest(params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = costruisci_config(params)
    dati = carica_serie(cfg.ticker, cfg.start_date, cfg.end_date)
    risultato = run_backtest(dati["mensile"],
                             dati.get("settimanale"), dati.get("giornaliero"), cfg)
    risultato.setdefault("warnings", []).extend(dati.get("avvisi", []))
    return risultato


def costruisci_export(risultato: Dict[str, Any],
                      calibrazione: Optional[Dict[str, Any]]) -> Tuple[bytes, str]:
    """Serializza il backtest una sola volta per esecuzione.

    Le schede vengono ridisegnate a ogni interazione: senza questa memoria il
    JSON (spesso centinaia di kB) verrebbe ricostruito a ogni click.
    """
    chiave = (st.session_state.get("run_id", 0),
              (calibrazione or {}).get("file_sorgente"),
              (calibrazione or {}).get("metriche", {}).get("vrp_calibrato"))
    memo = st.session_state.get("_export_memo")
    if memo and memo[0] == chiave:
        return memo[1], memo[2]
    esporta = build_export(risultato, calibrazione=calibrazione)
    blob = export_json_bytes(esporta)
    st.session_state["_export_memo"] = (chiave, blob, esporta["schema"])
    return blob, esporta["schema"]


# ---------------------------------------------------------------------------
# Schede
# ---------------------------------------------------------------------------
def scheda_sintesi(risultato: Dict[str, Any], figure: Dict[str, Any]) -> None:
    varianti = risultato["varianti"]
    reinvest = varianti.get("premi_reinvest", {}).get("metrics", {})
    cash = varianti.get("premi_cash", {}).get("metrics", {})

    kpi_cards([
        ("Utile netto — Reinvest", fmt_currency_compact(reinvest.get("pnl_netto")),
         f"su {fmt_currency_compact(reinvest.get('versamenti_totali'))} versati",
         segno_di(reinvest.get("pnl_netto"))),
        ("CAGR — Reinvest", fmt_pct(reinvest.get("cagr")),
         f"Buy & Hold {fmt_pct(reinvest.get('bh_cagr'))}",
         segno_di(reinvest.get("extra_cagr_vs_bh"))),
        ("Max drawdown — Cash", fmt_pct(cash.get("max_dd_pct")),
         f"Buy & Hold {fmt_pct(cash.get('bh_max_dd_pct'))}",
         segno_di(cash.get("riduzione_dd_vs_bh"))),
        ("Sharpe — Cash", fmt_num(cash.get("sharpe")),
         f"Buy & Hold {fmt_num(cash.get('bh_sharpe'))}", None),
        ("Premio medio stimato", fmt_pct(cash.get("premio_pct_medio")),
         "del prezzo del sottostante, al mese", None),
        ("Call in-the-money", f"{cash.get('mesi_call_assegnata', 0)}/{cash.get('mesi', 0)}",
         "mesi in cui il cap ha morso", None),
    ])

    dd_cash = cash.get("riduzione_dd_vs_bh")
    extra_re = reinvest.get("extra_cagr_vs_bh")
    verdetti = []
    if dd_cash is not None:
        verdetti.append(
            f"con i premi tenuti in cassa il drawdown massimo e' "
            f"<b>{'inferiore' if dd_cash > 0 else 'superiore'} del {abs(dd_cash):.0%}</b> "
            f"rispetto al Buy &amp; Hold")
    if extra_re is not None:
        verdetti.append(
            f"reinvestendo i premi il CAGR e' <b>{abs(extra_re):.1%} "
            f"{'sopra' if extra_re > 0 else 'sotto'}</b> il Buy &amp; Hold")
    if verdetti:
        nota("Sul periodo analizzato, " + " e ".join(verdetti) +
             ". Il confronto e' a parita' di versamenti: il Buy &amp; Hold riceve gli "
             "stessi soldi negli stessi mesi.")

    for chiave in ("verdetto_bh", "confronto_equity", "rendimenti_annuali"):
        if chiave in figure:
            grafico(figure[chiave], key=f"sintesi_{chiave}")


def scheda_rischio(figure: Dict[str, Any]) -> None:
    for chiave in ("pnl_netto", "underwater", "eq_dd_no_premi", "eq_dd_cash",
                   "eq_dd_reinvest", "rischio_rendimento", "durata_dd"):
        if chiave in figure:
            grafico(figure[chiave], key=f"rischio_{chiave}")


def scheda_opzione(risultato: Dict[str, Any], figure: Dict[str, Any]) -> None:
    fonte = risultato.get("mercato", {}).get("vol_source", "n.d.")
    cfg = risultato["config"]
    pm = cfg.get("premium_model", {})
    nota(
        f"Premio ricostruito con Black-Scholes su strike a delta "
        f"{pm.get('target_delta', 0.5):.2f}. La volatilita' implicita e' stimata come "
        f"volatilita' realizzata &times; VRP, con VRP {pm.get('vrp', 1.0):.2f} al 20% di "
        f"volatilita e pendenza {pm.get('vrp_slope', 0.0):+.3f}, su dati "
        f"{html.escape(str(fonte))} e sempre noti <b>prima</b> dell'inizio del mese. "
        f"L'incasso in valuta e' una percentuale del prezzo corrente, quindi cambia ogni mese."
    )
    cash = risultato["varianti"].get("premi_cash", {}).get("metrics", {})

    rif = riferimenti.riferimento(str(cfg.get("ticker", "")))
    verdetto = riferimenti.giudizio(cash.get("premio_pct_medio"), rif)
    if verdetto:
        st.info(verdetto)

    finanziamento = cash.get("finanziamento_massimo") or 0.0
    if finanziamento > 0.05 * float(cfg.get("capitale_iniziale", 1)):
        st.warning(
            f"Il riacquisto delle call in-the-money ha portato il conto a debito fino a "
            f"{fmt_currency_compact(finanziamento)} per {cash.get('mesi_a_debito', 0)} mesi "
            f"({fmt_pct(finanziamento / float(cfg.get('capitale_iniziale', 1)), 0)} del capitale "
            f"fisso). Sul finanziamento e' applicato il "
            f"{cfg.get('debit_cash_rate', 0):.1%} annuo impostato nella sidebar. "
            f"Se preferisci evitarlo, tieni i premi in cassa invece di reinvestirli, "
            f"oppure vendi la call a un delta piu' basso."
        )
    for chiave in ("premio", "prezzo_strike", "composizione_annuale"):
        if chiave in figure:
            grafico(figure[chiave], key=f"opzione_{chiave}")

    with st.expander("Premi reali misurati su call ATM mensili", expanded=False):
        st.caption(
            "Ricavati da 1.666 vendite effettive di call at-the-money a scadenza mensile. "
            "Servono a verificare che il premio stimato sia nell'ordine di grandezza giusto "
            "per il sottostante: come si vede, una percentuale fissa non puo' andare bene "
            "per tutti."
        )
        tabella = pd.DataFrame([
            {"Sottostante": f"{k} — {v['descrizione']}",
             "Operazioni": v["n"], "Periodo": f"{v['dal']} → {v['al']}",
             "Premio mediano": fmt_pct(v["premio_mediano"]),
             "Intervallo 5-95%": f"{v['premio_p05']:.2%} – {v['premio_p95']:.2%}",
             "Vol. implicita mediana": fmt_pct(v["iv_mediana"], 1)}
            for k, v in riferimenti.PREMI_REALI.items()
        ])
        st.dataframe(tabella, hide_index=True, **LARGO)


def scheda_btd(risultato: Dict[str, Any], figure: Dict[str, Any]) -> None:
    cfg = risultato["config"]
    cap0 = float(cfg.get("capitale_iniziale", 0))
    nota(
        f"Il segnale scatta quando il mese precedente chiude in negativo. Ogni acquisto vale "
        f"l'entita' del calo applicata al capitale iniziale <b>piu' il BTD Boost</b>, pari al "
        f"{cfg.get('boost_pct', 0):.1%} del capitale iniziale "
        f"({fmt_currency_compact(cap0 * float(cfg.get('boost_pct', 0)))} per ogni acquisto). "
        f"Il cumulato dell'anno non puo' superare il "
        f"{cfg.get('btd_cap_annuo_pct', 1):.0%} del capitale iniziale "
        f"({fmt_currency_compact(cap0 * float(cfg.get('btd_cap_annuo_pct', 1)))}); se il tetto "
        f"taglia un acquisto, calo e boost si riducono in proporzione. Esecuzione "
        f"{'all&#39;apertura' if cfg.get('btd_execution') == 'open' else 'alla chiusura'} del mese."
    )
    for chiave in ("btd", "dd_settimanale"):
        if chiave in figure:
            grafico(figure[chiave], key=f"btd_{chiave}")


def scheda_distribuzioni(figure: Dict[str, Any]) -> None:
    presenti = [c for c in ("heatmap", "distribuzione", "rolling") if c in figure]
    if not presenti:
        st.info("Attiva i grafici addizionali nella sidebar per popolare questa scheda.")
        return
    for chiave in presenti:
        grafico(figure[chiave], key=f"distr_{chiave}")


COLONNE_MONITOR = [
    ("mese", "Mese"), ("open", "Apertura"), ("close", "Chiusura"),
    ("rendimento_mese", "Rend. mese"), ("stato_btd", "Segnale"),
    ("btd_quota_calo", "BTD dal calo"), ("btd_quota_boost", "BTD dal boost"),
    ("btd_importo", "BTD totale"), ("btd_residuo_anno", "Residuo tetto"),
    ("quote_coperte", "Quote coperte"), ("quote_extra", "Quote extra"),
    ("strike", "Strike venduto"), ("premio_pct", "Premio %"), ("premio", "Premio incassato"),
    ("intrinseco_pagato", "Intrinseco pagato"), ("netto_opzione", "Netto opzione"),
    ("cassa", "Cassa"), ("valore_portafoglio", "Valore conto"),
    ("versamento_mese", "Versato"), ("pnl_netto", "Utile netto"),
    ("dd_valore", "Drawdown"),
]

def scheda_anno_corrente(risultato: Dict[str, Any], variante: str) -> None:
    dett = dettaglio_anno(risultato, variante)
    if not dett:
        st.info("Nessun dato disponibile per questa variante.")
        return
    r = dett["riepilogo"]
    anno = dett["anno"]

    anni = sorted(risultato["varianti"][variante]["monthly"]["anno"].unique())
    c1, c2 = st.columns([1, 3])
    with c1:
        scelto = st.selectbox("Anno", anni, index=len(anni) - 1, key="anno_monitor")
    if int(scelto) != anno:
        dett = dettaglio_anno(risultato, variante, int(scelto))
        r, anno = dett["riepilogo"], dett["anno"]

    st.markdown(f"#### Anno {anno} — {VARIANTS[variante]['label']}")
    kpi_cards([
        ("Risultato dell'anno", fmt_currency_compact(r["risultato_anno"]),
         f"{fmt_pct(r['twr_anno'], 1)} time-weighted", segno_di(r["risultato_anno"])),
        ("Premi incassati", fmt_currency_compact(r["premi_incassati"]),
         f"intrinseco pagato {fmt_currency_compact(r['intrinseco_pagato'])}", "pos"),
        ("Netto opzioni", fmt_currency_compact(r["netto_opzioni"]),
         f"call ITM in {r['mesi_call_itm']} mesi su {r['mesi_trascorsi']}",
         segno_di(r["netto_opzioni"])),
        ("BTD investito", fmt_currency_compact(r["btd_investito"]),
         f"{r['btd_numero']} acquisti · residuo {fmt_currency_compact(r['btd_residuo'])}", None),
        ("di cui boost", fmt_currency_compact(r["btd_da_boost"]),
         f"calo {fmt_currency_compact(r['btd_da_calo'])}", None),
        ("Valore del conto", fmt_currency_compact(r["valore_conto"]),
         f"posizione {fmt_currency_compact(r['valore_posizione'])} + cassa "
         f"{fmt_currency_compact(r['cassa'])}", None),
    ])

    dettagli = [
        f"capitale fisso impiegato {fmt_currency_compact(r['capitale_fisso'])}",
        f"quote coperte {fmt_num(r['quote_coperte'], 4)}",
        f"quote extra {fmt_num(r['quote_extra'], 4)}",
        f"tetto annuo BTD {fmt_currency_compact(r['btd_tetto'])}",
    ]
    dettagli.insert(1, f"BTD Boost {fmt_currency_compact(r['boost_per_acquisto'])} per acquisto")
    if r["capitale_addizionale"]:
        dettagli.insert(1, "capitale addizionale non coperto "
                           f"{fmt_currency_compact(r['capitale_addizionale'])}")
    if r["versamenti"]:
        dettagli.append(f"versato nell'anno {fmt_currency_compact(r['versamenti'])}")
    if r["segnali_bloccati"]:
        dettagli.append(f"{r['segnali_bloccati']} segnali bloccati dal filtro")
    nota(" · ".join(dettagli))

    # ---------------- Piano del mese prossimo ----------------
    piano = piano_prossimo_mese(risultato, variante)
    if piano and int(scelto) == int(risultato["varianti"][variante]["monthly"]["anno"].iloc[-1]):
        st.markdown("#### Cosa fare il mese prossimo")
        if piano["reset_annuale"]:
            st.warning(
                f"**{piano['mese']} — reset annuale.** Liquidare tutta la posizione alla "
                f"chiusura di dicembre e reimpiegare "
                f"{fmt_currency_compact(piano['capitale_da_impiegare'])} all'apertura di "
                f"gennaio. Il tetto BTD e il budget del boost ripartono da zero."
            )
        else:
            righe = []
            if piano["segnale_btd"] and piano["btd_importo"] > 0:
                righe.append(
                    f"**Acquisto BTD di {fmt_currency_compact(piano['btd_importo'])}** "
                    f"all'apertura del mese: {fmt_currency_compact(piano['btd_quota_calo'])} "
                    f"per il calo del {fmt_pct(abs(piano['rendimento_ultimo_mese']), 1)} e "
                    f"{fmt_currency_compact(piano['btd_quota_boost'])} di boost. "
                    f"Dopo l'acquisto resteranno "
                    f"{fmt_currency_compact(piano['btd_residuo_anno'] - piano['btd_importo'])} "
                    f"sotto il tetto annuo."
                )
            elif piano["btd_bloccato"]:
                righe.append(
                    f"Segnale BTD presente ma **bloccato dal filtro**: il drawdown "
                    f"settimanale e' {fmt_pct(piano['dd_weekly'], 1)}."
                )
            elif piano["segnale_btd"]:
                righe.append("Segnale BTD presente ma **il tetto annuo e' esaurito**.")
            else:
                righe.append(
                    f"**Nessun acquisto BTD**: l'ultimo mese ha chiuso a "
                    f"{fmt_pct(piano['rendimento_ultimo_mese'], 1)}."
                )
            if piano.get("premio_pct"):
                righe.append(
                    f"**Vendere la call** su {fmt_num(piano['quote_coperte'], 4)} quote "
                    f"con strike indicativo {fmt_num(piano['strike_indicativo'], 2)} "
                    f"({fmt_pct(piano['strike_indicativo'] / piano['prezzo_riferimento'] - 1, 1)} "
                    f"sopra il prezzo di riferimento {fmt_num(piano['prezzo_riferimento'], 2)}), "
                    f"premio atteso {fmt_pct(piano['premio_pct'])} dello spot pari a circa "
                    f"{fmt_currency_compact(piano['premio_atteso'])} "
                    f"(volatilita stimata {fmt_pct(piano['sigma_stimata'], 1)}, "
                    f"VRP {fmt_num(piano['vrp_applicato'])})."
                )
            st.info(f"**{piano['mese']}** — " + "  \n".join(righe))
            st.caption(
                "Segnale e volatilita' sono gia' determinati dalla chiusura del mese scorso. "
                "Lo strike e le quote si ricalcolano sul prezzo di apertura effettivo."
            )

    # ---------------- Tabella mese per mese ----------------
    st.markdown("#### Dettaglio mese per mese")
    g = dett["mesi"].copy()
    g["mese"] = [MESI_IT[d.month - 1].capitalize() for d in g.index]
    g["stato_btd"] = [
        "bloccato" if (b and s) else ("acquisto" if i > 0 else ("segnale" if s else "—"))
        for s, b, i in zip(g["segnale_btd"], g["btd_bloccato"], g["btd_importo"])
    ]
    vista = pd.DataFrame({etichetta: g[col] for col, etichetta in COLONNE_MONITOR
                          if col in g.columns})
    valuta = ["Apertura", "Chiusura", "BTD dal calo", "BTD dal boost", "BTD totale",
              "Residuo tetto", "Strike venduto", "Premio incassato", "Intrinseco pagato",
              "Netto opzione", "Cassa", "Valore conto", "Versato", "Utile netto", "Drawdown"]
    for c in valuta:
        if c in vista.columns:
            vista[c] = vista[c].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}")
    for c in ("Rend. mese", "Premio %"):
        if c in vista.columns:
            vista[c] = vista[c].map(lambda v: "—" if pd.isna(v) else f"{v * 100:,.2f}%")
    for c in ("Quote coperte", "Quote extra"):
        if c in vista.columns:
            vista[c] = vista[c].map(lambda v: f"{v:,.4f}")
    st.dataframe(vista.set_index("Mese"), **LARGO)

    csv = dett["mesi"].to_csv().encode("utf-8")
    st.download_button(f"Scarica il {anno} in CSV", data=csv,
                       file_name=f"boosted_covered_call_{anno}_{variante}.csv",
                       mime="text/csv")


def scheda_dati(risultato: Dict[str, Any], calibrazione: Optional[Dict[str, Any]]) -> None:
    st.markdown("#### Metriche a confronto")
    tb = metrics_table(risultato["varianti"])
    if not tb.empty:
        vista = pd.DataFrame(
            {col: [format_value(k, tb.loc[k, col]) for k in tb.index] for col in tb.columns},
            index=[ETICHETTE.get(k, k) for k in tb.index],
        )
        st.dataframe(vista, **LARGO)

    st.markdown("#### Dettaglio per anno")
    scelta = st.selectbox("Variante", list(VARIANTS.keys()),
                          format_func=lambda k: VARIANTS[k]["label"], index=1, key="anno_var")
    y = risultato["varianti"][scelta]["yearly"]
    if not y.empty:
        vista_y = y.copy()
        for c in ("rendimento_sottostante", "twr_anno"):
            vista_y[c] = vista_y[c].map(lambda v: fmt_pct(v, 1))
        for c in ("premi_incassati", "intrinseco_pagato", "netto_opzioni", "btd_investito",
                  "versamenti", "capitale_medio_impiegato", "valore_fine_anno", "risultato_anno"):
            vista_y[c] = vista_y[c].map(lambda v: f"${v:,.0f}")
        vista_y.columns = [c.replace("_", " ").capitalize() for c in vista_y.columns]
        st.dataframe(vista_y, **LARGO)

    st.markdown("#### Serie mensile")
    mdf = risultato["varianti"][scelta]["monthly"]
    colonne = st.multiselect(
        "Colonne", options=list(mdf.columns), key="col_mensili",
        default=[c for c in ["close", "rendimento_mese", "segnale_btd", "btd_importo",
                             "sigma_stimata", "strike", "premio_pct", "premio",
                             "intrinseco_pagato", "valore_portafoglio", "versamenti_cum",
                             "pnl_netto", "twr_mese"] if c in mdf.columns])
    if colonne:
        st.dataframe(mdf[colonne], height=420, **LARGO)

    st.markdown("#### Scarica il backtest")
    nota("Il file JSON contiene parametri, equity, serie mensile completa di ogni variante, "
         "tabelle annuali, flussi di cassa, metriche, volatilita' stimata, la calibrazione "
         "del premio e un dizionario che spiega ogni campo.")
    blob, schema = costruisci_export(risultato, calibrazione)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.download_button(
            "Scarica JSON", data=blob, file_name=nome_file_export(risultato["config"]),
            mime="application/json", type="primary", **LARGO)
    with c2:
        st.caption(f"{len(blob) / 1024:,.0f} kB — schema {schema}")


def scheda_calibrazione(risultato: Optional[Dict[str, Any]], params: Dict[str, Any]) -> None:
    st.markdown("#### Calibrazione del premio sui prezzi reali")
    nota(
        "Carica un file con i prezzi reali delle call vicine al delta obiettivo sul "
        "sottostante che stai studiando. Per ogni riga il modello ricostruisce il premio "
        "usando <b>solo</b> la volatilita' realizzata nota prima di quella data, senza mai "
        "vedere la volatilita' implicita, e cerca il coefficiente che minimizza l'errore. "
        "Servono almeno: data, prezzo del sottostante, prezzo dell'opzione (mid oppure bid "
        "e ask) e giorni a scadenza. Delta, strike e IV, se ci sono, vengono usati."
    )
    file = st.file_uploader("File dei prezzi reali", type=["csv", "txt", "xlsx"],
                            key="file_calibrazione")
    if file is None:
        return
    if risultato is None:
        st.warning("Esegui prima il backtest: serve la volatilita' del sottostante.")
        return

    try:
        grezzo, e_optionlab = calib.carica_file_opzioni(file, nome=file.name)
    except Exception as e:
        st.error(f"File non leggibile: {e}")
        return
    if e_optionlab:
        st.success(
            "Riconosciuto un export OptionLAB: le colonne sono associate automaticamente "
            "(data di apertura, sottostante all'apertura, premio incassato) e sono state "
            "tenute solo le vendite di call."
        )

    st.caption(f"{len(grezzo):,} righe · colonne: {', '.join(map(str, grezzo.columns[:14]))}"
               + (" ..." if len(grezzo.columns) > 14 else ""))
    with st.expander("Anteprima", expanded=False):
        st.dataframe(grezzo.head(20), **LARGO)

    suggerita = calib.suggerisci_mappatura(grezzo)
    st.markdown("**Associazione delle colonne**")
    opzioni = ["(nessuna)"] + [str(c) for c in grezzo.columns]
    campi = ["data", "spot", "mid", "bid", "ask", "dte", "scadenza", "strike", "delta", "tipo", "iv"]
    mappatura: Dict[str, Optional[str]] = {}
    colonne_ui = st.columns(4)
    for i, campo in enumerate(campi):
        with colonne_ui[i % 4]:
            corrente = suggerita.get(campo)
            idx = opzioni.index(str(corrente)) if corrente and str(corrente) in opzioni else 0
            scelta = st.selectbox(campo, opzioni, index=idx, key=f"map_{campo}")
            mappatura[campo] = None if scelta == "(nessuna)" else scelta

    c1, c2, c3 = st.columns(3)
    with c1:
        tolleranza = st.slider("Tolleranza sul delta", 0.02, 0.30, 0.10, 0.01)
        obiettivo = st.selectbox(
            "Obiettivo della calibrazione", list(calib.OBIETTIVI.keys()),
            format_func=lambda k: calib.OBIETTIVI[k], index=0,
            help="Misurato su 1.666 premi reali: 'livello' azzera lo scarto sul premio "
                 "medio incassato, che e' cio' che determina il risultato di un backtest; "
                 "gli altri due riducono di poco l'errore mese per mese ma lasciano una "
                 "sottostima sistematica dal 10% al 27%.")
    with c2:
        fit_add = st.checkbox("Calibra anche un addendo di volatilita", value=False)
    with c3:
        confronta = st.checkbox("Confronta tutti gli stimatori di volatilita", value=True)

    base = PremiumModel(**{**params["premium_model"].to_dict(), "vrp_add": 0.0})

    if st.button("Calibra", type="primary"):
        try:
            oss = calib.prepara_osservazioni(
                grezzo, mappatura,
                delta_target=float(params["premium_model"].target_delta),
                delta_tolleranza=float(tolleranza))
        except ValueError as e:
            st.error(str(e))
            return
        if oss.empty:
            st.error("Nessuna osservazione valida dopo i filtri. Allarga la tolleranza sul "
                     "delta o controlla l'associazione delle colonne.")
            return

        oss = calib.aggancia_volatilita(oss, risultato["mercato"]["vol"])
        if len(oss) < 3:
            st.error("Meno di tre osservazioni hanno una volatilita' realizzata disponibile. "
                     "Verifica che il periodo del file ricada dentro quello del backtest.")
            return

        fit = calib.calibra_vrp(oss, base=base, fit_addendo=fit_add,
                                obiettivo=obiettivo)
        if not fit.get("ok"):
            st.error(fit.get("errore", "Calibrazione fallita."))
            return

        # Il risultato vive in sessione: le schede si ridisegnano a ogni click e
        # senza questo la calibrazione sparirebbe alla prima interazione.
        st.session_state["fit_calibrazione"] = fit
        st.session_state["fit_osservazioni"] = oss
        st.session_state["calibrazione"] = calib.pacchetto_export(fit, nome_file=file.name)

    fit = st.session_state.get("fit_calibrazione")
    if not fit:
        return
    oss = st.session_state.get("fit_osservazioni")
    m = fit["metriche"]
    kpi_cards([
        ("VRP calibrato", f"{m['vrp_calibrato']:.3f}", f"su {m['n']} osservazioni", None),
        ("Errore medio", f"{m['mae'] * 100:.2f} pt", "punti percentuali di spot", None),
        ("Errore quadratico", f"{m['rmse'] * 100:.2f} pt", "RMSE in punti di spot", None),
        ("Distorsione", f"{m['bias'] * 100:+.2f} pt",
         "positiva = il modello sovrastima", segno_di(-abs(m['bias']))),
        ("R quadro", f"{m['r2']:.3f}" if m.get("r2") is not None else "n.d.",
         "quota di variabilita spiegata", None),
        ("Premio medio reale", f"{m['premio_medio_reale']:.2%}",
         f"stimato {m['premio_medio_stimato']:.2%}", None),
        ("Scarto sul livello",
         f"{m['premio_medio_stimato'] / m['premio_medio_reale'] - 1:+.1%}",
         "quanto il backtest incassa in piu o in meno", None),
    ])
    grafico(charts.fig_calibrazione(fit), key="calib_fig")

    if confronta and oss is not None:
        st.markdown("**Quale stimatore di volatilita descrive meglio i prezzi reali**")
        tabella = calib.confronta_modelli_vol(
            oss.drop(columns=["sigma_realizzata"]),
            risultato.get("_giornaliero"), risultato["config"], base=base,
            obiettivo=obiettivo)
        if tabella is not None and not tabella.empty:
            vista = tabella.copy()
            vista["vrp"] = vista["vrp"].map(lambda v: f"{v:.3f}")
            for c in ("mae", "rmse", "bias"):
                vista[c] = vista[c].map(lambda v: f"{v * 100:+.3f} pt")
            vista["r2"] = vista["r2"].map(lambda v: f"{v:.3f}")
            if "mape" in vista.columns:
                vista["mape"] = vista["mape"].map(lambda v: f"{v:.0%}")
            vista = vista.rename(columns={
                "modello": "Stimatore", "vrp": "VRP", "mae": "MAE", "rmse": "RMSE",
                "bias": "Distorsione", "mape": "Errore medio", "r2": "R quadro", "n": "Oss."})
            st.dataframe(vista.drop(columns=["chiave"]), hide_index=True,
                         **LARGO)
            st.caption("Ordinati per errore quadratico crescente: il primo e' quello da "
                       "impostare nella sidebar.")
        else:
            st.caption("Confronto non disponibile: servono i dati giornalieri del sottostante.")

    sigma_med = float(oss["sigma_realizzata"].median()) if oss is not None else None
    effettivo = fit["modello"].vrp_effettivo(sigma_med) if sigma_med else None
    st.success(
        f"**VRP calibrato: {m['vrp_calibrato']:.3f}** — e' il valore da mettere nello slider "
        f"*Volatility risk premium a volatilita 20%* nella sidebar, poi rilancia il backtest."
        + (f" Alla volatilita mediana di questo sottostante ({sigma_med:.0%}) il VRP "
           f"effettivamente applicato diventa {effettivo:.3f}, per effetto della pendenza."
           if effettivo else "")
        + " La calibrazione e' gia' inclusa nel JSON di export."
    )


# ---------------------------------------------------------------------------
# Corpo
# ---------------------------------------------------------------------------
st.title("Boosted Covered Call — Studio Mensile")
st.caption("Capitale fisso annuo coperto da una call mensile a delta 0.50, Buy-The-Dip "
           "potenziato sui mesi negativi, liquidazione e reset a fine anno.")

params, prefs, esegui = sidebar()

if not ha_api_key():
    st.warning(
        "Manca la chiave EODHD. Su Streamlit Cloud vai in Settings → Secrets e aggiungi "
        "`EODHD_API_KEY = \"la-tua-chiave\"`; in locale puoi usare la variabile d'ambiente "
        "omonima oppure `.streamlit/secrets.toml`."
    )
    st.stop()

if esegui:
    with st.spinner("Scarico i dati ed eseguo il backtest…"):
        try:
            st.session_state["risultato"] = esegui_backtest(params)
            st.session_state["params"] = params
            st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1
        except (ChiaveMancante, DatiNonDisponibili) as e:
            st.session_state.pop("risultato", None)
            st.error(str(e))
        except Exception as e:
            st.session_state.pop("risultato", None)
            st.error("Errore durante l'esecuzione del backtest.")
            st.exception(e)

risultato = st.session_state.get("risultato")

if risultato is None:
    st.info("Imposta i parametri nella sidebar e premi **Esegui il backtest**.")
    with st.expander("Cosa fa questa strategia", expanded=True):
        st.markdown(
            """
Ogni anno si impiega lo **stesso capitale fisso**, deciso in partenza, comprando il
sottostante all'apertura di gennaio. Su quelle quote si vende ogni mese una call a
delta 0.50 con scadenza a fine mese: si incassa un premio pari a una percentuale del
prezzo corrente e, se a scadenza la call e' in-the-money, la si riacquista al valore
intrinseco. E' cosi' che il cap sull'upside costa davvero, mese dopo mese.

Quando il sottostante chiude un mese in negativo scatta il **Buy-The-Dip**: si investe
l'entita' del calo applicata al capitale fisso, maggiorata di un boost, fino a un tetto
annuo. Queste quote extra restano scoperte.

A fine anno si **liquida tutto** e si riparte dallo stesso capitale fisso. L'eccedenza
resta come cassa; se manca capitale si versa la differenza — e quella e' un versamento,
non un utile. Ogni euro entrato dall'esterno viene tracciato, cosi' l'utile mostrato e'
al netto dei versamenti e i rendimenti sono time-weighted.
            """
        )
    st.stop()

if not risultato.get("ok"):
    st.error(risultato.get("errore", "Backtest non riuscito."))
    st.stop()

for avviso in risultato.get("warnings", []):
    st.warning(avviso)

with st.spinner("Costruisco i grafici…"):
    costruite = costruisci_figure(
        risultato, prefs,
        variante_dettaglio=str(prefs.get("variante_dettaglio", "premi_cash")),
        log_equity=bool(prefs.get("log_equity", False)))
figure = costruite["figure"]
for errore in costruite["errori"]:
    st.warning(f"Grafico non disponibile — {errore}")

cfg = risultato["config"]
periodo_a = risultato["varianti"]["premi_cash"]["metrics"].get("periodo_inizio", "—")
periodo_b = risultato["varianti"]["premi_cash"]["metrics"].get("periodo_fine", "—")
st.caption(
    f"**{html.escape(str(cfg.get('ticker')))}** · dal {periodo_a} al {periodo_b} · "
    f"capitale fisso ${cfg.get('capitale_iniziale', 0):,.0f} · "
    f"boost {cfg.get('boost_pct', 0):.1%} · "
    f"volatilita da {risultato['mercato'].get('vol_source', 'n.d.')}"
    + ("" if cfg.get("applica_cap") else " · **cap della call disattivato**")
)

schede = st.tabs(["Sintesi", "Anno in corso", "Equity e rischio", "Opzione e premio",
                  "Buy-The-Dip", "Distribuzioni", "Dati ed export", "Calibrazione premio"])
with schede[0]:
    scheda_sintesi(risultato, figure)
with schede[1]:
    scheda_anno_corrente(risultato, str(prefs.get("variante_dettaglio", "premi_cash")))
with schede[2]:
    scheda_rischio(figure)
with schede[3]:
    scheda_opzione(risultato, figure)
with schede[4]:
    scheda_btd(risultato, figure)
with schede[5]:
    scheda_distribuzioni(figure)
with schede[6]:
    scheda_dati(risultato, st.session_state.get("calibrazione"))
with schede[7]:
    scheda_calibrazione(risultato, params)
