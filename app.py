"""Boosted Covered Call — Studio Mensile e Settimanale.

Dashboard Streamlit: capitale fisso annuo coperto da una call venduta a delta
0.50, Buy-The-Dip potenziato dopo ogni periodo negativo, reset a fine anno.

Uno switch in cima alla sidebar sceglie il passo: mensile (l'originale) oppure
settimanale, dove tutto quello che si fa a fine mese si fa a fine settimana.
Il ciclo annuale non cambia mai. I testi seguono la cadenza scelta: `A()` li
riscrive al volo, cosi' il codice resta scritto una volta sola.

Il premio non e' piu' un numero da indovinare: viene stimato dalla volatilita'
realizzata del sottostante e puo' essere calibrato sui prezzi reali delle
opzioni caricati dall'utente.
"""
from __future__ import annotations

import datetime as dt
import html
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from kq_btd_cc import CSS, __version__ as VERSIONE
from kq_btd_cc import calibration as calib
from kq_btd_cc.cadenza import CADENZE, adatta, label as etichetta_cadenza, normalizza
from kq_btd_cc import charts
from kq_btd_cc.core import PREFERENZE_DEFAULT, costruisci_config, costruisci_figure
from kq_btd_cc.data_api import ChiaveMancante, DatiNonDisponibili, carica_serie, ha_api_key
from kq_btd_cc.engine import (VARIANTS, dettaglio_anno, piano_prossimo_mese,
                              run_backtest)
from kq_btd_cc.export import build_export, export_json_bytes, nome_file_export
from kq_btd_cc.metrics import etichette as etichette_metriche, format_value, metrics_table
from kq_btd_cc.pricing import PremiumModel
from kq_btd_cc import riferimenti
from kq_btd_cc.utils import fmt_currency_compact, fmt_num, fmt_pct
from kq_btd_cc.vol import VOL_MODELS

st.set_page_config(
    page_title="Boosted Covered Call",
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


def cadenza_corrente() -> str:
    """La cadenza da usare nei testi.

    Sono i RISULTATI mostrati a comandare: se si cambia lo switch senza rilanciare
    il backtest, le tabelle a schermo parlano ancora del passo con cui sono state
    calcolate. Prima del primo backtest vale la scelta fatta nella sidebar.
    """
    r = st.session_state.get("risultato")
    if isinstance(r, dict) and isinstance(r.get("config"), dict):
        return normalizza(r["config"].get("cadenza"))
    return normalizza(st.session_state.get("cadenza"))


def A(testo: str) -> str:
    """Il testo, scritto al mensile, riscritto per la cadenza in uso."""
    return adatta(testo, cadenza_corrente())

# Estremi del periodo selezionabile. 1970 copre tutto lo storico che EODHD puo'
# restituire su indici e azioni, bolla dot-com e crisi del 2008 comprese.
ANNO_MINIMO = 1970
OGGI = dt.date.today()

MESI_IT = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
           "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]


def _ultimo_giorno(anno: int, mese: int) -> dt.date:
    """Ultimo giorno del mese indicato."""
    return (dt.date(anno + (mese == 12), (mese % 12) + 1, 1) - dt.timedelta(days=1))


PREMIO_DEFAULT = PremiumModel()

# Tre sottostanti-tipo per far vedere subito l'effetto della taratura, con la
# volatilita' misurata sui prezzi reali di quei nomi.
ESEMPI_VOL = [(0.14, "un indice"), (0.30, "un titolo"), (0.60, "una crypto")]


# Configurazioni della stima di volatilita'. La predefinita e' quella risultata
# migliore sui premi reali di sei sottostanti; le altre due spostano il
# compromesso fra seguire i cambi di regime e non inseguire il rumore.
PRESET_VOL: Dict[str, Dict[str, Any]] = {
    "Predefinita": dict(
        vol_model="yang_zhang", vol_window=126, vol_long_window=504, vol_blend=0.60,
        descrizione="Guarda gli ultimi sei mesi, smorzati verso la media di due anni. "
                    "E' la taratura che ha vinto su tutti e sei i sottostanti testati. "
                    "Su un cambio di regime se ne accorge in 4 mesi, e il premio si muove "
                    "in media del 2,7% da un mese all'altro."),
    "Piu reattiva": dict(
        vol_model="yang_zhang", vol_window=63, vol_long_window=252, vol_blend=0.85,
        descrizione="Guarda gli ultimi tre mesi e li smorza poco. Su un cambio di regime "
                    "se ne accorge in 1 mese, ma il premio balla del 3,6% al mese con "
                    "salti fino all'80%."),
    "Piu stabile": dict(
        vol_model="yang_zhang", vol_window=252, vol_long_window=756, vol_blend=0.40,
        descrizione="Guarda un anno intero, appoggiandosi molto alla media di tre anni. "
                    "Premio regolare, si muove del 2,3% al mese e non salta mai oltre il "
                    "18%, ma su un cambio di regime ci mette 7 mesi ad adeguarsi."),
    "Manuale": dict(
        vol_model="yang_zhang", vol_window=126, vol_long_window=504, vol_blend=0.60,
        descrizione="Tutti i parametri dello stimatore, uno per uno."),
}


def _anteprima_premio(vrp: float, pendenza: float, delta: float) -> str:
    """Traduce i parametri del modello in premi leggibili."""
    pm = PremiumModel(vrp=vrp, vrp_slope=pendenza, target_delta=delta)
    pezzi = []
    for sigma, nome in ESEMPI_VOL:
        q = pm.quote(100.0, sigma, 30.0 / 365.0)
        pezzi.append(f"{q['premium_pct']:.2%} su {nome} ({sigma:.0%} di volatilita)")
    return "Con questa taratura una call mensile incassa: " + " · ".join(pezzi)


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

        # Lo switch sta in cima perche' decide il significato di tutto quello che
        # viene dopo: premio, segnale, acquisti sui cali e interessi cambiano passo.
        cadenza = st.radio(
            "Cadenza della strategia", options=list(CADENZE), key="cadenza",
            format_func=lambda c: CADENZE[c]["label"], horizontal=True, index=0,
            help="Il passo con cui si vende la call, si controlla il segnale e si "
                 "comprano i cali. Il ciclo annuale — capitale deciso a gennaio, "
                 "tutto liquidato a dicembre — resta identico in entrambe.")
        st.caption(CADENZE[cadenza]["descrizione"])

        def _a(testo: str) -> str:
            """I testi della sidebar seguono lo switch, non i risultati a schermo."""
            return adatta(testo, cadenza)

        with st.expander("Sottostante e periodo", expanded=True):
            ticker = st.text_input(
                "Ticker EODHD", value="BTC-USD.CC",
                help="Formato EODHD: BTC-USD.CC, ETH-USD.CC, SPY.US, AAPL.US...").strip()
            # Niente st.date_input qui: il calendario di Streamlit elenca nel menu
            # degli anni una finestra fissa di vent'anni, e ignora min_value. Con
            # max_value a oggi si ferma al 2007, tagliando fuori la bolla dot-com.
            # Il periodo si sceglie comunque per mese: il motore poi lo taglia
            # sulle barre della cadenza scelta.
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
                     "E' anche il riferimento percentuale degli acquisti BTD e del boost.")
            capitale_add = st.number_input(
                "Capitale addizionale annuale — non coperto", value=0, step=1_000, min_value=0,
                help="Entra una volta sola all'apertura di gennaio, insieme al capitale "
                     "iniziale, e compra sottostante che resta li' per tutto l'anno. La call "
                     "NON lo cappa: si tiene tutto il rialzo, e su di esso non si incassa "
                     "nessun premio. Viene liquidato a fine anno come il resto.")
            modo_label = st.radio(
                "Capitale rimesso al lavoro ogni gennaio",
                ["Sempre lo stesso importo", "Cresce insieme al conto"], index=0,
                help="Con l'importo fisso i profitti restano in cassa e la strategia non "
                     "compone: guadagna ogni anno piu' o meno la stessa cifra e la curva "
                     "cresce in linea retta e il capitale che lavora non aumenta mai. "
                     "Con la crescita gli utili tornano al lavoro, mantenendo la proporzione "
                     "fra parte coperta e parte scoperta, e non servono nuovi versamenti.")
            capitale_modo = ("fisso" if modo_label.startswith("Sempre") else "composto")
            riserva = st.slider(
                "Riserva liquida per gli acquisti sui cali", 0, 200, 75, 5,
                disabled=(capitale_modo == "fisso"),
                help="In percentuale del capitale impiegato a gennaio. Quando il capitale "
                     "cresce col conto, questa e' la parte che resta liquida per comprare "
                     "sui cali durante l'anno: senza, ogni Buy-The-Dip richiederebbe denaro "
                     "fresco. Se la riserva finisce, si compra meno.") / 100.0
            idle = st.slider(
                "Remunerazione della cassa non impiegata (annua)", 0.0, 6.0, 0.0, 0.25,
                help="Conta piu' di quanto sembri. Il reset annuale reimpiega solo il "
                     "capitale fisso e lascia ferma tutta la liquidita' accumulata: dopo "
                     "qualche anno puo' essere meta' del conto. A 0% quella meta' non rende "
                     "nulla per tutto il backtest. Metti il tasso che ti riconoscono davvero "
                     "sul saldo, o quello di un monetario se ce la parcheggi.") / 100.0
            debito = st.slider(
                "Costo del saldo a debito (annuo)", 0.0, 15.0, 6.0, 0.5,
                help="Quando il riacquisto della call a scadenza supera la liquidita' "
                     "disponibile, il conto va a debito contro le azioni in portafoglio. "
                     "Questo e' il tasso applicato a quel finanziamento.") / 100.0

        with st.expander("Buy-The-Dip", expanded=True):
            reinvesto_label = st.radio(
                _a("Quando reinvestire i premi (solo variante Reinvest)"),
                [_a("Subito, alla chiusura del mese"),
                 _a("Al prossimo acquisto sui cali")], index=0,
                help=_a("Sulla variante Reinvest. Subito: il risultato netto delle opzioni "
                        "compra quote alla chiusura di ogni mese, al prezzo che c'e' in quel "
                        "momento. Al prossimo acquisto sui cali: i premi restano in un conto "
                        "a parte finche' non scatta un BTD, e allora entrano tutti insieme e "
                        "AL LORDO, compreso quello del mese in corso, allo stesso prezzo del "
                        "BTD. L'intrinseco si paga dopo, a scadenza, e puo' mandare a debito "
                        "il conto delle opzioni: e' il prezzo di aver messo il premio al "
                        "lavoro subito. Quello che a dicembre non e' ancora rientrato viene "
                        "liquidato con il resto."))
            reinvesto_modo = ("al_btd" if "cali" in reinvesto_label else "subito")
            boost = st.slider(
                "BTD Boost", 0.0, 15.0, 5.0, 0.5,
                help="Percentuale del capitale iniziale che si aggiunge a OGNI acquisto BTD, "
                     "oltre alla quota legata all'entita' del calo. Eredita tutto dal BTD: "
                     "stesso momento e prezzo di acquisto, quote non coperte dalla call, "
                     "liquidazione a fine anno. Da non confondere con il "
                     "capitale addizionale annuale, che entra invece una volta sola.") / 100.0
            limite_dd = st.slider(
                "Sospendi il BTD sotto questo drawdown settimanale", -95.0, -20.0, -90.0, 5.0,
                help=_a("Se il drawdown settimanale dell'asset e' piu' profondo di questa "
                        "soglia, l'acquisto del mese viene saltato.")) / 100.0
            esecuzione = st.radio(
                _a("Esecuzione dell'acquisto"),
                [_a("Apertura del mese"), _a("Chiusura del mese")],
                index=0, horizontal=False,
                help="Il segnale e' noto alla chiusura del mese precedente: comprare "
                     "all'apertura e' la scelta eseguibile. La chiusura riproduce il "
                     "comportamento della versione precedente della dashboard.")

        with st.expander("Premio della call", expanded=True):
            st.caption("Il premio lo calcola il modello dalla volatilita' del sottostante. "
                       "Nella maggior parte dei casi qui non c'e' niente da toccare.")
            delta_target = st.slider(
                "Delta della call venduta", 0.20, 0.80, 0.50, 0.05,
                help="0.50 e' la call at-the-money classica della strategia. Piu' basso "
                     "significa strike piu' lontano: meno premio, ma la call finisce "
                     "in-the-money molto piu' di rado.")

            cal = st.session_state.get("calibrazione") or {}
            mod_cal = cal.get("modello_premio") or {}
            opzioni = ["Predefinito", "Manuale"]
            if mod_cal:
                opzioni.insert(1, "Calibrato sui prezzi reali")
            scelta = st.radio(
                "Taratura del premio", opzioni, index=1 if mod_cal else 0,
                help="'Predefinito' usa la taratura misurata su 1.666 vendite reali di call "
                     "ATM mensili: e' il punto di partenza giusto per un sottostante "
                     "qualunque. 'Calibrato' compare dopo aver caricato i prezzi reali nella "
                     "scheda Calibrazione premio. 'Manuale' apre i due parametri del modello.")

            if scelta == "Calibrato sui prezzi reali":
                vrp = float(mod_cal.get("vrp", PREMIO_DEFAULT.vrp))
                vrp_slope = float(mod_cal.get("vrp_slope", PREMIO_DEFAULT.vrp_slope))
                st.caption(f"Da {cal.get('file_sorgente', 'file caricato')} · "
                           f"{cal.get('metriche', {}).get('n', 0)} osservazioni")
            elif scelta == "Manuale":
                vrp = st.slider(
                    "Livello: quanto la volatilita implicita supera quella realizzata",
                    0.60, 2.00, PREMIO_DEFAULT.vrp, 0.01,
                    help="Riferito a un sottostante che oscilla del 20% annuo. Alzarlo "
                         "aumenta tutti i premi in proporzione. Sopra 1 significa che il "
                         "mercato paga le opzioni piu' di quanto il sottostante si muova.")
                vrp_slope = st.slider(
                    "Pendenza: quanto quel margine cala sui sottostanti volatili",
                    -0.40, 0.10, PREMIO_DEFAULT.vrp_slope, 0.005,
                    help="Sui prezzi reali il margine si assottiglia quando il sottostante "
                         "e' piu' agitato. A zero il rapporto resta uguale per tutti.")
            else:
                vrp, vrp_slope = PREMIO_DEFAULT.vrp, PREMIO_DEFAULT.vrp_slope

            st.caption(_anteprima_premio(vrp, vrp_slope, delta_target))
            c3, c4 = st.columns(2)
            with c3:
                tasso = st.number_input("Tasso privo di rischio", value=4.0, step=0.25,
                                        min_value=-2.0, max_value=25.0) / 100.0
            with c4:
                dividendo = st.number_input("Dividend yield", value=0.0, step=0.25,
                                            min_value=0.0, max_value=25.0) / 100.0
            filtro_label = st.radio(
                _a("Quando vendere la call"),
                [_a("Sempre, ogni mese"),
                 _a("Solo sotto il prezzo di carico (in perdita)"),
                 _a("Solo sopra il prezzo di carico (in guadagno)")], index=0,
                help=_a("Sempre: la posizione e' sempre coperta, com'e' stato finora. "
                        "Solo sotto: finche' si e' in guadagno la posizione resta scoperta "
                        "e si tiene tutto il rialzo, ma si torna a vendere proprio dopo una "
                        "discesa, quindi il cap morde sul rimbalzo. Solo sopra: se la call "
                        "viene assegnata si esce comunque in utile e nelle discese si resta "
                        "liberi di recuperare, ma si vende nelle salite, che e' dove "
                        "l'intrinseco costa di piu'. In entrambi i casi, alla pari si vende."))
            filtro_call = ("sotto_carico" if "sotto" in filtro_label
                           else "sopra_carico" if "sopra" in filtro_label else "sempre")
            carico_label = st.radio(
                "Prezzo di carico di riferimento",
                ["Quote coperte dalla call", "Intera posizione (media)"], index=0,
                disabled=(filtro_call == "sempre"),
                help=_a("Quote coperte: il prezzo a cui sono state comprate a gennaio, fisso "
                        "per tutto l'anno. Intera posizione: la media di tutto quello che si "
                        "ha in mano, acquisti sui cali e premi reinvestiti compresi, che "
                        "scende a ogni acquisto fatto piu' in basso e quindi rende piu' "
                        "difficile trovarsi 'sotto carico'."))
            carico_riferimento = ("medio" if "Intera" in carico_label else "coperte")
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
            st.caption("Quanto il modello guarda indietro per capire di quanto si muove il "
                       "sottostante. E' l'ingrediente da cui esce il premio.")
            nome_preset = st.radio(
                "Memoria del modello", list(PRESET_VOL.keys()), index=0,
                help="La predefinita e' quella che ha funzionato meglio su tutti e sei i "
                     "sottostanti di cui ho i prezzi reali delle opzioni. Le altre due "
                     "spostano il compromesso fra reattivita' e stabilita'.")
            preset = PRESET_VOL[nome_preset]
            st.caption(preset["descrizione"])

            if nome_preset == "Manuale":
                modello_vol = st.selectbox(
                    "Stimatore", options=list(VOL_MODELS.keys()),
                    format_func=lambda k: VOL_MODELS[k],
                    index=list(VOL_MODELS).index("yang_zhang"),
                    help="Yang-Zhang usa anche massimi, minimi e gap di apertura, quindi "
                         "stima la volatilita' con molto meno rumore del semplice "
                         "close-to-close.")
                finestra = st.slider("Finestra corta (giorni di borsa)", 20, 252, 126, 1)
                finestra_lunga = st.slider("Finestra lunga (giorni di borsa)", 120, 1260, 504, 6)
                blend = st.slider(
                    "Peso della finestra corta", 0.0, 1.0, 0.60, 0.05,
                    help="A 1 conta solo il periodo recente. Sotto 1 si mescola la finestra "
                         "lunga, che smorza gli errori di stima.")
                lam = st.slider("Lambda EWMA", 0.80, 0.99, 0.94, 0.01,
                                disabled=(modello_vol != "ewma"))
            else:
                modello_vol = preset["vol_model"]
                finestra = preset["vol_window"]
                finestra_lunga = preset["vol_long_window"]
                blend = preset["vol_blend"]
                lam = 0.94
                st.caption(f"{VOL_MODELS[modello_vol]} · finestra corta "
                           f"{finestra // 21} mesi, lunga {finestra_lunga // 252} anni · "
                           f"peso della corta {blend:.0%}")

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
                "mostra_grafico_giornaliero": st.checkbox(
                    "Valorizzazione giornaliera", value=True,
                    help="Rivaluta il conto ogni giorno di borsa invece che solo alla "
                         "chiusura del periodo: e' li' che si vedono i drawdown veri."),
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
        "cadenza": cadenza,
        "ticker": ticker or "BTC-USD.CC",
        "start_date": data_inizio.strftime("%Y-%m-%d"),
        "end_date": data_fine.strftime("%Y-%m-%d") if fine_manuale else None,
        "capitale_iniziale": float(capitale),
        "capitale_addizionale": float(capitale_add),
        "capitale_modo": capitale_modo,
        "riserva_btd_pct": float(riserva),
        "boost_pct": float(boost),
        "reinvesto_modo": reinvesto_modo,
        "btd_dd_weekly_limit": float(limite_dd),
        "btd_execution": "open" if esecuzione.startswith("Apertura") else "close",
        "filtro_call": filtro_call,
        "carico_riferimento": carico_riferimento,
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
def _con_benchmark(risultato: Dict[str, Any]) -> Dict[str, Any]:
    """Le tre varianti piu' il benchmark, per le viste che li mostrano insieme."""
    tutte = dict(risultato.get("varianti", {}))
    bm = risultato.get("benchmark")
    if bm and isinstance(bm.get("monthly"), pd.DataFrame) and not bm["monthly"].empty:
        tutte["benchmark"] = bm
    return tutte


def _avvisi_sottostante(risultato: Dict[str, Any], metriche: Dict[str, Any]) -> None:
    """Segnala i casi in cui il confronto rischia di essere frainteso."""
    prezzi = risultato.get("mercato", {}).get("prezzi")
    if isinstance(prezzi, pd.DataFrame) and not prezzi.empty:
        moltiplicatore = float(prezzi["Close"].iloc[-1] / prezzi["Open"].iloc[0])
        anni = float(metriche.get("anni") or (len(prezzi) / 12.0))
        if moltiplicatore > 100 and anni > 3:
            st.warning(
                f"**Il sottostante si e' moltiplicato per {moltiplicatore:,.0f} volte** in "
                + A(f"{anni:.0f} anni. Vendere ogni mese una call at-the-money su un rialzo di ") +
                f"questa portata significa cedere quasi tutto il guadagno: il costo del cap "
                f"e' {fmt_currency_compact(metriche.get('intrinseco_totale'))} contro "
                f"{fmt_currency_compact(metriche.get('premi_totali'))} di premi incassati. "
                f"Non e' un errore del modello, e' cosa succede a incassare premi su un "
                f"sottostante che corre. Se vuoi tenere piu' rialzo, abbassa il delta della "
                f"call nella sidebar."
            )
    ticker = str(risultato.get("config", {}).get("ticker", "")).upper()
    if ticker.endswith(".INDX") or ticker.lstrip("$^").split(".")[0] in ("GSPC", "SPX", "NDX", "DJI"):
        st.warning(
            "**Stai usando un indice di prezzo**, che per costruzione esclude i dividendi: "
            "storicamente circa due punti percentuali all'anno sull'S&P 500. La strategia "
            "compra e tiene il sottostante, quindi nella realta' quei dividendi li "
            "incasseresti, e il rendimento reale sarebbe piu' alto di quanto vedi qui. "
            "Per una simulazione completa usa l'ETF corrispondente (per esempio `SPY.US`), "
            "dove il motore lavora sui prezzi rettificati e i dividendi sono inclusi."
        )

    cfgd = risultato.get("config", {})
    if str(cfgd.get("capitale_modo", "fisso")) == "fisso":
        x = (risultato.get("varianti", {}).get("premi_cash", {}) or {}).get("monthly")
        if isinstance(x, pd.DataFrame) and not x.empty and "capitale_impiegato_anno" in x:
            gen = x.groupby("anno").head(1)
            cap = (gen["quote_coperte"] + gen["quote_extra"]) * gen["open"]
            crescita_cap = cap.iloc[-1] / cap.iloc[0] - 1 if cap.iloc[0] else 0
            crescita_conto = (x["valore_portafoglio"].iloc[-1]
                              / x["valore_portafoglio"].iloc[0] - 1)
            if crescita_conto > 0.5 and crescita_cap < 0.25:
                st.warning(
                    f"**La strategia non compone.** Il capitale rimesso al lavoro a gennaio "
                    f"e' passato da {fmt_currency_compact(cap.iloc[0])} a "
                    f"{fmt_currency_compact(cap.iloc[-1])} ({crescita_cap:+.0%}) mentre il "
                    f"conto e' cresciuto del {crescita_conto:+.0%}: gli utili restano in "
                    f"cassa e ogni anno si riparte dallo stesso importo. La curva cresce in "
                    f"linea retta invece che esponenzialmente. Nella sidebar, sotto "
                    f"*Capitale*, scegli **Cresce insieme al "
                    f"conto** per farli tornare al lavoro."
                )

    quota = metriche.get("quota_conto_investita")
    if quota is not None and quota < 0.80:
        tasso = float(risultato.get("config", {}).get("idle_cash_rate", 0.0))
        msg = (
            f"**Solo il {quota:.0%} del conto e' investito**, il resto e' cassa: il reset "
            f"annuale reimpiega il capitale fisso e lascia ferma tutta la liquidita' "
            f"accumulata (in media {fmt_currency_compact(metriche.get('cassa_media'))}). "
            f"Il rendimento riportato e' quello sul capitale davvero investito "
            f"(**{fmt_pct(metriche.get('rendimento_medio'), 2)} medio annuo**), "
            f"non sul conto intero: la cassa ferma non entra al denominatore."
        )
        if tasso <= 0:
            msg += (" E quella cassa e' remunerata allo 0%: se la parcheggi in un monetario, "
                    "imposta il tasso nella sidebar sotto *Capitale*, perche' su questi "
                    "importi cambia molto il risultato.")
        st.info(msg)

    bh = metriche.get("bh_stessi_flussi_pnl")
    ciclo = metriche.get("ciclo_pnl")
    if bh and ciclo and abs(ciclo) > 0 and abs(bh) > 20 * abs(ciclo):
        st.info(
            f"Il Buy &amp; Hold che non liquida mai arriva a "
            f"{fmt_currency_compact(bh)}, cioe' {abs(bh) / abs(ciclo):,.0f} volte il ciclo "
            f"annuale: tiene per sempre le quote comprate all'inizio, quando il sottostante "
            f"costava una frazione. Non e' un confronto a parita di mandato ed e' escluso "
            f"dai grafici a scala lineare, dove renderebbe piatte tutte le altre curve. "
            f"Lo trovi attivando la scala logaritmica.".replace("&amp;", "&")
        )


def scheda_sintesi(risultato: Dict[str, Any], figure: Dict[str, Any]) -> None:
    varianti = risultato["varianti"]
    reinvest = varianti.get("premi_reinvest", {}).get("metrics", {})
    cash = varianti.get("premi_cash", {}).get("metrics", {})

    kpi_cards([
        ("Utile netto — Reinvest", fmt_currency_compact(reinvest.get("pnl_netto")),
         f"su {fmt_currency_compact(reinvest.get('versamenti_totali'))} versati",
         segno_di(reinvest.get("pnl_netto"))),
        ("Rendimento medio — Reinvest", fmt_pct(reinvest.get("rendimento_medio")),
         f"Buy & Hold {fmt_pct(reinvest.get('ciclo_rendimento_medio'))}",
         segno_di(reinvest.get("extra_rendimento_vs_ciclo"))),
        # Il drawdown da mostrare e' quello valorizzato ogni giorno: quello di
        # fine periodo non vede i crolli rientrati prima della chiusura.
        ("Max drawdown VERO — Cash",
         fmt_pct(cash.get("max_dd_giornaliero_pct", cash.get("max_dd_pct"))),
         (f"a fine periodo sembrava {fmt_pct(cash.get('max_dd_pct'))}"
          if cash.get("max_dd_giornaliero_pct") is not None
          else f"Buy & Hold {fmt_pct(cash.get('bh_max_dd_pct'))}"),
         segno_di(cash.get("riduzione_dd_vs_bh"))),
        ("Rendimento / oscillazione — Cash", fmt_num(cash.get("rendimento_su_rischio")),
         f"Buy & Hold {fmt_num(cash.get('ciclo_rendimento_su_rischio'))}",
         segno_di((cash.get("rendimento_su_rischio") or 0)
                  - (cash.get("ciclo_rendimento_su_rischio") or 0))),
        ("Anni in utile — Reinvest",
         f"{reinvest.get('anni_positivi', 0)}/{reinvest.get('anni_totali', 0)}",
         f"peggior anno {fmt_pct(reinvest.get('peggior_anno'), 1)}", None),
        ("Premio medio stimato", fmt_pct(cash.get("premio_pct_medio")),
         A("del prezzo del sottostante, al mese"), None),
        (A("Call in-the-money"), f"{cash.get('mesi_call_assegnata', 0)}/{cash.get('mesi', 0)}",
         A("mesi in cui il cap ha morso"), None),
    ])

    usa_ciclo = cash.get("ciclo_rendimento_medio") is not None
    metro = ("solo sottostante con lo stesso ciclo annuale" if usa_ciclo
             else "Buy &amp; Hold a parita di versamenti")
    dd_cash = cash.get("riduzione_dd_vs_ciclo" if usa_ciclo else "riduzione_dd_vs_bh")
    extra_re = reinvest.get("extra_rendimento_vs_ciclo")
    verdetti = []
    if dd_cash is not None:
        verdetti.append(
            f"con i premi tenuti in cassa il drawdown massimo e' "
            f"<b>{'inferiore' if dd_cash > 0 else 'superiore'} del {abs(dd_cash):.0%}</b>")
    if extra_re is not None:
        verdetti.append(
            f"reinvestendo i premi il rendimento medio annuo e' <b>{abs(extra_re):.1%} "
            f"{'sopra' if extra_re > 0 else 'sotto'}</b>")
    if verdetti:
        nota(f"Sul periodo analizzato, rispetto al <b>{metro}</b>, " + " e ".join(verdetti) +
             ". E' il confronto corretto: stesso capitale impiegato a gennaio e liquidato a "
             "dicembre, l'unica differenza sono le opzioni e i Buy-The-Dip.")

    _avvisi_sottostante(risultato, cash)

    for chiave in ("verdetto_bh", "confronto_equity", "rendimenti_annuali"):
        if chiave in figure:
            grafico(figure[chiave], key=f"sintesi_{chiave}")


def scheda_rischio(risultato: Dict[str, Any], figure: Dict[str, Any]) -> None:
    cash = risultato["varianti"].get("premi_cash", {}).get("metrics", {})
    vero = cash.get("max_dd_giornaliero_pct")
    periodo = cash.get("max_dd_pct")
    if vero is None:
        nota("Senza dati giornalieri il conto e' valorizzato solo alla chiusura di ogni "
             "periodo: un crollo rientrato prima della chiusura non compare, e il "
             "drawdown qui sotto e' piu' tenero di quello vero.")
    else:
        nascosto = cash.get("dd_nascosto_dal_periodo")
        intra = cash.get("max_dd_intraday_pct")
        nota(
            f"I drawdown di questa scheda sono misurati <b>valorizzando il conto ogni "
            f"giorno di borsa</b>, non solo alla chiusura del periodo. Sulla variante "
            f"Cash il massimo vero e' {fmt_pct(vero, 1)} contro il {fmt_pct(periodo, 1)} "
            f"che si leggeva sulle sole chiusure"
            + (f", {fmt_pct(abs(nascosto or 0), 1)} in piu'" if nascosto else "")
            + (f"; guardando i minimi di giornata si e' arrivati a {fmt_pct(intra, 1)}"
               if intra is not None else "") + "."
        )
    for chiave in ("valorizzazione", "dd_frequenza", "pnl_netto", "underwater",
                   "eq_dd_no_premi", "eq_dd_cash", "eq_dd_reinvest",
                   "rischio_rendimento", "durata_dd"):
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
        + A(f"{html.escape(str(fonte))} e sempre noti <b>prima</b> dell'inizio del mese. "
            f"L'incasso in valuta e' una percentuale del prezzo corrente, quindi cambia "
            f"ogni mese.")
    )
    cash = risultato["varianti"].get("premi_cash", {}).get("metrics", {})

    filtro = str(cfg.get("filtro_call", "sempre"))
    if filtro != "sempre":
        quota = cash.get("quota_periodi_con_call")
        verso = "sotto" if filtro == "sotto_carico" else "sopra"
        base = ("delle quote coperte, fissato all'apertura di gennaio"
                if cfg.get("carico_riferimento") != "medio"
                else "medio dell'intera posizione, che scende a ogni acquisto sui cali")
        nota(A(
            f"<b>La call non si vende sempre.</b> Si vende solo quando il prezzo di apertura "
            f"sta {verso} il prezzo di carico {base}: e' successo nel "
            f"{fmt_pct(quota, 0)} dei mesi. Negli altri la posizione resta scoperta, senza "
            f"premio incassato e senza cap sul rialzo."))

    rif = riferimenti.riferimento(str(cfg.get("ticker", "")))
    verdetto = riferimenti.giudizio(cash.get("premio_pct_medio"), rif)
    if verdetto:
        st.info(verdetto)

    finanziamento = cash.get("finanziamento_massimo") or 0.0
    if finanziamento > 0.05 * float(cfg.get("capitale_iniziale", 1)):
        st.warning(
            f"Il riacquisto delle call in-the-money ha portato il conto a debito fino a "
            + A(f"{fmt_currency_compact(finanziamento)} per "
                f"{cash.get('mesi_a_debito', 0)} mesi ") +
            f"({fmt_pct(finanziamento / float(cfg.get('capitale_iniziale', 1)), 0)} del capitale "
            f"fisso). Sul finanziamento e' applicato il "
            f"{cfg.get('debit_cash_rate', 0):.1%} annuo impostato nella sidebar. "
            f"Se preferisci evitarlo, tieni i premi in cassa invece di reinvestirli, "
            f"oppure vendi la call a un delta piu' basso."
        )
    st.markdown("#### Da dove viene il risultato")
    kpi_cards([
        ("Movimento delle quote", fmt_currency_compact(cash.get("contributo_prezzo")),
         "il sottostante che sale o scende", segno_di(cash.get("contributo_prezzo"))),
        ("Premi incassati", fmt_currency_compact(cash.get("premi_totali")),
         A(f"{fmt_pct(cash.get('premio_pct_medio'))} dello spot al mese"), "pos"),
        ("Intrinseco pagato", fmt_currency_compact(-(cash.get("intrinseco_totale") or 0)),
         A(f"call ITM in {cash.get('mesi_call_assegnata', 0)} mesi "
           f"su {cash.get('mesi', 0)}"), "neg"),
        ("Netto delle opzioni", fmt_currency_compact(cash.get("netto_opzioni")),
         "premi meno intrinseco", segno_di(cash.get("netto_opzioni"))),
        ("Interessi", fmt_currency_compact(cash.get("interessi_netti")),
         "su cassa e saldo a debito", segno_di(cash.get("interessi_netti"))),
        ("Utile netto", fmt_currency_compact(cash.get("pnl_netto")),
         "somma delle voci precedenti", segno_di(cash.get("pnl_netto"))),
    ])
    netto = cash.get("netto_opzioni")
    if netto is not None and cash.get("premi_totali"):
        quota_persa = -netto / cash["premi_totali"] if cash["premi_totali"] else 0
        if netto < 0:
            nota(f"Le opzioni hanno <b>tolto</b> {fmt_currency_compact(-netto)}: l'intrinseco "
                 f"pagato ha superato i premi del {quota_persa:.0%}. Su un sottostante che "
                 f"sale con decisione e' il comportamento atteso di una call venduta vicino "
                 f"al denaro: per tenere piu' rialzo, abbassa il delta nella sidebar.")
        else:
            nota(f"Le opzioni hanno <b>aggiunto</b> {fmt_currency_compact(netto)} netti, "
                 f"pari al {netto / max(abs(cash.get('pnl_netto') or 1), 1):.0%} dell'utile.")

    rein = risultato["varianti"].get("premi_reinvest", {}).get("metrics", {})
    if cfg.get("reinvesto_modo") == "al_btd" and rein.get("attesa_media_periodi") is not None:
        nota(
            A(f"<b>Reinvestimento differito.</b> I premi non comprano quote alla chiusura "
              f"del mese: restano in un conto a parte e rientrano tutti insieme al prossimo "
              f"acquisto sui cali, <b>al lordo</b> e allo stesso prezzo del BTD. "
              f"L'intrinseco si paga poi a scadenza. In media hanno aspettato "
              f"{rein['attesa_media_periodi']:.1f} mesi")
            + (f", e {fmt_currency_compact(rein.get('premi_mai_reinvestiti'))} sono stati "
               f"liquidati a dicembre senza fare in tempo a rientrare"
               if rein.get("premi_mai_reinvestiti") else "")
            + f". In tutto e' tornato al lavoro il "
              f"{fmt_pct(rein.get('quota_premi_reinvestiti'))} di quello che le opzioni "
              f"hanno prodotto."
        )
    for chiave in ("premio", "prezzo_strike", "reinvestimento", "composizione_annuale"):
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
        A("Il segnale scatta quando il mese precedente chiude in negativo. ") +
        f"Ogni acquisto vale "
        f"l'entita' del calo applicata al capitale iniziale <b>piu' il BTD Boost</b>, pari al "
        f"{cfg.get('boost_pct', 0):.1%} del capitale iniziale "
        f"({fmt_currency_compact(cap0 * float(cfg.get('boost_pct', 0)))} per ogni acquisto). "
        f"Nessun limite al cumulato dell'anno: ogni segnale viene eseguito per intero. "
        f"Esecuzione "
        + A(f"{'all&#39;apertura' if cfg.get('btd_execution') == 'open' else 'alla chiusura'}"
            f" del mese.")
    )
    m = risultato["varianti"].get("premi_cash", {}).get("metrics", {})
    # Di default non c'e' nessun tetto; l'avviso serve solo a chi ne imposta uno
    # da configurazione, perche' l'effetto e' controintuitivo.
    anni_pieni = m.get("anni_con_tetto_esaurito") or 0
    anni_tot = m.get("anni_totali") or 0
    if cfg.get("btd_cap_annuo_pct") and anni_tot and anni_pieni >= max(2, anni_tot // 3):
        peggiore = m.get("btd_calo_peggiore_saltato")
        st.warning(
            f"**Il tetto annuo e' il vincolo che decide.** Si e' esaurito in "
            f"{anni_pieni} anni su {anni_tot}, togliendo "
            f"{fmt_currency_compact(m.get('btd_tagliato_dal_tetto'))} agli acquisti e "
            f"saltandone {m.get('btd_segnali_saltati', 0)} del tutto"
            + (f", incluso un calo del {abs(peggiore):.0%}" if peggiore else "")
            + ". In questa situazione alzare il BTD Boost non fa comprare di piu': fa "
              "esaurire il budget prima, sui cali superficiali di inizio anno, lasciando "
              "scoperti quelli profondi che arrivano dopo. Se vuoi che il boost si esprima, "
              "Toglilo del tutto, oppure alzalo, se vuoi che il boost si esprima."
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
    ("btd_importo", "BTD totale"), ("capitale_impiegato_anno", "Capitale impiegato"),
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
         A(f"call ITM in {r['mesi_call_itm']} mesi su {r['mesi_trascorsi']}"),
         segno_di(r["netto_opzioni"])),
        ("BTD investito", fmt_currency_compact(r["btd_investito"]),
         f"{r['btd_numero']} acquisti nell'anno", None),
        ("di cui boost", fmt_currency_compact(r["btd_da_boost"]),
         f"calo {fmt_currency_compact(r['btd_da_calo'])}", None),
        ("Valore del conto", fmt_currency_compact(r["valore_conto"]),
         f"posizione {fmt_currency_compact(r['valore_posizione'])} + cassa "
         f"{fmt_currency_compact(r['cassa'])}", None),
    ])

    dettagli = [
        f"capitale fisso impiegato a gennaio {fmt_currency_compact(r['capitale_fisso'])}",
        f"quote coperte {fmt_num(r['quote_coperte'], 4)}",
        f"quote extra {fmt_num(r['quote_extra'], 4)}",
    ]
    if r.get("btd_tetto"):
        dettagli.append(f"tetto annuo BTD {fmt_currency_compact(r['btd_tetto'])}")
    dettagli.insert(1, f"BTD Boost {fmt_currency_compact(r['boost_per_acquisto'])} per acquisto")
    if r["capitale_addizionale"]:
        dettagli.insert(1, "capitale addizionale non coperto "
                           f"{fmt_currency_compact(r['capitale_addizionale'])}")
    if r["versamenti"]:
        dettagli.append(f"versato nell'anno {fmt_currency_compact(r['versamenti'])}")
    if r["segnali_bloccati"]:
        dettagli.append(f"{r['segnali_bloccati']} segnali bloccati dal filtro")
    nota(" · ".join(dettagli))

    # ---------------- Piano del periodo prossimo ----------------
    piano = piano_prossimo_mese(risultato, variante)
    if piano and int(scelto) == int(risultato["varianti"][variante]["monthly"]["anno"].iloc[-1]):
        st.markdown(A("#### Cosa fare il mese prossimo"))
        if piano["reset_annuale"]:
            st.warning(
                f"**{piano['mese']} — reset annuale.** Liquidare tutta la posizione alla "
                f"chiusura di dicembre e reimpiegare "
                f"{fmt_currency_compact(piano['capitale_da_impiegare'])} all'apertura di "
                f"gennaio. Il ciclo degli acquisti sui cali riparte da zero."
            )
        else:
            righe = []
            if piano["segnale_btd"] and piano["btd_importo"] > 0:
                righe.append(
                    f"**Acquisto BTD di {fmt_currency_compact(piano['btd_importo'])}** "
                    + A("all'apertura del mese: ")
                    + f"{fmt_currency_compact(piano['btd_quota_calo'])} "
                    f"per il calo del {fmt_pct(abs(piano['rendimento_ultimo_mese']), 1)} e "
                    f"{fmt_currency_compact(piano['btd_quota_boost'])} di boost."
                    + (f" Dopo l'acquisto resteranno "
                       f"{fmt_currency_compact(piano['btd_residuo_anno'] - piano['btd_importo'])} "
                       f"sotto il tetto annuo." if piano.get("btd_residuo_anno") else "")
                )
            elif piano["btd_bloccato"]:
                righe.append(
                    f"Segnale BTD presente ma **bloccato dal filtro**: il drawdown "
                    f"settimanale e' {fmt_pct(piano['dd_weekly'], 1)}."
                )
            elif piano["segnale_btd"]:
                righe.append("Segnale BTD presente ma l'importo calcolato e' nullo.")
            else:
                righe.append(
                    A("**Nessun acquisto BTD**: l'ultimo mese ha chiuso a ")
                    + f"{fmt_pct(piano['rendimento_ultimo_mese'], 1)}."
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
            st.caption(A(
                "Segnale e volatilita' sono gia' determinati dalla chiusura del mese scorso. "
                "Lo strike e le quote si ricalcolano sul prezzo di apertura effettivo."
            ))

    # ---------------- Tabella periodo per periodo ----------------
    st.markdown(A("#### Dettaglio mese per mese"))
    g = dett["mesi"].copy()
    # Sul settimanale il nome del mese non identifica la riga: serve la data.
    g["mese"] = ([d.strftime("%d/%m") for d in g.index]
                 if cadenza_corrente() == "settimanale"
                 else [MESI_IT[d.month - 1].capitalize() for d in g.index])
    g["stato_btd"] = [
        "bloccato" if (b and s) else ("acquisto" if i > 0 else ("segnale" if s else "—"))
        for s, b, i in zip(g["segnale_btd"], g["btd_bloccato"], g["btd_importo"])
    ]
    vista = pd.DataFrame({A(etichetta): g[col] for col, etichetta in COLONNE_MONITOR
                          if col in g.columns})
    valuta = ["Apertura", "Chiusura", "BTD dal calo", "BTD dal boost", "BTD totale",
              "Capitale impiegato", "Strike venduto", "Premio incassato",
              "Intrinseco pagato",
              "Netto opzione", "Cassa", "Valore conto", "Versato", "Utile netto", "Drawdown"]
    for c in valuta:
        if c in vista.columns:
            vista[c] = vista[c].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}")
    for c in (A("Rend. mese"), "Premio %"):
        if c in vista.columns:
            vista[c] = vista[c].map(lambda v: "—" if pd.isna(v) else f"{v * 100:,.2f}%")
    for c in ("Quote coperte", "Quote extra"):
        if c in vista.columns:
            vista[c] = vista[c].map(lambda v: f"{v:,.4f}")
    st.dataframe(vista.set_index(A("Mese")), **LARGO)

    csv = dett["mesi"].to_csv().encode("utf-8")
    st.download_button(f"Scarica il {anno} in CSV", data=csv,
                       file_name=f"boosted_covered_call_{anno}_{variante}.csv",
                       mime="text/csv")


def scheda_dati(risultato: Dict[str, Any], calibrazione: Optional[Dict[str, Any]]) -> None:
    st.markdown("#### Metriche a confronto")
    nota("L'ultima colonna e' il solo sottostante comprato e liquidato con lo stesso ciclo "
         "annuale della strategia: stesso capitale a gennaio, tutto chiuso a dicembre, "
         "senza opzioni e senza Buy-The-Dip. E' il metro corretto.")
    tb = metrics_table(_con_benchmark(risultato))
    if not tb.empty:
        vista = pd.DataFrame(
            {col: [format_value(k, tb.loc[k, col]) for k in tb.index] for col in tb.columns},
            index=[etichette_metriche(cadenza_corrente()).get(k, k) for k in tb.index],
        )
        st.dataframe(vista, **LARGO)

    st.markdown("#### Dettaglio per anno")
    scelta = st.selectbox("Variante", list(VARIANTS.keys()),
                          format_func=lambda k: VARIANTS[k]["label"], index=1, key="anno_var")
    y = risultato["varianti"][scelta]["yearly"]
    if not y.empty:
        vista_y = y.copy()
        # Ogni anno affiancato a quello che avrebbe fatto il solo sottostante
        bm = risultato.get("benchmark") or {}
        yb = bm.get("yearly")
        if isinstance(yb, pd.DataFrame) and not yb.empty:
            vista_y["risultato_buy_hold"] = yb["risultato_anno"].reindex(vista_y.index)
            vista_y["rendimento_buy_hold"] = yb["rendimento_anno"].reindex(vista_y.index)
            vista_y["differenza"] = vista_y["risultato_anno"] - vista_y["risultato_buy_hold"]
            vista_y["differenza_rendimento"] = (vista_y["rendimento_anno"]
                                                - vista_y["rendimento_buy_hold"])
        for c in ("rendimento_sottostante", "rendimento_anno", "twr_anno",
                  "twr_buy_hold", "differenza_twr", "rendimento_buy_hold",
                  "differenza_rendimento"):
            if c in vista_y.columns:
                vista_y[c] = vista_y[c].map(lambda v: fmt_pct(v, 1))
        for c in ("premi_incassati", "intrinseco_pagato", "netto_opzioni", "btd_investito",
                  "btd_da_calo", "btd_da_boost", "versamenti", "capitale_investito",
                  "capitale_medio_impiegato", "valore_fine_anno", "risultato_anno",
                  "risultato_buy_hold", "differenza"):
            if c in vista_y.columns:
                vista_y[c] = vista_y[c].map(lambda v: "—" if pd.isna(v) else f"${v:,.0f}")
        vista_y = vista_y.drop(columns=[c for c in ("twr_anno",) if c in vista_y.columns])
        vista_y.columns = [c.replace("_", " ").capitalize() for c in vista_y.columns]
        st.dataframe(vista_y, **LARGO)
        st.caption("Il rendimento di ogni anno e' il risultato diviso il capitale davvero "
                   "investito in quel ciclo: capitale di gennaio piu' gli acquisti sui cali. "
                   "I premi reinvestiti non entrano al denominatore perche' arrivano dal "
                   "mercato. Le ultime colonne confrontano con il solo sottostante che segue "
                   "lo stesso ciclo annuale.")

    st.markdown(A("#### Serie mensile"))
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
    nota(A("Il file JSON contiene parametri, equity, serie mensile completa di ogni "
           "variante, ") +
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
        st.session_state["calibrazione"] = calib.pacchetto_export(
            fit, nome_file=file.name, ticker=str(params.get("ticker", "")))

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
    st.success(
        "**Calibrazione pronta.** Nella sidebar, sotto *Premio della call*, scegli "
        "**Calibrato sui prezzi reali** e rilancia il backtest: i parametri vengono presi "
        "da qui, non c'e' niente da ricopiare a mano. La calibrazione finisce anche nel JSON "
        "di export."
        + (f" Alla volatilita mediana di questo sottostante ({sigma_med:.0%}) il rapporto "
           f"applicato e' {fit['modello'].vrp_effettivo(sigma_med):.3f}." if sigma_med else "")
    )
    st.caption(_anteprima_premio(fit["modello"].vrp, fit["modello"].vrp_slope,
                                 fit["modello"].target_delta))


# ---------------------------------------------------------------------------
# Corpo
# ---------------------------------------------------------------------------
params, prefs, esegui = sidebar()

# Il titolo va scritto prima che il backtest giri, quindi non puo' leggere la
# cadenza dai risultati: quando si sta per lanciare una nuova esecuzione vale
# quella appena scelta nella sidebar, altrimenti quella dei risultati a schermo.
_cad_titolo = normalizza(params.get("cadenza")) if esegui else cadenza_corrente()
st.title(f"Boosted Covered Call — Studio {etichetta_cadenza(_cad_titolo)}")
st.caption(adatta("Capitale fisso annuo coperto da una call mensile a delta 0.50, "
                  "Buy-The-Dip potenziato sui mesi negativi, liquidazione e reset a "
                  "fine anno.", _cad_titolo))

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
        st.markdown(A(
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
        ))
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
    f"cadenza {etichetta_cadenza(cadenza_corrente()).lower()} · "
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
    scheda_rischio(risultato, figure)
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
