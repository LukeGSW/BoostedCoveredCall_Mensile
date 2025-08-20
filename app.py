# app.py
# =============================================================================
# Boosted Covered Call — Studio Mensile (Streamlit)
# =============================================================================
# UI pulita, nessun unsafe_allow_html. Parametri in sidebar.
# La logica è interamente in kq_btd_cc.core.esegui_analisi_completa(),
# che deve restituire una LISTA di matplotlib.figure.Figure.
# =============================================================================

from __future__ import annotations

import os
import datetime as dt
from typing import Dict, Any, List

import streamlit as st

# Import del pacchetto della strategia
from kq_btd_cc import esegui_analisi_completa, STYLE_CONFIG


# -----------------------------------------------------------------------------
# Config pagina (deve essere la PRIMA chiamata Streamlit)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Boosted Covered Call — Studio Mensile",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Header (semplice, senza HTML unsafe)
# -----------------------------------------------------------------------------
st.title("Boosted Covered Call — Studio Mensile")
st.caption(
    "Backtest interattivo con premi mensili, Buy-The-Dip potenziato e filtro su drawdown settimanale dell'asset."
)


# -----------------------------------------------------------------------------
# Check segreto EODHD (silenzioso se presente). L'API key viene comunque letta
# in kq_btd_cc/data_api.py, ma qui evitiamo messaggi fuorvianti in dashboard.
# -----------------------------------------------------------------------------
_HAS_KEY = ("EODHD_API_KEY" in st.secrets) or bool(os.getenv("EODHD_API_KEY"))
if not _HAS_KEY:
    st.warning(
        "Per usare l'app imposta `EODHD_API_KEY` nei Secrets (Streamlit Cloud → Settings → Secrets)."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar — Parametri della strategia
# Nomi dei parametri allineati 1:1 con core.esegui_analisi_completa()
# -----------------------------------------------------------------------------
def build_sidebar_params() -> Dict[str, Any]:
    with st.sidebar:
        st.header("Parametri strategia")

        # Ticker EODHD (default richiesto: BTC-USD.CC)
        ticker = st.text_input(
            "Ticker (EODHD)",
            value="BTC-USD.CC",
            help="Formato EODHD (es. BTC-USD.CC, ETH-USD.CC, AAPL.US, etc.)",
        ).strip()

        # Date
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Data inizio",
                value=dt.date(2015, 1, 1),
                help="Data iniziale del backtest.",
            )
        with col2:
            use_end_override = st.checkbox(
                "Usa data fine manuale", value=False, help="Se attivo, imposta una data di fine."
            )
            end_date_val = st.date_input(
                "Data fine (se attiva)",
                value=dt.date.today() - dt.timedelta(days=1),
                help="Data finale del backtest. Ignorata se il toggle non è attivo.",
            )

        # Parametri economici
        st.divider()
        st.subheader("Configurazioni economiche")

        options_prem_pct = st.slider(
            "Premio Opzioni Mensile (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Percentuale mensile del premio sul pacchetto covered (espresso in %).",
        ) / 100.0

        initial_capital = st.number_input(
            "Capitale Iniziale (USD) per Covered Call",
            value=25_000,
            step=500,
            min_value=0,
            help="Capitale di partenza su cui si applica il premio.",
        )
        additional_capital = st.number_input(
            "Capitale Addizionale Annuale (USD)",
            value=0,
            step=500,
            min_value=0,
            help="Versamento addizionale effettuato ad ogni reset annuale.",
        )
        capital_boost_pct = st.slider(
            "Boost % Capitale BTD (su cap. iniziale)",
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.5,
            help="Quota fissa di boost per il BTD (espresso in % sul capitale iniziale).",
        ) / 100.0

        # Rischio
        st.divider()
        st.subheader("Rischio e filtri")

        var_conf_level = st.slider(
            "Livello confidenza VaR",
            min_value=0.90,
            max_value=0.999,
            value=0.99,
            step=0.005,
            help="Usato per il calcolo del VaR/CVaR sulle metriche riassuntive.",
        )
        btd_dd_limit_pct = st.slider(
            "Limite Drawdown Weekly Asset per bloccare BTD (%)",
            min_value=-95.0,
            max_value=-30.0,
            value=-90.0,
            step=5.0,
            help="Se il DD settimanale dell'asset scende oltre questa soglia, il BTD è bloccato per quel mese.",
        ) / 100.0  # core si aspetta numero in frazione (negativo)

        # Preferenze plotting (allineate a core)
        st.divider()
        with st.expander("Preferenze grafici (avanzato)", expanded=False):
            g1 = st.checkbox("Grafico 1 — Equity vs Cap. cumulativo", value=True)
            g_abc = st.checkbox("Grafici A–B–C — Equity & DD per strategia", value=True)
            g5 = st.checkbox("Grafico 5 — Reinvestimenti BTD mensili", value=True)
            g6 = st.checkbox("Grafico 6 — Drawdown weekly asset", value=True)
            g7 = st.checkbox("Grafico 7 — Confronto finale con Buy&Hold", value=True)
            g_add = st.checkbox("Grafici addizionali (violin, rolling, risk/return, DD duration, dashboard)", value=False)
            g_ann = st.checkbox("Grafico rendimenti annuali operativi", value=True)

            # Debug opzionali (default False)
            dbg_reset = st.checkbox("Debug reset annuale", value=False)
            dbg_reset_np = st.checkbox("Debug reset BTD no premi", value=False)
            dbg_btd = st.checkbox("Debug calcolo BTD mensile", value=False)
            dbg_mese = st.checkbox("Debug mensile BTD no premi", value=False)
            dbg_primo = st.checkbox("Debug primo mese anno", value=False)

        # Pulsante esecuzione
        st.divider()
        run_btn = st.button("Esegui analisi", type="primary", use_container_width=True)

        # Costruzione dict coerente con core
        params_gui: Dict[str, Any] = {
            "EODHD_TICKER": ticker if ticker else "BTC-USD.CC",
            "START_DATE": start_date.strftime("%Y-%m-%d"),
            "OPTIONS_PREMIUM_PERCENT": options_prem_pct,
            "INITIAL_CAPITAL": float(initial_capital),
            "ADDITIONAL_CAPITAL": float(additional_capital),
            "CAPITAL_BOOST_PERCENT": capital_boost_pct,
            "VAR_CONFIDENCE_LEVEL": float(var_conf_level),
            "BUY_THE_DIP_DRAWDOWN_LIMIT_PERCENT": float(btd_dd_limit_pct),
        }
        if use_end_override:
            params_gui["END_DATE_OVERRIDE"] = end_date_val.strftime("%Y-%m-%d")

        plot_prefs: Dict[str, Any] = {
            "mostra_grafico_1": g1,
            "mostra_grafici_abc": g_abc,
            "mostra_grafico_5": g5,
            "mostra_grafico_6": g6,
            "mostra_grafico_7": g7,
            "mostra_grafici_addizionali": g_add,
            "mostra_grafico_rend_annuali": g_ann,
            # Debug
            "mostra_debug_reset": dbg_reset,
            "mostra_debug_reset_btd_no_premi": dbg_reset_np,
            "mostra_debug_btd_amount": dbg_btd,
            "mostra_debug_mensile_btd_no_premi": dbg_mese,
            "mostra_debug_primo_mese": dbg_primo,
        }

    return params_gui, plot_prefs, run_btn


# -----------------------------------------------------------------------------
# Corpo principale
# -----------------------------------------------------------------------------
params_gui, plot_prefs, run_btn = build_sidebar_params()

# Info sintetiche sopra i grafici (non invasivo)
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Ticker", params_gui.get("EODHD_TICKER", "—"))
with colB:
    st.metric("Data inizio", params_gui.get("START_DATE", "—"))
with colC:
    if "END_DATE_OVERRIDE" in params_gui:
        st.metric("Data fine", params_gui["END_DATE_OVERRIDE"])
    else:
        st.metric("Data fine", "Auto (ieri)")

st.divider()

if run_btn:
    with st.spinner("Esecuzione backtest e generazione grafici…"):
        try:
            # La funzione deve restituire una LISTA di matplotlib Figure
            figs: List = esegui_analisi_completa(
                params_gui=params_gui,
                plot_prefs=plot_prefs,
                # export_* rimangono ai default della funzione (nessun salvataggio)
            )

            if not figs:
                st.warning("Nessuna figura generata. Verifica i parametri o i dati disponibili per il ticker selezionato.")
            else:
                for idx, fig in enumerate(figs, start=1):
                    st.pyplot(fig, use_container_width=True)
                    # opzionale: un separatore leggero tra i grafici
                    if idx < len(figs):
                        st.markdown("<hr>", unsafe_allow_html=True)

        except Exception as ex:
            st.error("Si è verificato un errore durante l'esecuzione dell'analisi.")
            # Mostra stacktrace utile in fase di sviluppo; rimuovi se non desideri esporlo
            st.exception(ex)
else:
    st.info("Imposta i parametri nella sidebar e clicca **Esegui analisi**.")
