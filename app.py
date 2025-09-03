# app.py
# =============================================================================
# Boosted Covered Call — Studio Mensile (Streamlit)
# =============================================================================
# - UI pulita, nessun unsafe_allow_html
# - Parametri in sidebar
# - Chiamata a kq_btd_cc.esegui_analisi_completa()
# - Mostra grafici base + (opzionale) grafici addizionali originali
# =============================================================================

from __future__ import annotations

import os
import datetime as dt
from typing import Dict, Any, List, Union

import streamlit as st
import matplotlib.pyplot as plt

# Pacchetto strategia
from kq_btd_cc import esegui_analisi_completa, STYLE_CONFIG


# -----------------------------------------------------------------------------
# Config pagina (PRIMA chiamata Streamlit)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Boosted Covered Call — Studio Mensile",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("Boosted Covered Call — Studio Mensile")
st.caption("Backtest interattivo con premi mensili, Buy-The-Dip potenziato e filtro su drawdown settimanale dell'asset.")


# -----------------------------------------------------------------------------
# Check chiave EODHD
# -----------------------------------------------------------------------------
_HAS_KEY = ("EODHD_API_KEY" in st.secrets) or bool(os.getenv("EODHD_API_KEY"))
if not _HAS_KEY:
    st.warning("Per usare l'app imposta `EODHD_API_KEY` nei Secrets (Streamlit Cloud → Settings → Secrets).")
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar — Parametri
# -----------------------------------------------------------------------------
def build_sidebar_params() -> Dict[str, Any]:
    with st.sidebar:
        st.header("Parametri strategia")

        ticker = st.text_input(
            "Ticker (EODHD)",
            value="BTC-USD.CC",
            help="Formato EODHD (es. BTC-USD.CC, ETH-USD.CC, AAPL.US, …)",
        ).strip()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Data inizio", value=dt.date(2018, 1, 1))
        with col2:
            use_end_override = st.checkbox("Usa data fine manuale", value=False)
            end_date_val = st.date_input(
                "Data fine (se attiva)",
                value=dt.date.today() - dt.timedelta(days=1),
                help="Ignorata se il toggle non è attivo.",
            )

        st.divider()
        st.subheader("Configurazioni economiche")

        options_prem_pct = st.slider(
            "Premio Opzioni Mensile (%)",
            0.0, 20.0, 5.0, 0.5,
            help="Percentuale mensile del premio sul pacchetto covered (in %).",
        ) / 100.0

        initial_capital = st.number_input(
            "Capitale Iniziale (USD) per Covered Call",
            value=25_000, step=500, min_value=0,
        )
        additional_capital = st.number_input(
            "Capitale Addizionale Annuale (USD)",
            value=0, step=500, min_value=0,
        )
        capital_boost_pct = st.slider(
            "Boost % Capitale BTD (su cap. iniziale)",
            0.0, 10.0, 5.0, 0.5,
        ) / 100.0

        st.divider()
        st.subheader("Rischio e filtri")

        var_conf_level = st.slider(
            "Livello confidenza VaR",
            0.90, 0.999, 0.99, 0.005,
        )
        btd_dd_limit_pct = st.slider(
            "Limite Drawdown Weekly Asset per bloccare BTD (%)",
            -95.0, -30.0, -90.0, 5.0,
        ) / 100.0  # frazione (negativo)

        st.divider()
        with st.expander("Preferenze grafici (avanzato)", expanded=False):
            g1 = st.checkbox("Grafico 1 — Equity vs Cap. cumulativo", value=True)
            g_abc = st.checkbox("Grafici A–B–C — Equity & DD per strategia", value=True)
            g5 = st.checkbox("Grafico 5 — Reinvestimenti BTD mensili", value=True)
            g6 = st.checkbox("Grafico 6 — Drawdown weekly asset", value=True)
            g7 = st.checkbox("Grafico 7 — Confronto finale con Buy&Hold", value=True)
            g_add = st.checkbox(
                "Grafici addizionali (Violin, Rolling Sharpe, Risk/Return, DD duration, Dashboard)",
                value=True
            )
            g_ann = st.checkbox("Grafico rendimenti annuali operativi", value=True)

            # Debug opzionali
            dbg_reset = st.checkbox("Debug reset annuale", value=False)
            dbg_reset_np = st.checkbox("Debug reset BTD no premi", value=False)
            dbg_btd = st.checkbox("Debug calcolo BTD mensile", value=False)
            dbg_mese = st.checkbox("Debug mensile BTD no premi", value=False)
            dbg_primo = st.checkbox("Debug primo mese anno", value=False)

        st.divider()
        run_btn = st.button("Esegui analisi", type="primary", use_container_width=True)

        params_gui: Dict[str, Any] = {
            "EODHD_TICKER": ticker or "BTC-USD.CC",
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
            # >>> FLAG per il core: raccogli e restituisci le Figure invece di chiuderle
            "streamlit_collect_figs": True,
        }

    return params_gui, plot_prefs, run_btn


# -----------------------------------------------------------------------------
# Corpo principale
# -----------------------------------------------------------------------------
params_gui, plot_prefs, run_btn = build_sidebar_params()

colA, colB, colC = st.columns(3)
with colA:
    st.metric("Ticker", params_gui.get("EODHD_TICKER", "—"))
with colB:
    st.metric("Data inizio", params_gui.get("START_DATE", "—"))
with colC:
    st.metric("Data fine", params_gui.get("END_DATE_OVERRIDE", "Auto (ieri)"))

st.divider()

def _display_fig_list(figs: List, title: str | None = None):
    if title:
        st.subheader(title)
    for i, fig in enumerate(figs, start=1):
        st.pyplot(fig, use_container_width=True)
        if i < len(figs):
            st.divider()

if run_btn:
    with st.spinner("Esecuzione backtest e generazione grafici…"):
        try:
            # Chiamata al core — compatibile con firma estesa (gli extra args hanno default)
            result: Union[List, Dict[str, Any]] = esegui_analisi_completa(
                params_gui=params_gui,
                plot_prefs=plot_prefs,
            )

            base_figs: List = []
            extra_figs: List = []

            # --- 1) Caso: il core restituisce DIZIONARIO con le figure (nuovo)
            if isinstance(result, dict):
                base_figs = result.get("figures", []) or result.get("figures_base", []) or []
                extra_figs = result.get("figures_extra", []) or result.get("figures_additional", []) or []

            # --- 2) Caso: il core restituisce LISTA (vecchio comportamento)
            elif isinstance(result, list):
                base_figs = result

            # --- 3) Fallback: se non ci sono figure ritornate, prova a pescare quelle ancora aperte
            #      (funziona se il core NON le chiude quando streamlit_collect_figs=True)
            if not base_figs and not extra_figs:
                open_nums = plt.get_fignums()
                if open_nums:
                    figs_open = [plt.figure(n) for n in open_nums]
                    # Non sappiamo quali siano "extra": mostriamo tutto come base.
                    base_figs = figs_open

            if not base_figs and not extra_figs:
                st.info(
                    "Nessuna figura ricevuta dal core. "
                    "Per i grafici addizionali originali, assicurati che in `core.esegui_analisi_completa` "
                    "le figure vengano **raccolte** quando `plot_prefs['streamlit_collect_figs']=True` "
                    "e non chiuse con `plt.close(...)` (vedi patch suggerita in basso)."
                )
            else:
                if base_figs:
                    _display_fig_list(base_figs, title="Grafici principali")
                if plot_prefs.get("mostra_grafici_addizionali", False) and extra_figs:
                    _display_fig_list(extra_figs, title="Grafici addizionali")

        except Exception as ex:
            st.error("Si è verificato un errore durante l'esecuzione dell'analisi.")
            st.exception(ex)
else:
    st.info("Imposta i parametri nella sidebar e clicca **Esegui analisi**.")

# -----------------------------------------------------------------------------
# Nota per il core (mostrata come hint discreto sotto la pagina)
# -----------------------------------------------------------------------------
with st.expander("Se i grafici addizionali non compaiono, apri questo suggerimento (per core.py)"):
    st.markdown(
        """
**Mini-patch consigliata in `core.esegui_analisi_completa` (stile originale invariato):**

1. All'inizio della funzione:
```python
collect_figs = bool((plot_prefs or {}).get("streamlit_collect_figs", False))
figs_base, figs_extra = [], []

def _keep(fig, extra: bool = False):
    if collect_figs and fig is not None:
        (figs_extra if extra else figs_base).append(fig)
