import os
import io
import traceback
import streamlit as st
import pandas as pd

# App layout & page config
st.set_page_config(
    page_title="Kriterion — BTC Boosted Covered Call",
    page_icon="💹",
    layout="wide",
)

# ---- Header ----
st.title("Kriterion — BTC Boosted Covered Call")
st.caption("Dashboard interattiva della strategia. Logica identica al notebook, senza export CSV/HTML.")

# ---- Sidebar: Parametri ----
st.sidebar.header("⚙️ Parametri")

def _pct_slider(label, value, min_value, max_value, step=0.005, help=None):
    pct = st.sidebar.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, help=help)
    return float(pct)

def _money_input(label, value, step=1000, help=None):
    return float(st.sidebar.number_input(label, min_value=0.0, value=float(value), step=float(step), help=help))

# Ticker & periodo
default_ticker = "BTC-USD.CC"
ticker = st.sidebar.text_input("Ticker (EODHD)", value=default_ticker, help="Formato EODHD, es. BTC-USD.CC")
start_date = st.sidebar.date_input("Data inizio", value=pd.to_datetime("2015-01-01")).strftime("%Y-%m-%d")
use_end_override = st.sidebar.checkbox("Imposta data fine manuale?")
end_date_override = None
if use_end_override:
    end_date_override = st.sidebar.date_input("Data fine", value=pd.to_datetime("today")).strftime("%Y-%m-%d")

# Parametri strategia
opt_premium_pct = _pct_slider("Premio opzioni mensile (%)", 0.05, 0.00, 0.15, step=0.005,
                              help="Percentuale su sottostante per la vendita di covered call.")
initial_capital = _money_input("Capitale iniziale", 25000, step=1000)
additional_capital = _money_input("Capitale addizionale annuo", 0, step=1000)
boost_pct = _pct_slider("Boost % capitale BTD (su Cap. Iniziale)", 0.025, 0.0, 0.50, step=0.005)
var_conf = _pct_slider("Confidence level VaR", 0.99, 0.90, 0.999, step=0.001)
dd_limit_pct = st.sidebar.slider("Limite Drawdown asset per BTD (%)", min_value=-0.99, max_value=-0.05, value=-0.90, step=0.01,
                                 help="Se il DD settimanale dell'asset è sopra questa soglia (meno negativo), blocca BTD")

# Preferenze grafici
st.sidebar.header("📈 Grafici")
show_g1 = st.sidebar.checkbox("Grafico 1 — Equity vs BTD cumul.", value=True)
show_abc = st.sidebar.checkbox("Grafici A/B/C — Equity+DD", value=True)
show_g5 = st.sidebar.checkbox("Grafico 5 — Reinvestimenti BTD", value=True)
show_g6 = st.sidebar.checkbox("Grafico 6 — DD Asset sett.", value=True)
show_g7 = st.sidebar.checkbox("Grafico 7 — Confronto finale", value=True)
show_add = st.sidebar.checkbox("Extra — violin, rolling, scatter, dashboard", value=False)
show_ann = st.sidebar.checkbox("Tabella rendimenti annui (%)", value=True)

# Debug
st.sidebar.header("🧪 Debug")
show_debug = st.sidebar.checkbox("Mostra messaggi debug", value=False)

plot_prefs = {
    "mostra_grafico_1": show_g1,
    "mostra_grafici_abc": show_abc,
    "mostra_grafico_5": show_g5,
    "mostra_grafico_6": show_g6,
    "mostra_grafico_7": show_g7,
    "mostra_grafici_addizionali": show_add,
    "mostra_grafico_rend_annuali": show_ann,
    # opzionali di debug usati nel codice originale
    "mostra_debug_btd_amount": show_debug,
    "mostra_debug_mensile_btd_no_premi": show_debug,
}

params_gui = {
    "EODHD_TICKER": ticker,
    "START_DATE": start_date,
    "END_DATE_OVERRIDE": end_date_override,
    "OPTIONS_PREMIUM_PERCENT": float(opt_premium_pct),
    "INITIAL_CAPITAL": float(initial_capital),
    "ADDITIONAL_CAPITAL": float(additional_capital),
    "CAPITAL_BOOST_PERCENT": float(boost_pct),
    "VAR_CONFIDENCE_LEVEL": float(var_conf),
    "BUY_THE_DIP_DRAWDOWN_LIMIT_PERCENT": float(dd_limit_pct),
}

st.sidebar.markdown("---")
run = st.sidebar.button("🚀 Esegui Strategia", type="primary")

# ---- Body ----
st.info(
    f"**Ticker:** {ticker}  |  **Inizio:** {start_date}  |  **Fine:** {end_date_override or 'auto (ieri)'}",
    icon="ℹ️",
)

# API Key
api_ok = "EODHD_API_KEY" in st.secrets and bool(st.secrets["EODHD_API_KEY"]) 
st.success("API key EODHD: OK") if api_ok else st.warning("API key EODHD mancante nei *secrets*.")

# Lazy import del core (evita import costosi finché non serve)
from kq_btd_cc import core as kqcore

if run:
    if not api_ok:
        st.error("Imposta `EODHD_API_KEY` in `.streamlit/secrets.toml` o nei Secrets dell'app.")
        st.stop()

    # Imposta la API key come variabile globale nel modulo core (richiesto dal codice originale)
    kqcore.EODHD_API_KEY = st.secrets["EODHD_API_KEY"]

    with st.spinner("Esecuzione strategia in corso…"):
        try:
            results = kqcore.esegui_analisi_completa(
                params_gui=params_gui,
                plot_prefs=plot_prefs,
                export_enabled=False,
                export_html_report=False,
                num_columns_html_report=2,
                # estensione streamlit
                streamlit_mode=True,
            )
        except NotImplementedError as e:
            st.error("Core non ancora portato integralmente. Vedi sezione *Prossimi passi* sotto.")
            st.code(traceback.format_exc())
            st.stop()
        except Exception:
            st.error("Errore durante l'esecuzione della strategia.")
            st.code(traceback.format_exc())
            st.stop()

    # --- Output ---
    figs = results.get("figures", [])

    # KPI sintetiche (se disponibili)
    c1, c2, c3 = st.columns(3)
    for idx, (col, key) in enumerate(zip([c1, c2, c3], [
        "df_simulazione_np_export", "df_simulazione_cash_export", "df_simulazione_reinv_export"
    ])):
        df = results.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            last_equity = float(df["EquityValue_Strategy"].iloc[-1])
            col.metric(["BTD No Premi", "BTD+Premi(Cash)", "BTD+Premi(Reinv)"][idx], f"${last_equity:,.0f}")

    # Grafici
    if figs:
        st.subheader("Grafici")
        for fig in figs:
            st.pyplot(fig)
    else:
        st.info("Nessun grafico raccolto dal core (verifica le preferenze di plot o la portabilità del codice).")

    # Tabelle
    st.subheader("Tabelle di output")
    tabs = st.tabs([
        "BTD No Premi", "BTD + Premi (Cash)", "BTD + Premi (Reinvest)"
    ])
    keys = ["df_simulazione_np_export", "df_simulazione_cash_export", "df_simulazione_reinv_export"]
    for tab, key in zip(tabs, keys):
        with tab:
            df = results.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.caption("(tabella non disponibile)")

    st.success("Esecuzione completata.")

# Sezione informativa
with st.expander("ℹ️ Prossimi passi / Note di porting"):
    st.markdown(
        """
        - La logica del core è stata separata per mantenere **identità** con il notebook e togliere export CSV/HTML.
        - Questa app si aspetta che `core.esegui_analisi_completa()` **restituisca anche** la lista `figures` per il rendering in Streamlit.
        - Se vedi l'errore *NotImplementedError* significa che dobbiamo completare il porting del core (file `kq_btd_cc/core.py`).
        """
    )
