# app.py
import streamlit as st
import matplotlib.pyplot as plt

from kq_btd_cc.core import esegui_analisi_completa
from kq_btd_cc.style import STYLE_CONFIG

st.set_page_config(page_title="Boosted Covered Call (Mensile)", layout="wide")
st.title("Boosted Covered Call — Studio Mensile")

with st.sidebar:
    st.header("Parametri")
    ticker = st.text_input("Ticker EODHD", value="BTC-USD.CC", help="Formato EODHD, es. AAPL.US, BTC-USD.CC")
    start_date = st.date_input("Data Inizio", value=st.session_state.get("_default_start", None) or None, help="Se vuoto, usa 2015-01-01")
    end_date = st.date_input("Data Fine (opzionale)", value=None, help="Se vuoto, usa ieri")

    col1, col2 = st.columns(2)
    with col1:
        initial_capital = st.number_input("Capitale Iniziale ($)", min_value=0.0, value=25_000.0, step=1_000.0)
        additional_capital = st.number_input("Capitale Addizionale Ann. ($)", min_value=0.0, value=0.0, step=500.0)
    with col2:
        premium_pct = st.number_input("Premio Mensile (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%0.3f")
        boost_pct = st.number_input("Boost su BTD (% di Cap. Iniziale)", min_value=0.0, max_value=1.0, value=0.025, step=0.005, format="%0.3f")

    var_conf = st.slider("Livello VaR (solo per report)", min_value=0.90, max_value=0.999, value=0.99, step=0.005)
    dd_limit = st.slider("Limite Drawdown per attivare BTD (sett., %)", min_value=-0.99, max_value=-0.10, value=-0.90, step=0.01)

    run = st.button("Esegui Analisi", type="primary")

params_gui = {
    "EODHD_TICKER": ticker,
    "START_DATE": (start_date.isoformat() if start_date else "2015-01-01"),
    "END_DATE_OVERRIDE": (end_date.isoformat() if end_date else None),
    "OPTIONS_PREMIUM_PERCENT": float(premium_pct),
    "INITIAL_CAPITAL": float(initial_capital),
    "ADDITIONAL_CAPITAL": float(additional_capital),
    "CAPITAL_BOOST_PERCENT": float(boost_pct),
    "VAR_CONFIDENCE_LEVEL": float(var_conf),
    "BUY_THE_DIP_DRAWDOWN_LIMIT_PERCENT": float(dd_limit),
}

plot_prefs = {
    # placeholder per futuri interruttori grafici/HTML
}

if run:
    with st.spinner("Esecuzione analisi in corso..."):
        figures = esegui_analisi_completa(params_gui=params_gui, plot_prefs=plot_prefs)

    if not figures:
        st.error("Nessuna figura generata. Controlla i parametri e la chiave API di EODHD nei secrets.")
    else:
        for i, fig in enumerate(figures, start=1):
            st.pyplot(fig)

st.markdown(

    unsafe_allow_html=True,
)
