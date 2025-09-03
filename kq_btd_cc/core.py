"""Logica principale della strategia e generazione figure.

La funzione pubblica `esegui_analisi_completa(...)` restituisce un DIZIONARIO:
- "figures": lista di matplotlib.figure.Figure (grafici base)
- "figures_extra": lista di matplotlib.figure.Figure (grafici addizionali, opzionali)

Nota: Se desideri mantenere piena retro-compatibilità con codice esterno che attende una lista,
puoi usare solo il campo "figures". L'app Streamlit aggiornata gestisce il dizionario.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import STYLE_CONFIG
from .utils import (
    add_watermark,
    setup_common_axis_elements,
    setup_date_axis,
    currency_formatter,
    percentage_formatter,
    check_plot_data,
    compute_drawdown_monetary,
)
from .data_api import fetch_eodhd_ohlc


# ----------------------------------------------------------------------------
# Parametri e dataclass
# ----------------------------------------------------------------------------
DEFAULTS = {
    "EODHD_TICKER": "BTC-USD.CC",
    "START_DATE": "2015-01-01",
    "OPTIONS_PREMIUM_PERCENT": 0.05,
    "INITIAL_CAPITAL": 25_000.0,
    "ADDITIONAL_CAPITAL": 0.0,
    "CAPITAL_BOOST_PERCENT": 0.025,
    "VAR_CONFIDENCE_LEVEL": 0.99,
    "BUY_THE_DIP_DRAWDOWN_LIMIT_PERCENT": -0.90,  # blocca BTD se DD weekly < limite
    "END_DATE_OVERRIDE": None,
}


# ----------------------------------------------------------------------------
# Helpers interni (solo per extra plots)
# ----------------------------------------------------------------------------
def _compute_dd_pct(series: pd.Series) -> pd.Series:
    """Drawdown percentuale da una serie (equity o prezzo)."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    s = series.astype(float)
    cummax = s.cummax()
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (s / cummax) - 1.0
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return dd


def _rolling_return(returns_m: pd.Series, window: int = 6) -> pd.Series:
    """Rendimento rolling su finestra (composto): prod(1+r)-1."""
    if returns_m is None or returns_m.empty:
        return pd.Series(dtype=float)
    rr = (1.0 + returns_m).rolling(window=window).apply(lambda x: np.prod(x) - 1.0, raw=True)
    return rr.dropna()


def _risk_return_from_monthly(returns_m: pd.Series) -> Tuple[float, float, float]:
    """(ann_return, ann_vol, sharpe) da rendimenti mensili."""
    if returns_m is None or returns_m.empty:
        return (np.nan, np.nan, np.nan)
    mu_m = returns_m.mean()
    sd_m = returns_m.std(ddof=1)
    ann_ret = (1.0 + mu_m) ** 12 - 1.0
    ann_vol = sd_m * np.sqrt(12)
    sharpe = (ann_ret / ann_vol) if (ann_vol and ann_vol > 0) else np.nan
    return (float(ann_ret), float(ann_vol), float(sharpe))


def _dd_durations(dd_pct: pd.Series) -> List[int]:
    """Durate (in mesi) degli episodi di drawdown (<0)."""
    if dd_pct is None or dd_pct.empty:
        return []
    durations = []
    cur = 0
    for val in dd_pct.values:
        if val < 0:
            cur += 1
        else:
            if cur > 0:
                durations.append(cur)
                cur = 0
    if cur > 0:
        durations.append(cur)
    return durations


# ----------------------------------------------------------------------------
# Funzione principale
# ----------------------------------------------------------------------------
def esegui_analisi_completa(
    params_gui: Dict,
    plot_prefs: Optional[Dict] = None,
) -> Dict[str, List[plt.Figure]]:
    """Esegue la simulazione e genera le figure.

    Args:
        params_gui: dizionario parametri; chiavi come in DEFAULTS
        plot_prefs: preferenze plotting dalla UI (chiavi 'mostra_*')
    Returns:
        Dict con:
          - "figures": Lista di `matplotlib.figure.Figure` nell'ordine: 1, A, B, C, 5, 6, 7
          - "figures_extra": Lista di figure addizionali se richieste
    """
    plot_prefs = plot_prefs or {}
    show_extra = bool(plot_prefs.get("mostra_grafici_addizionali", False))
    # (Opzionale: in futuro puoi usare anche gli altri flag 'mostra_grafico_*' per gating fine.)

    # Parametri
    p = {**DEFAULTS, **(params_gui or {})}
    ticker = p["EODHD_TICKER"]
    start = p["START_DATE"]
    end = p["END_DATE_OVERRIDE"] or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    prem_pct = float(p["OPTIONS_PREMIUM_PERCENT"])  # mensile, su pacchetto iniziale
    cap0 = float(p["INITIAL_CAPITAL"])  # base covered (e cap annuo BTD)
    cap_add = float(p["ADDITIONAL_CAPITAL"])  # contributo annuo
    boost_pct = float(p["CAPITAL_BOOST_PERCENT"])  # % di cap0 aggiunta ai BTD
    dd_limit = float(p["BUY_THE_DIP_DRAWDOWN_LIMIT_PERCENT"])  # es. -0.90

    total_annual_capital = cap0 + cap_add
    boost_fixed = cap0 * boost_pct

    run_date = date.today().strftime("%Y-%m-%d")
    watermark = f"Studio by Luca De Cesare - {run_date} - Ticker: {ticker}"

    # ----------------------------------------------------------------------------
    # Download dati
    # ----------------------------------------------------------------------------
    m = fetch_eodhd_ohlc(ticker, start, end, period="m")
    if m.empty or m.shape[0] < 2:
        return {"figures": [], "figures_extra": []}

    w = fetch_eodhd_ohlc(ticker, start, end, period="w")
    weekly_ready = False
    if not w.empty and "Close" in w.columns:
        w = w.copy()
        w["Cumulative_Max"] = w["Close"].cummax()
        w["Drawdown_Asset"] = (w["Close"] - w["Cumulative_Max"]) / w["Cumulative_Max"].replace(0, np.nan)
        w["Drawdown_Asset"] = w["Drawdown_Asset"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        weekly_ready = True

    # Preparazioni mensili
    m = m.copy()
    m["Monthly_Return"] = m["Close"].pct_change()
    m["Drawdown_Signal"] = (m["Monthly_Return"] < 0).shift(1).fillna(False).astype(bool)

    # ----------------------------------------------------------------------------
    # Inizializzazioni
    # ----------------------------------------------------------------------------
    figs: List[plt.Figure] = []
    figs_extra: List[plt.Figure] = []

    eq_np = [total_annual_capital]
    eq_cash = [total_annual_capital]
    eq_reinv = [total_annual_capital]
    dates = [m.index[0]]

    # quote operative sulle linee BTD (cash/reinv) e pacchetto iniziale covered
    first_open = m["Open"].iloc[0]
    if not (np.isfinite(first_open) and first_open > 0):
        return {"figures": [], "figures_extra": []}

    initial_main_shares = cap0 / first_open if cap0 > 0 else 0.0
    shares_cash = (total_annual_capital / first_open) if total_annual_capital > 0 else 0.0
    shares_reinv = (total_annual_capital / first_open) if total_annual_capital > 0 else 0.0

    # capitali correnti (linea no premi)
    cur_cap_np = total_annual_capital
    acc_prem_cash = 0.0

    # offset per azzerare a ogni anno la base capitale e sommare i profitti al cumulato
    offset_np = 0.0
    offset_cash = 0.0
    offset_reinv = 0.0

    # Tracker annuali
    tot_btd_year = 0.0
    n_btd_year = 0
    tot_prem_cash_year = 0.0
    tot_prem_reinv_year = 0.0

    # Capitale cumulativo lordo versato per grafico 1
    inflow_np = total_annual_capital
    inflow_cash = total_annual_capital
    inflow_reinv = total_annual_capital
    inflow_np_list = [inflow_np]
    inflow_cash_list = [inflow_cash]
    inflow_reinv_list = [inflow_reinv]

    # Storico BTD mensili (lordi)
    monthly_btd = []

    # ----------------------------------------------------------------------------
    # Loop mensile
    # ----------------------------------------------------------------------------
    for i in range(1, len(m)):
        prev = m.iloc[i - 1]
        cur = m.iloc[i]
        cur_date = cur.name
        dates.append(cur_date)

        # Reset annuale
        if cur_date.year > prev.name.year:
            # aggiorna offsets con i guadagni dell'anno appena chiuso
            gain_np = eq_np[-1] - (total_annual_capital + offset_np)
            gain_cash = eq_cash[-1] - (total_annual_capital + offset_cash)
            gain_reinv = eq_reinv[-1] - (total_annual_capital + offset_reinv)
            offset_np += gain_np
            offset_cash += gain_cash
            offset_reinv += gain_reinv

            # reset capitale base e quote operative
            open_reset = cur["Open"]
            if not (np.isfinite(open_reset) and open_reset > 0):
                open_reset = prev["Close"]
                if not (np.isfinite(open_reset) and open_reset > 0):
                    break

            initial_main_shares = cap0 / open_reset if cap0 > 0 else 0.0
            shares_cash = (total_annual_capital / open_reset) if total_annual_capital > 0 else 0.0
            shares_reinv = (total_annual_capital / open_reset) if total_annual_capital > 0 else 0.0
            cur_cap_np = total_annual_capital
            acc_prem_cash = 0.0

            # reset tracker annuali
            tot_btd_year = 0.0
            n_btd_year = 0
            tot_prem_cash_year = 0.0
            tot_prem_reinv_year = 0.0

            # nuovo versamento annuale lordo
            inflow_np += total_annual_capital
            inflow_cash += total_annual_capital
            inflow_reinv += total_annual_capital

        prev_close = prev["Close"]
        cur_open = cur["Open"]
        cur_close = cur["Close"]

        # se dati non validi, replicate la equity precedente
        if not all(np.isfinite(x) for x in [prev_close, cur_open, cur_close]) or cur_open <= 0 or cur_close <= 0:
            eq_np.append(eq_np[-1])
            eq_cash.append(eq_cash[-1])
            eq_reinv.append(eq_reinv[-1])
            inflow_np_list.append(inflow_np)
            inflow_cash_list.append(inflow_cash)
            inflow_reinv_list.append(inflow_reinv)
            monthly_btd.append(0.0)
            continue

        # Premio mensile su pacchetto iniziale (covered)
        premio = (initial_main_shares * cur_open * prem_pct) if initial_main_shares > 0 else 0.0
        tot_prem_cash_year += premio
        tot_prem_reinv_year += premio

        # Linea NO premi (solo PnL quote + eventuale BTD)
        pnl_np_quotes = shares_cash * (cur_close - prev_close)
        cur_cap_np_before_btd = cur_cap_np + pnl_np_quotes

        # Trigger BTD: mese precedente negativo
        invest_btd = 0.0
        if bool(m.loc[cur_date, "Drawdown_Signal"]):
            allow = True
            dd_weekly = np.nan
            if weekly_ready:
                try:
                    dd_weekly = w["Drawdown_Asset"].asof(cur_date)
                    if pd.notna(dd_weekly) and dd_weekly < dd_limit:
                        # se drawdown settimanale va oltre (più negativo) del limite: blocca BTD
                        allow = False
                except Exception:
                    pass
            if allow:
                trig_ret = m["Monthly_Return"].iloc[i - 1]
                if pd.notna(trig_ret) and trig_ret < 0:
                    add_from_ret = abs(trig_ret) * cap0
                    potential = add_from_ret + (cap0 * boost_pct)
                    remaining = max(0.0, cap0 - tot_btd_year)  # cap annuo = cap0
                    invest_btd = min(potential, remaining)
                    if invest_btd > 1e-9:
                        tot_btd_year += invest_btd
                        n_btd_year += 1
                        inflow_np += invest_btd
                        inflow_cash += invest_btd
                        inflow_reinv += invest_btd

        # applica BTD (quote aggiuntive acquistate a close)
        if invest_btd > 0:
            cur_cap_np_before_btd += invest_btd
            add_shares = invest_btd / cur_close
            shares_cash += add_shares
            shares_reinv += add_shares

        monthly_btd.append(invest_btd)

        # Aggiorna equity NO premi
        cur_cap_np = cur_cap_np_before_btd
        eq_np_month = cur_cap_np + offset_np

        # Cash: pacchetto iniziale cappato + quote extra a mercato + premi accumulati
        base_capped = initial_main_shares * (cur_open if cur_close >= cur_open else cur_close)
        extra_shares_cash = max(0.0, shares_cash - initial_main_shares)
        extra_val_cash = extra_shares_cash * cur_close
        cur_cap_cash_quotes = base_capped + extra_val_cash
        acc_prem_cash += premio
        eq_cash_month = cur_cap_cash_quotes + acc_prem_cash + offset_cash

        # Reinvest: premi in quote
        if premio > 0:
            shares_reinv += (premio / cur_close)
        base_capped_reinv = base_capped
        extra_shares_reinv = max(0.0, shares_reinv - initial_main_shares)
        extra_val_reinv = extra_shares_reinv * cur_close
        eq_reinv_month = (base_capped_reinv + extra_val_reinv) + offset_reinv

        # Append
        eq_np.append(eq_np_month)
        eq_cash.append(eq_cash_month)
        eq_reinv.append(eq_reinv_month)

        inflow_np_list.append(inflow_np)
        inflow_cash_list.append(inflow_cash)
        inflow_reinv_list.append(inflow_reinv)

    # ----------------------------------------------------------------------------
    # Serie finali & Buy&Hold
    # ----------------------------------------------------------------------------
    idx = pd.to_datetime(dates)
    s_np = pd.Series(eq_np, index=idx, name="BTD No Premi").dropna()
    s_cash = pd.Series(eq_cash, index=idx, name="BTD+Premi(Cash)").dropna()
    s_reinv = pd.Series(eq_reinv, index=idx, name="BTD+Premi(Reinv)").dropna()

    s_btd = pd.Series(monthly_btd, index=idx[1:]) if len(idx) > 1 else pd.Series(dtype=float)

    # Buy&Hold (normalizzata a total_annual_capital sul primo close valido)
    bh = pd.Series(dtype=float, name="Buy & Hold")
    if "Close" in m.columns and m["Close"].notna().any() and total_annual_capital > 0:
        first_close_idx = m["Close"].first_valid_index()
        if first_close_idx is not None:
            ref = m.loc[first_close_idx, "Close"]
            if np.isfinite(ref) and ref > 0:
                norm = (m["Close"] / ref) * total_annual_capital
                bh = norm.rename("Buy & Hold").loc[first_close_idx:]
            else:
                bh = pd.Series([total_annual_capital] * len(m.index), index=m.index, name="Buy & Hold")
        else:
            bh = pd.Series([total_annual_capital] * len(m.index), index=m.index, name="Buy & Hold")

    dd_bh = compute_drawdown_monetary(bh)
    dd_np = compute_drawdown_monetary(s_np)
    dd_cash = compute_drawdown_monetary(s_cash)
    dd_reinv = compute_drawdown_monetary(s_reinv)

    # ----------------------------------------------------------------------------
    # Figure 1: Equity 3 strategie + capitale cumulativo investito (asse dx)
    # ----------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=STYLE_CONFIG["figure_figsize"], constrained_layout=True)
    fig1.suptitle(f"Equity Cumulativa vs Capitale Cumulativo Investito — {ticker}")
    ax1.plot(s_np.index, s_np, color=STYLE_CONFIG["colors"]["equity_no_prem"], label="BTD No Premi")
    ax1.plot(s_cash.index, s_cash, color=STYLE_CONFIG["colors"]["equity_prem_accum"], linestyle="--", label="BTD + Premi (Cash)")
    ax1.plot(s_reinv.index, s_reinv, color=STYLE_CONFIG["colors"]["equity_prem_reinvest"], linewidth=STYLE_CONFIG["line_width"]["thick"], label="BTD + Premi (Reinvest)")
    setup_common_axis_elements(ax1, title="", xlabel="Data", ylabel="Valore Portafoglio ($)", y_formatter=currency_formatter)
    setup_date_axis(ax1, major_locator_base=1, minor_locator_interval=3, minor_format="null")
    ax2 = ax1.twinx()
    inflow_ref = pd.Series(inflow_np_list, index=idx)
    ax2.plot(inflow_ref.index, inflow_ref, color=STYLE_CONFIG["colors"]["investment"], linestyle=":", label="Cap. Iniziale + BTD Cumul.")
    ax2.set_ylabel("Cap. Iniziale + BTD Cumul. ($)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(False)
    ax2.set_ylim(bottom=0)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    add_watermark(fig1, watermark)
    figs.append(fig1)

    # ----------------------------------------------------------------------------
    # A/B/C: Equity + Drawdown per singola strategia
    # ----------------------------------------------------------------------------
    def _fig_eq_dd(eq: pd.Series, dd: pd.Series, label: str, color: str) -> plt.Figure:
        fig, (ax_eq, ax_dd) = plt.subplots(
            nrows=2, ncols=1, figsize=STYLE_CONFIG["figure_figsize"], sharex=True,
            constrained_layout=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.suptitle(f"{label} — Equity e Drawdown Monetario")
        # Equity
        ax_eq.plot(eq.index, eq, color=color, label=f"Equity {label}")
        setup_common_axis_elements(ax_eq, "", "", "Equity ($)", y_formatter=currency_formatter)
        ax_eq.legend(loc="upper left")
        # Drawdown
        if not dd.empty:
            ax_dd.plot(dd.index, dd, color=STYLE_CONFIG["colors"]["drawdown_portfolio_usd"], linewidth=STYLE_CONFIG["line_width"]["thin"], label="Drawdown ($)")
            ax_dd.fill_between(dd.index, dd, 0, where=(dd < 0), color=STYLE_CONFIG["colors"]["drawdown_portfolio_usd"], alpha=0.25)
        setup_common_axis_elements(ax_dd, "", "Data", "Drawdown ($)", y_formatter=currency_formatter)
        top_lim = 0
        if not dd.empty and dd.min() < 0:
            top_lim = abs(dd.min()) * 0.05
        ax_dd.set_ylim(top=top_lim)
        setup_date_axis(ax_dd, major_locator_base=1, minor_locator_interval=3, minor_format="null")
        add_watermark(fig, watermark)
        return fig

    figs.append(_fig_eq_dd(s_np, dd_np, "BTD No Premi", STYLE_CONFIG["colors"]["equity_no_prem"]))
    figs.append(_fig_eq_dd(s_cash, dd_cash, "BTD + Premi (Cash)", STYLE_CONFIG["colors"]["equity_prem_accum"]))
    figs.append(_fig_eq_dd(s_reinv, dd_reinv, "BTD + Premi (Reinvest)", STYLE_CONFIG["colors"]["equity_prem_reinvest"]))

    # ----------------------------------------------------------------------------
    # 5. Reinvestimenti mensili BTD (bar)
    # ----------------------------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=STYLE_CONFIG["figure_figsize"], constrained_layout=True)
    fig5.suptitle(f"Reinvestimenti Mensili (BTD + Boost) — {ticker}")
    if check_plot_data(pd.Series(monthly_btd)):
        pos = pd.Series(monthly_btd, index=idx[1:])
        pos = pos[pos > 1e-6]
        if not pos.empty:
            barw = max(15, int(400 / max(1, len(pos))))
            barw = min(barw, 25)
            ax5.bar(pos.index, pos.values, width=barw, color=STYLE_CONFIG["colors"]["reinvest"], label="Reinvestimento BTD Mensile")
    setup_common_axis_elements(ax5, "", "Data", "Importo Reinvestito ($)", y_formatter=currency_formatter)
    setup_date_axis(ax5, major_locator_base=1, minor_locator_interval=3, minor_format="null")
    ax5.legend(loc="upper right")
    ax5.set_ylim(bottom=0)
    add_watermark(fig5, watermark)
    figs.append(fig5)

    # ----------------------------------------------------------------------------
    # 6. Drawdown settimanale asset vs limite BTD
    # ----------------------------------------------------------------------------
    fig6, ax6 = plt.subplots(figsize=STYLE_CONFIG["figure_figsize"], constrained_layout=True)
    fig6.suptitle(f"Drawdown Settimanale Asset {ticker} (%) vs Limite BTD")
    if weekly_ready and "Drawdown_Asset" in w.columns and w["Drawdown_Asset"].notna().any():
        dd_pct_asset = w["Drawdown_Asset"] * 100
        ax6.plot(dd_pct_asset.index, dd_pct_asset.values, color=STYLE_CONFIG["colors"]["drawdown_asset"], linewidth=STYLE_CONFIG["line_width"]["thin"], label="Drawdown Asset (%)")
        ax6.fill_between(dd_pct_asset.index, dd_pct_asset.values, 0, where=(dd_pct_asset.values < 0), color=STYLE_CONFIG["colors"]["drawdown_asset"], alpha=0.3)
        limit_val = dd_limit * 100
        ax6.axhline(limit_val, color="black", linestyle=":", linewidth=1.5, label=f"Limite Attivazione BTD ({limit_val:.0f}%)")
    setup_common_axis_elements(ax6, "", "Data", "Drawdown Asset (%)", y_formatter=percentage_formatter)
    setup_date_axis(ax6, major_locator_base=1, minor_locator_interval=None)
    ax6.legend(loc="lower left")
    ax6.set_ylim(top=5)
    add_watermark(fig6, watermark)
    figs.append(fig6)

    # ----------------------------------------------------------------------------
    # 7. Confronto finale con Buy & Hold
    # ----------------------------------------------------------------------------
    fig7, ax7 = plt.subplots(figsize=STYLE_CONFIG["figure_figsize"], constrained_layout=True)
    fig7.suptitle(f"Confronto Performance: B&H vs Strategie BTD — {ticker}")
    if check_plot_data(bh):
        ax7.plot(bh.index, bh, color=STYLE_CONFIG["colors"]["buy_hold"], linestyle="--", label="Buy & Hold")
    ax7.plot(s_np.index, s_np, color=STYLE_CONFIG["colors"]["equity_no_prem"], label="BTD No Premi")
    ax7.plot(s_cash.index, s_cash, color=STYLE_CONFIG["colors"]["equity_prem_accum"], linestyle="-.", label="BTD + Premi (Cash)")
    ax7.plot(s_reinv.index, s_reinv, color=STYLE_CONFIG["colors"]["equity_prem_reinvest"], linewidth=STYLE_CONFIG["line_width"]["thick"], label="BTD + Premi (Reinvest)")
    setup_common_axis_elements(ax7, "", "Data", "Valore Portafoglio ($)", y_formatter=currency_formatter)
    setup_date_axis(ax7, major_locator_base=1, minor_locator_interval=3, minor_format="null")
    ax7.legend(loc="upper left")
    add_watermark(fig7, watermark)
    figs.append(fig7)

    # ============================================================================
    # GRAFICI ADDIZIONALI (opzionali)
    # ============================================================================
    if show_extra:
        # Rendimenti mensili (NB: derivati da equity nominale -> includono flussi; per analisi "pura"
        # si potrebbe normalizzare per inflow. Qui l'obiettivo è visuale).
        r_np = s_np.pct_change().dropna()
        r_cash = s_cash.pct_change().dropna()
        r_reinv = s_reinv.pct_change().dropna()
        r_bh = bh.pct_change().dropna() if not bh.empty else pd.Series(dtype=float)

        # 1) VIOLIN dei rendimenti mensili
        fig_v, ax_v = plt.subplots(figsize=STYLE_CONFIG["figure_figsize"], constrained_layout=True)
        fig_v.suptitle(f"Distribuzione Rendimenti Mensili — {ticker}")
        data_v = [r for r in [r_bh, r_np, r_cash, r_reinv] if not r.empty]
        labels_v = []
        if not r_bh.empty: labels_v.append("B&H")
        labels_v += ["BTD No Premi", "BTD+Premi(Cash)", "BTD+Premi(Reinv)"][: len(data_v) - (1 if not r_bh.empty else 0)]
        parts = ax_v.violinplot([d.values for d in data_v], showmeans=True, showmedians=False, showextrema=True)
        ax_v.set_xticks(range(1, len(labels_v) + 1))
        ax_v.set_xticklabels(labels_v, rotation=0)
        setup_common_axis_elements(ax_v, "", "Strategie", "Rendimento Mensile", y_formatter=percentage_formatter)
        add_watermark(fig_v, watermark)
        figs_extra.append(fig_v)

        # 2) ROLLING (6 e 12 mesi) dei rendimenti composti
        fig_roll, ax_roll = plt.subplots(figsize=STYLE_CONFIG["figure_figsize"], constrained_layout=True)
        fig_roll.suptitle(f"Rolling Return (6M / 12M) — {ticker}")
        for series, label, color in [
            (r_bh, "B&H", STYLE_CONFIG["colors"]["buy_hold"]) if not r_bh.empty else (None, None, None),
            (r_np, "BTD No Premi", STYLE_CONFIG["colors"]["equity_no_prem"]),
            (r_cash, "BTD + Premi (Cash)", STYLE_CONFIG["colors"]["equity_prem_accum"]),
            (r_reinv, "BTD + Premi (Reinvest)", STYLE_CONFIG["colors"]["equity_prem_reinvest"]),
        ]:
            if series is None or series.empty:
                continue
            rr6 = _rolling_return(series, 6)
            rr12 = _rolling_return(series, 12)
            if not rr6.empty:
                ax_roll.plot(rr6.index, rr6.values, linestyle="-", label=f"{label} — 6M", color=color, alpha=0.8)
            if not rr12.empty:
                ax_roll.plot(rr12.index, rr12.values, linestyle="--", label=f"{label} — 12M", color=color, alpha=0.8)
        setup_common_axis_elements(ax_roll, "", "Data", "Rendimento Rolling", y_formatter=percentage_formatter)
        setup_date_axis(ax_roll, major_locator_base=1, minor_locator_interval=3, minor_format="null")
        ax_roll.legend(loc="upper left")
        add_watermark(fig_roll, watermark)
        figs_extra.append(fig_roll)

        # 3) RISK/RETURN scatter (ann. return vs ann. vol) + Sharpe
        fig_rr, ax_rr = plt.subplots(figsize=STYLE_CONFIG["figure_figsize"], constrained_layout=True)
        fig_rr.suptitle(f"Risk / Return (ann.) — {ticker}")
        for series, label, color in [
            (r_bh, "B&H", STYLE_CONFIG["colors"]["buy_hold"]) if not r_bh.empty else (None, None, None),
            (r_np, "BTD No Premi", STYLE_CONFIG["colors"]["equity_no_prem"]),
            (r_cash, "BTD + Premi (Cash)", STYLE_CONFIG["colors"]["equity_prem_accum"]),
            (r_reinv, "BTD + Premi (Reinvest)", STYLE_CONFIG["colors"]["equity_prem_reinvest"]),
        ]:
            if series is None or series.empty:
                continue
            ann_r, ann_v, shr = _risk_return_from_monthly(series)
            if not (np.isnan(ann_r) or np.isnan(ann_v)):
                ax_rr.scatter([ann_v], [ann_r], label=f"{label} (Sh {shr:.2f})", s=60)
        setup_common_axis_elements(ax_rr, "", "Volatilità Annua", "Rendimento Annuo", y_formatter=percentage_formatter)
        ax_rr.xaxis.set_major_formatter(percentage_formatter)
        ax_rr.legend(loc="best")
        add_watermark(fig_rr, watermark)
        figs_extra.append(fig_rr)

        # 4) Drawdown Duration (mesi)
        fig_ddur, ax_ddur = plt.subplots(figsize=STYLE_CONFIG["figure_figsize"], constrained_layout=True)
        fig_ddur.suptitle(f"Durata degli Episodi di Drawdown (mesi) — {ticker}")
        for series, label, color in [
            (s_np, "BTD No Premi", STYLE_CONFIG["colors"]["equity_no_prem"]),
            (s_cash, "BTD + Premi (Cash)", STYLE_CONFIG["colors"]["equity_prem_accum"]),
            (s_reinv, "BTD + Premi (Reinvest)", STYLE_CONFIG["colors"]["equity_prem_reinvest"]),
        ]:
            dd_pct = _compute_dd_pct(series)
            durs = _dd_durations(dd_pct)
            if durs:
                # Istogramma "lineare" per confronto
                values, bins = np.histogram(durs, bins=range(1, max(durs) + 2))
                centers = (bins[:-1] + bins[1:]) / 2
                ax_ddur.plot(centers, values, label=label)
        ax_ddur.set_xlabel("Durata episodio (mesi)")
        ax_ddur.set_ylabel("Frequenza")
        ax_ddur.legend(loc="upper right")
        add_watermark(fig_ddur, watermark)
        figs_extra.append(fig_ddur)

        # 5) Dashboard sintetica KPI (CAGR/Vol/Sharpe/MaxDD%)
        fig_dash, ax_dash = plt.subplots(figsize=STYLE_CONFIG["figure_figsize"], constrained_layout=True)
        fig_dash.suptitle(f"Dashboard KPI (ann.) — {ticker}")

        def _kpi_block(ax, x, y, text):
            ax.text(x, y, text, va="top", ha="left", fontsize=11, family="monospace")

        # KPI per ciascuna linea
        def _kpis(series_eq: pd.Series, label: str) -> str:
            r_m = series_eq.pct_change().dropna()
            ann_r, ann_v, shr = _risk_return_from_monthly(r_m)
            dd_pct = _compute_dd_pct(series_eq)
            maxdd = dd_pct.min() if not dd_pct.empty else np.nan
            return (
                f"{label}\n"
                f"CAGR: {ann_r:,.2%}\n"
                f"Vol:  {ann_v:,.2%}\n"
                f"Sharpe: {shr:,.2f}\n"
                f"MaxDD: {maxdd:,.2%}"
            )

        ax_dash.axis("off")
        txt = "\n\n".join([
            _kpis(s_np, "BTD No Premi"),
            _kpis(s_cash, "BTD + Premi (Cash)"),
            _kpis(s_reinv, "BTD + Premi (Reinvest)"),
            _kpis(bh, "Buy & Hold") if not bh.empty else "Buy & Hold\n(n.d.)",
        ])
        _kpi_block(ax_dash, 0.05, 0.95, txt)
        add_watermark(fig_dash, watermark)
        figs_extra.append(fig_dash)

    # Return strutturato (base + extra)
    return {"figures": figs, "figures_extra": figs_extra}
