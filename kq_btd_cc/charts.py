"""Grafici Plotly a tema scuro per la dashboard.

Ogni funzione restituisce una `go.Figure` gia' impaginata e non tocca lo stato
globale. I colori arrivano tutti da `style.PALETTE`, cosi' che una serie abbia
sempre lo stesso colore in ogni grafico in cui compare.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .style import (PALETTE, COLORE_BENCHMARK, COLORE_VARIANTE, SCALA_DIVERGENTE,
                    TEMPLATE_NAME, FONT_MONO)
from .metrics import drawdown_durations

MESI_IT = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]

# hovertemplate riusabili
HT_VAL = "%{fullData.name}: <b>$%{y:,.0f}</b><extra></extra>"
HT_PCT = "%{fullData.name}: <b>%{y:.2%}</b><extra></extra>"


# ----------------------------------------------------------------------------
# Impaginazione comune
# ----------------------------------------------------------------------------
def _layout(fig: go.Figure, titolo: str, sottotitolo: str = "", altezza: int = 460,
            legenda: bool = True) -> go.Figure:
    testo = titolo
    if sottotitolo:
        testo += (f"<br><span style='font-size:12px;color:{PALETTE['text_muted']};"
                  f"font-weight:400'>{sottotitolo}</span>")
    fig.update_layout(
        template=TEMPLATE_NAME, title=dict(text=testo), height=altezza,
        showlegend=legenda,
        margin=dict(l=10, r=10, t=86 if sottotitolo else 64, b=10),
    )
    return fig


def _asse_tempo(fig: go.Figure, selettore: bool = True, riga: Optional[int] = None) -> go.Figure:
    cfg: Dict[str, Any] = dict(showgrid=False, ticklabelmode="period")
    if selettore:
        cfg["rangeselector"] = dict(
            buttons=[
                dict(count=1, label="1A", step="year", stepmode="backward"),
                dict(count=3, label="3A", step="year", stepmode="backward"),
                dict(count=5, label="5A", step="year", stepmode="backward"),
                dict(step="all", label="Tutto"),
            ],
            bgcolor=PALETTE["surface"], activecolor=PALETTE["vol"],
            font=dict(color=PALETTE["text_muted"], size=11),
            bordercolor="rgba(148,163,184,0.2)", borderwidth=1,
            x=1.0, xanchor="right", y=1.12, yanchor="bottom",
        )
    if riga is None:
        fig.update_xaxes(**cfg)
    else:
        fig.update_xaxes(**cfg, row=riga, col=1)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    """'#38BDF8' -> 'rgba(56,189,248,0.25)'."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _vuoto(messaggio: str, altezza: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=messaggio, showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(color=PALETTE["text_muted"], size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, "", altezza=altezza, legenda=False)


def _bande_anni(fig: go.Figure, index: pd.DatetimeIndex, riga: Optional[int] = None) -> None:
    """Bande alternate a ogni anno: rende visibile il reset annuale."""
    if index is None or len(index) == 0:
        return
    anni = sorted({int(t.year) for t in index})
    for k, anno in enumerate(anni):
        if k % 2 == 0:
            continue
        kw = dict(x0=pd.Timestamp(anno, 1, 1), x1=pd.Timestamp(anno + 1, 1, 1),
                  fillcolor="rgba(148,163,184,0.045)", line_width=0, layer="below")
        if riga is None:
            fig.add_vrect(**kw)
        else:
            fig.add_vrect(**kw, row=riga, col=1)


def _fuori_scala(serie: pd.Series, riferimento: float, fattore: float = 8.0) -> bool:
    """La serie schiaccerebbe tutte le altre se disegnata sullo stesso asse?

    Succede sui sottostanti che si moltiplicano per ordini di grandezza: un buy
    and hold che non liquida mai arriva a valori tali da rendere piatte tutte le
    altre curve. Meglio toglierlo e dirlo, che disegnare un grafico illeggibile.
    """
    if serie is None or serie.empty or not np.isfinite(riferimento) or riferimento <= 0:
        return False
    picco = float(np.nanmax(np.abs(serie.values)))
    return np.isfinite(picco) and picco > fattore * abs(riferimento)


def _nota_fuori_scala(fig: go.Figure, testo: str) -> None:
    fig.add_annotation(
        text=testo, xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left", yanchor="top",
        showarrow=False, align="left",
        font=dict(color=PALETTE["bh_flussi"], size=11),
        bgcolor="rgba(17,24,39,0.85)", borderpad=6)


def _benchmark(risultato: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Il termine di paragone a parita' di mandato, se disponibile."""
    b = risultato.get("benchmark")
    if not b or not isinstance(b.get("monthly"), pd.DataFrame) or b["monthly"].empty:
        return None
    return b


def _traccia_benchmark(fig: go.Figure, x, y, nome: str = "", spesso: bool = False,
                       riempi: bool = False, riga: Optional[int] = None) -> None:
    """Linea del benchmark, sempre tratteggiata e dello stesso colore ovunque."""
    kw: Dict[str, Any] = dict(
        x=x, y=y, name=nome or "Buy & Hold (stesso ciclo annuale)",
        line=dict(color=COLORE_BENCHMARK, width=2.4 if spesso else 2.0, dash="dash"))
    if riempi:
        kw["fill"] = "tozeroy"
        kw["fillcolor"] = _rgba(COLORE_BENCHMARK, 0.12)
    traccia = go.Scatter(**kw)
    if riga is None:
        fig.add_trace(traccia)
    else:
        fig.add_trace(traccia, row=riga, col=1)


def _segna_ultimo_versamento(fig: go.Figure, df: pd.DataFrame,
                             riga: Optional[int] = None) -> None:
    """Marca il punto oltre il quale la strategia non ha piu' chiesto denaro.

    La curva del capitale versato diventa piatta quando i profitti bastano a
    coprire il capitale fisso di ogni gennaio e gli acquisti sui cali. Senza
    questa nota la linea orizzontale sembra un errore, mentre e' il momento in
    cui la strategia comincia ad autofinanziarsi.
    """
    if "versamento_mese" not in df.columns:
        return
    con_versamento = df.index[df["versamento_mese"] > 1e-9]
    if len(con_versamento) == 0 or con_versamento[-1] >= df.index[-4:][0]:
        return
    quando = con_versamento[-1]
    totale = float(df["versamenti_cum"].iloc[-1])
    kw = dict(x=quando, y=totale,
              text=(f"da qui in poi non serve altro denaro<br>"
                    f"totale versato ${totale:,.0f}"),
              showarrow=True, arrowhead=0, arrowsize=1, arrowwidth=1,
              arrowcolor=PALETTE["versamenti"], ax=30, ay=-34, xanchor="left",
              align="left", font=dict(color=PALETTE["text_muted"], size=10))
    if riga is None:
        fig.add_annotation(**kw)
    else:
        fig.add_annotation(**kw, row=riga, col=1)


def _serie_varianti(risultato: Dict[str, Any], colonna: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for chiave, res in risultato.get("varianti", {}).items():
        df = res.get("monthly")
        if isinstance(df, pd.DataFrame) and not df.empty and colonna in df.columns:
            out[chiave] = df[colonna]
    return out


# ============================================================================
# 1. Confronto di tutte le equity
# ============================================================================
def fig_confronto_equity(risultato: Dict[str, Any], log: bool = False,
                         mostra_versamenti: bool = True) -> go.Figure:
    """Tutte le curve sullo stesso asse: le tre varianti piu' i due Buy & Hold."""
    varianti = risultato.get("varianti", {})
    if not varianti:
        return _vuoto("Nessun dato disponibile")

    fig = go.Figure()
    rif = next(iter(varianti.values()))["monthly"]
    scala = max(float(np.nanmax(res["monthly"]["valore_portafoglio"].values))
                for res in varianti.values() if not res["monthly"].empty)
    fuori = []

    # Il benchmark a parita' di mandato, sempre presente
    bm = _benchmark(risultato)
    if bm is not None:
        s = bm["monthly"]["valore_portafoglio"]
        _traccia_benchmark(fig, s.index, s.values, bm["label"], spesso=True)
        fig.data[-1].update(hovertemplate=HT_VAL)
        scala = max(scala, float(np.nanmax(s.values)))

    # I due buy and hold che non liquidano mai: su un sottostante che si
    # moltiplica per ordini di grandezza escono dal grafico. In scala
    # logaritmica ci stanno, altrimenti li si segnala e basta.
    for serie, nome, colore, tratto in (
        (risultato.get("mercato", {}).get("bh_semplice"),
         "Buy & Hold (solo capitale iniziale)", PALETTE["bh_semplice"], "dash"),
        (rif["bh_stessi_flussi"] if "bh_stessi_flussi" in rif.columns else None,
         "Buy & Hold (stessi versamenti)", PALETTE["bh_flussi"], "dot"),
    ):
        if not isinstance(serie, pd.Series) or serie.empty:
            continue
        if log or not _fuori_scala(serie, scala):
            fig.add_trace(go.Scatter(
                x=serie.index, y=serie.values, name=nome,
                line=dict(color=colore, width=1.8, dash=tratto), hovertemplate=HT_VAL))
        else:
            fuori.append(f"{nome}: {serie.iloc[-1]:,.0f} $")

    for chiave, res in varianti.items():
        df = res["monthly"]
        if df.empty:
            continue
        spesso = chiave == "premi_reinvest"
        fig.add_trace(go.Scatter(
            x=df.index, y=df["valore_portafoglio"], name=res["label"],
            line=dict(color=COLORE_VARIANTE.get(chiave, PALETTE["text"]),
                      width=3.0 if spesso else 2.1),
            hovertemplate=HT_VAL))

    if mostra_versamenti and "versamenti_cum" in rif.columns:
        s = rif["versamenti_cum"]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name="Capitale versato (cumulato)",
            line=dict(color=PALETTE["versamenti"], width=1.4, dash="longdash"),
            fill="tozeroy", fillcolor="rgba(100,116,139,0.10)",
            hovertemplate=HT_VAL))
        _segna_ultimo_versamento(fig, rif)
        if "capitale_impiegato_anno" in rif.columns:
            s_imp = rif["capitale_impiegato_anno"]
            fig.add_trace(go.Scatter(
                x=s_imp.index, y=s_imp.values, name="Capitale impiegato nell'anno",
                line=dict(color=PALETTE["btd"], width=1.4, shape="hv"),
                hovertemplate=HT_VAL))

    _bande_anni(fig, rif.index)
    fig.update_yaxes(tickprefix="$", type="log" if log else "linear",
                     title_text="Valore del conto")
    _asse_tempo(fig)
    fig = _layout(
        fig, "Confronto delle equity",
        "Valore totale del conto (quote a mercato piu' cassa). La banda grigia e' il denaro "
        "entrato dall'esterno, che non si azzera mai perche' e' il metro dell'utile; la "
        "linea a gradini e' il capitale davvero al lavoro, che invece riparte a ogni gennaio.",
        altezza=520)
    if fuori:
        _nota_fuori_scala(
            fig, "Fuori scala, non disegnati: " + " · ".join(fuori)
                 + ".<br>Non liquidano mai, mentre la strategia chiude ogni dodici mesi. "
                   "Attiva la scala logaritmica nella sidebar per vederli.")
    return fig


# ============================================================================
# 2. Utile netto dei versamenti
# ============================================================================
def fig_pnl_netto(risultato: Dict[str, Any]) -> go.Figure:
    serie = _serie_varianti(risultato, "pnl_netto")
    if not serie:
        return _vuoto("Nessun dato disponibile")

    fig = go.Figure()
    rif = next(iter(risultato["varianti"].values()))["monthly"]
    scala = max(float(np.nanmax(np.abs(s.values))) for s in serie.values())

    # Il confronto a parita' di mandato, sempre sulla stessa scala della strategia
    bm = _benchmark(risultato)
    if bm is not None:
        s = bm["monthly"]["pnl_netto"]
        _traccia_benchmark(fig, s.index, s.values, bm["label"], spesso=True)
        fig.data[-1].update(hovertemplate=HT_VAL)
        scala = max(scala, float(np.nanmax(np.abs(s.values))))

    nota_extra = ""
    if "bh_stessi_flussi" in rif.columns:
        s = rif["bh_stessi_flussi"] - rif["versamenti_cum"]
        if _fuori_scala(s, scala):
            nota_extra = (f"Buy &amp; Hold che non liquida mai: {s.iloc[-1]:,.0f} $, "
                          f"fuori scala di {abs(s.iloc[-1]) / max(scala, 1):,.0f} volte. "
                          f"Tiene per sempre le quote comprate all'inizio, mentre la "
                          f"strategia liquida ogni dodici mesi: non e' un confronto "
                          f"a parita di mandato.")
        else:
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, name="Buy & Hold (stessi versamenti)",
                line=dict(color=PALETTE["bh_flussi"], width=1.8, dash="dot"),
                hovertemplate=HT_VAL))

    for chiave, s in serie.items():
        label = risultato["varianti"][chiave]["label"]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=label,
            line=dict(color=COLORE_VARIANTE.get(chiave, PALETTE["text"]),
                      width=3.0 if chiave == "premi_reinvest" else 2.1),
            hovertemplate=HT_VAL))

    fig.add_hline(y=0, line=dict(color=PALETTE["axis"], width=1.2))
    _bande_anni(fig, rif.index)
    fig.update_yaxes(tickprefix="$", title_text="Utile netto")
    _asse_tempo(fig)
    fig = _layout(
        fig, "Utile netto dei versamenti",
        "Valore del conto meno tutto il denaro versato dall'esterno, BTD compresi. "
        "Sotto lo zero la strategia sta perdendo, per quanto il conto sia cresciuto.",
        altezza=460)
    if nota_extra:
        _nota_fuori_scala(fig, nota_extra)
    return fig


# ============================================================================
# 3. Equity + drawdown di una singola variante
# ============================================================================
def fig_equity_drawdown(risultato: Dict[str, Any], chiave: str) -> go.Figure:
    res = risultato.get("varianti", {}).get(chiave)
    if not res or res["monthly"].empty:
        return _vuoto("Nessun dato disponibile")
    df = res["monthly"]
    colore = COLORE_VARIANTE.get(chiave, PALETTE["text"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        row_heights=[0.68, 0.32],
                        subplot_titles=("", ""))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["versamenti_cum"], name="Capitale versato (cumulato)",
        line=dict(color=PALETTE["versamenti"], width=1.3, dash="longdash"),
        fill="tozeroy", fillcolor="rgba(100,116,139,0.12)",
        hovertemplate=HT_VAL), row=1, col=1)
    _segna_ultimo_versamento(fig, df, riga=1)
    # Il capitale davvero al lavoro nel ciclo in corso: questo si azzera a ogni
    # gennaio, mentre il versato cumulato non torna mai indietro.
    if "capitale_impiegato_anno" in df.columns:
        s_imp = df["capitale_impiegato_anno"]
        fig.add_trace(go.Scatter(
            x=s_imp.index, y=s_imp.values, name="Capitale impiegato nell'anno",
            line=dict(color=PALETTE["btd"], width=1.5, shape="hv"),
            hovertemplate=HT_VAL), row=1, col=1)
    bm = _benchmark(risultato)
    if bm is not None:
        sb = bm["monthly"]["valore_portafoglio"]
        _traccia_benchmark(fig, sb.index, sb.values, bm["label"], riga=1)
        fig.data[-1].update(hovertemplate=HT_VAL)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["valore_portafoglio"], name=res["label"],
        line=dict(color=colore, width=2.6), hovertemplate=HT_VAL), row=1, col=1)

    # Il drawdown del pannello basso e' in PERCENTUALE, non in valuta.
    # Confrontare due portafogli in dollari assoluti inganna: i Buy-The-Dip fanno
    # lavorare piu' capitale, quindi la strategia puo' perdere piu' dollari pur
    # perdendo una percentuale molto minore. La percentuale e' l'unica misura
    # confrontabile, ed e' la stessa che finisce nella tabella delle metriche.
    dd = df["dd_twr_pct"]
    if bm is not None:
        sb = bm["monthly"]["dd_twr_pct"]
        _traccia_benchmark(fig, sb.index, sb.values,
                           f"Drawdown del {bm['label']}", riempi=True, riga=2)
        fig.data[-1].update(hovertemplate=HT_PCT)
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name="Drawdown del conto",
        line=dict(color=PALETTE["drawdown"], width=1.6),
        fill="tozeroy", fillcolor="rgba(244,63,94,0.22)",
        hovertemplate=HT_PCT), row=2, col=1)

    peggiore = dd.idxmin() if dd.notna().any() else None
    if peggiore is not None and dd.min() < 0:
        in_valuta = float(df["dd_valore"].min())
        fig.add_annotation(
            x=peggiore, y=dd.min(), row=2, col=1,
            text=f"{dd.min():.1%} (${in_valuta:,.0f})",
            showarrow=True, arrowhead=0, arrowcolor=PALETTE["drawdown"], ax=0, ay=24,
            font=dict(color=PALETTE["drawdown"], size=11, family=FONT_MONO))

    _bande_anni(fig, df.index, riga=1)
    fig.update_yaxes(tickprefix="$", title_text="Valore", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", title_text="Drawdown", row=2, col=1)
    _asse_tempo(fig, selettore=False, riga=1)
    _asse_tempo(fig, selettore=False, riga=2)

    capitale = float(((df["quote_coperte"] + df["quote_extra"]) * df["close"]).mean())
    sotto = ("In alto il conto, il denaro entrato dall'esterno (che non si azzera mai: e' il "
             "metro dell'utile) e il capitale al lavoro nel ciclo in corso (che riparte a "
             "ogni gennaio); in basso la discesa dal massimo, in percentuale.")
    if bm is not None:
        cap_bm = float(((bm["monthly"]["quote_coperte"] + bm["monthly"]["quote_extra"])
                        * bm["monthly"]["close"]).mean())
        if cap_bm > 0 and abs(capitale / cap_bm - 1) > 0.05:
            sotto += (f" La strategia impiega in media {capitale / cap_bm - 1:+.0%} di "
                      f"capitale rispetto al benchmark, per via dei Buy-The-Dip: per questo "
                      f"il confronto va fatto in percentuale e non in dollari.")
    return _layout(fig, f"{res['label']} — valore e drawdown", sotto, altezza=560)


# ============================================================================
# 4. Underwater comparato
# ============================================================================
def fig_underwater(risultato: Dict[str, Any]) -> go.Figure:
    serie = _serie_varianti(risultato, "dd_twr_pct")
    if not serie:
        return _vuoto("Nessun dato disponibile")
    fig = go.Figure()
    rif = next(iter(risultato["varianti"].values()))["monthly"]
    bm = _benchmark(risultato)
    if bm is not None:
        s = bm["monthly"]["dd_twr_pct"]
        _traccia_benchmark(fig, s.index, s.values, bm["label"], riempi=True)
        fig.data[-1].update(hovertemplate=HT_PCT)
    if "bh_dd_twr_pct" in rif.columns:
        s = rif["bh_dd_twr_pct"]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name="Buy & Hold (stessi versamenti)",
            line=dict(color=PALETTE["bh_flussi"], width=2.0, dash="dot"),
            fill="tozeroy", fillcolor=_rgba(PALETTE["bh_flussi"], 0.16),
            hovertemplate=HT_PCT))

    for chiave, s in serie.items():
        colore = COLORE_VARIANTE.get(chiave, PALETTE["text"])
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=risultato["varianti"][chiave]["label"],
            line=dict(color=colore, width=1.9),
            fill="tozeroy", fillcolor=_rgba(colore, 0.13),
            hovertemplate=HT_PCT))
    fig.update_yaxes(tickformat=".0%", title_text="Drawdown")
    _asse_tempo(fig)
    return _layout(fig, "Drawdown a confronto",
                   "Percentuale calcolata sul rendimento time-weighted, quindi ripulita dai "
                   "versamenti. La linea punteggiata e' il Buy & Hold con gli stessi versamenti: "
                   "il premio incassato in cash dovrebbe tenerla sopra le altre.",
                   altezza=420)


# ============================================================================
# 4-bis. Verdetto contro il Buy & Hold
# ============================================================================
def fig_verdetto_vs_bh(risultato: Dict[str, Any]) -> go.Figure:
    """Le due domande che contano: quanto drawdown in meno, quanto rendimento in piu'."""
    # Se il ciclo annuale e' disponibile e' quello il metro giusto: stesso
    # mandato, stessa liquidazione a dicembre, unica differenza le opzioni e i BTD.
    prima = next(iter(risultato.get("varianti", {}).values()), {}).get("metrics") or {}
    usa_ciclo = prima.get("ciclo_rendimento_medio") is not None
    k_dd = "riduzione_dd_vs_ciclo" if usa_ciclo else "riduzione_dd_vs_bh"
    k_rend = "extra_rendimento_vs_ciclo"
    k_mio_dd, k_suo_dd = "max_dd_pct", ("ciclo_max_dd_pct" if usa_ciclo else "bh_max_dd_pct")
    k_mio_c, k_suo_c = "rendimento_medio", "ciclo_rendimento_medio"
    metro = "solo sottostante, stesso ciclo annuale" if usa_ciclo else "Buy & Hold"

    righe = []
    for chiave, res in risultato.get("varianti", {}).items():
        mt = res.get("metrics") or {}
        if mt.get(k_dd) is None and mt.get(k_rend) is None:
            continue
        righe.append((res["label"], mt.get(k_dd), mt.get(k_rend),
                      COLORE_VARIANTE.get(chiave, PALETTE["text"]), mt))
    if not righe:
        return _vuoto("Metriche non disponibili")

    etichette = [r[0] for r in righe]
    colori = [r[3] for r in righe]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=(f"Drawdown in meno del {metro}",
                                        f"Rendimento in piu del {metro}"))

    fig.add_trace(go.Bar(
        y=etichette, x=[r[1] for r in righe], orientation="h", showlegend=False,
        marker=dict(color=colori, line=dict(width=0)),
        text=[f"{r[1]:+.1%}" if r[1] is not None else "n.d." for r in righe],
        textposition="outside", textfont=dict(family=FONT_MONO, size=12),
        customdata=np.c_[[r[4].get(k_mio_dd) or np.nan for r in righe],
                         [r[4].get(k_suo_dd) or np.nan for r in righe]],
        hovertemplate=("strategia %{customdata[0]:.1%} contro B&H %{customdata[1]:.1%}"
                       "<extra></extra>")), row=1, col=1)

    fig.add_trace(go.Bar(
        y=etichette, x=[r[2] for r in righe], orientation="h", showlegend=False,
        marker=dict(color=colori, line=dict(width=0)),
        text=[f"{r[2]:+.1%}" if r[2] is not None else "n.d." for r in righe],
        textposition="outside", textfont=dict(family=FONT_MONO, size=12),
        customdata=np.c_[[r[4].get(k_mio_c) or np.nan for r in righe],
                         [r[4].get(k_suo_c) or np.nan for r in righe]],
        hovertemplate=("strategia %{customdata[0]:.1%} contro B&H %{customdata[1]:.1%}"
                       "<extra></extra>")), row=1, col=2)

    for col in (1, 2):
        fig.add_vline(x=0, line=dict(color=PALETTE["axis"], width=1.3), row=1, col=col)
        fig.update_xaxes(tickformat="+.0%", showgrid=True, zeroline=False, row=1, col=col)
        fig.update_yaxes(showgrid=False, autorange="reversed", row=1, col=col)
    fig.update_layout(hovermode="closest", bargap=0.42)
    return _layout(fig, f"Verdetto contro il {metro}",
                   "Il metro e' il solo sottostante comprato e liquidato con lo stesso ciclo "
                   "annuale della strategia: cambia solo la presenza delle opzioni e dei "
                   "Buy-The-Dip. A sinistra il valore positivo significa meno drawdown; a "
                   "destra piu rendimento." if usa_ciclo else
                   "Confronto a parita di versamenti. A sinistra il valore positivo significa "
                   "meno drawdown della strategia; a destra significa piu rendimento.",
                   altezza=340, legenda=False)


# ============================================================================
# 5. Acquisti Buy-The-Dip
# ============================================================================
def fig_btd(risultato: Dict[str, Any], chiave: str = "premi_cash") -> go.Figure:
    res = risultato.get("varianti", {}).get(chiave)
    if not res or res["monthly"].empty:
        return _vuoto("Nessun dato disponibile")
    df = res["monthly"]
    acquisti = df[df["btd_importo"] > 0]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not acquisti.empty:
        fig.add_trace(go.Bar(
            x=acquisti.index, y=acquisti["btd_importo"], name="Acquisto BTD",
            marker=dict(color=PALETTE["btd"], line=dict(width=0)), opacity=0.9,
            customdata=np.c_[acquisti["btd_prezzo"].values, acquisti["dd_weekly"].values],
            hovertemplate=("Acquisto BTD: <b>$%{y:,.0f}</b><br>prezzo %{customdata[0]:,.2f}"
                           "<br>DD settimanale %{customdata[1]:.1%}<extra></extra>"),
        ), secondary_y=False)

    bloccati = df[df["btd_bloccato"] & df["segnale_btd"]]
    if not bloccati.empty:
        fig.add_trace(go.Scatter(
            x=bloccati.index, y=np.zeros(len(bloccati)), name="Segnale bloccato dal filtro",
            mode="markers", marker=dict(color=PALETTE["intrinseco"], size=8, symbol="x"),
            hovertemplate="Segnale bloccato dal filtro sul drawdown<extra></extra>",
        ), secondary_y=False)

    # Segnali arrivati a tetto annuo gia' esaurito: sono i piu' insidiosi,
    # perche' i cali profondi tendono ad arrivare tardi nell'anno, quando il
    # budget e' stato speso su quelli superficiali.
    if "btd_saltato_dal_tetto" in df.columns:
        saltati = df[df["btd_saltato_dal_tetto"].astype(bool)]
        if not saltati.empty:
            cali = df["rendimento_mese"].shift(1).reindex(saltati.index)
            fig.add_trace(go.Scatter(
                x=saltati.index, y=np.zeros(len(saltati)),
                name="Segnale saltato: tetto annuo esaurito", mode="markers",
                marker=dict(color=PALETTE["strike"], size=11, symbol="triangle-up",
                            line=dict(color=PALETTE["bg"], width=1)),
                customdata=np.c_[cali.values],
                hovertemplate=("Tetto annuo gia' esaurito<br>"
                               "il mese prima aveva fatto %{customdata[0]:.1%}"
                               "<extra></extra>"),
            ), secondary_y=False)

        tagliati = df[df["btd_tagliato_dal_tetto"] > 1e-9]
        if not tagliati.empty:
            fig.add_trace(go.Bar(
                x=tagliati.index, y=tagliati["btd_tagliato_dal_tetto"],
                name="Quota tagliata dal tetto",
                marker=dict(color=PALETTE["strike"], line=dict(width=0)), opacity=0.35,
                hovertemplate="Tagliato dal tetto: <b>$%{y:,.0f}</b><extra></extra>",
            ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], name="Prezzo del sottostante",
        line=dict(color=PALETTE["prezzo"], width=1.5),
        hovertemplate="Prezzo: <b>%{y:,.2f}</b><extra></extra>",
    ), secondary_y=True)

    _bande_anni(fig, df.index)
    fig.update_yaxes(tickprefix="$", title_text="Importo investito", rangemode="tozero",
                     secondary_y=False)
    fig.update_yaxes(title_text="Prezzo", showgrid=False, secondary_y=True)
    _asse_tempo(fig)
    tot = float(df["btd_importo"].sum())
    n = int((df["btd_importo"] > 0).sum())
    sotto = f"{n} acquisti per ${tot:,.0f} complessivi."
    if "btd_tagliato_dal_tetto" in df.columns:
        tagliato = float(df["btd_tagliato_dal_tetto"].sum())
        saltati = int(df["btd_saltato_dal_tetto"].astype(bool).sum())
        if tagliato > 1e-9:
            sotto += (f" Il tetto annuo ha tolto ${tagliato:,.0f} agli acquisti e ne ha "
                      f"saltati {saltati} del tutto: quando morde, alzare il boost non fa "
                      f"comprare di piu', fa solo esaurire prima il budget sui cali "
                      f"superficiali.")
    fig.update_layout(barmode="overlay")
    return _layout(fig, "Acquisti Buy-The-Dip", sotto, altezza=460)


# ============================================================================
# 6. Drawdown settimanale dell'asset vs limite
# ============================================================================
def fig_dd_settimanale(risultato: Dict[str, Any]) -> go.Figure:
    dd = risultato.get("mercato", {}).get("dd_weekly")
    if not isinstance(dd, pd.Series) or dd.empty:
        return _vuoto("Dati settimanali non disponibili per questo ticker")
    limite = float(risultato["config"].get("btd_dd_weekly_limit", -0.90))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name="Drawdown settimanale",
        line=dict(color=PALETTE["drawdown"], width=1.4),
        fill="tozeroy", fillcolor="rgba(244,63,94,0.18)", hovertemplate=HT_PCT))
    fig.add_hline(y=limite, line=dict(color=PALETTE["strike"], width=1.6, dash="dash"),
                  annotation_text=f"Soglia di blocco {limite:.0%}",
                  annotation_position="bottom left",
                  annotation_font=dict(color=PALETTE["strike"], size=11))
    sotto = dd[dd < limite]
    if not sotto.empty:
        fig.add_trace(go.Scatter(
            x=sotto.index, y=sotto.values, name="BTD bloccato", mode="markers",
            marker=dict(color=PALETTE["strike"], size=5), hovertemplate=HT_PCT))
    fig.update_yaxes(tickformat=".0%", title_text="Drawdown dell'asset")
    _asse_tempo(fig)
    return _layout(fig, "Drawdown settimanale dell'asset e filtro sul BTD",
                   "Sotto la soglia il Buy-The-Dip viene sospeso. La decisione usa il dato "
                   "disponibile a fine mese precedente, mai quello del mese in corso.",
                   altezza=400)


# ============================================================================
# 7. Rendimenti annuali operativi
# ============================================================================
def fig_rendimenti_annuali(risultato: Dict[str, Any]) -> go.Figure:
    varianti = risultato.get("varianti", {})
    if not varianti:
        return _vuoto("Nessun dato disponibile")

    fig = go.Figure()
    rif_y = None
    tutte = dict(varianti)
    bm = _benchmark(risultato)
    if bm is not None:
        tutte["benchmark"] = bm
    for chiave, res in tutte.items():
        y = res.get("yearly")
        if y is None or y.empty:
            continue
        rif_y = y
        fig.add_trace(go.Bar(
            x=y.index.astype(str), y=y["twr_anno"], name=res["label"],
            marker=dict(color=COLORE_VARIANTE.get(chiave, PALETTE["text"]),
                        line=dict(width=2 if chiave == "benchmark" else 0,
                                  color=COLORE_BENCHMARK)),
            opacity=0.55 if chiave == "benchmark" else 1.0,
            hovertemplate="%{fullData.name} %{x}: <b>%{y:.1%}</b><extra></extra>"))

    if rif_y is not None and "rendimento_sottostante" in rif_y.columns:
        fig.add_trace(go.Scatter(
            x=rif_y.index.astype(str), y=rif_y["rendimento_sottostante"],
            name="Sottostante (senza ciclo)", mode="markers",
            marker=dict(color=PALETTE["prezzo"], size=10, symbol="diamond",
                        line=dict(color=PALETTE["bg"], width=1.5)),
            hovertemplate="Sottostante %{x}: <b>%{y:.1%}</b><extra></extra>"))

    fig.add_hline(y=0, line=dict(color=PALETTE["axis"], width=1.2))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06, hovermode="x unified")
    fig.update_yaxes(tickformat=".0%", title_text="Rendimento dell'anno")
    fig.update_xaxes(showgrid=False, title_text="")
    return _layout(fig, "Rendimenti annuali operativi",
                   "Ogni anno e' un ciclo chiuso: si parte dallo stesso capitale fisso e si "
                   "liquida a dicembre. Rendimenti time-weighted, ripuliti dai versamenti.",
                   altezza=440)


# ============================================================================
# 8. Heatmap dei rendimenti mensili
# ============================================================================
def fig_heatmap_mensile(risultato: Dict[str, Any], chiave: str = "premi_reinvest") -> go.Figure:
    res = risultato.get("varianti", {}).get(chiave)
    if not res or res["monthly"].empty:
        return _vuoto("Nessun dato disponibile")
    df = res["monthly"]
    piv = (df.assign(m=df.index.month, a=df.index.year)
             .pivot_table(index="a", columns="m", values="twr_mese", aggfunc="last")
             .reindex(columns=range(1, 13)))
    if piv.empty:
        return _vuoto("Nessun dato disponibile")

    limite = float(np.nanmax(np.abs(piv.values))) if np.isfinite(piv.values).any() else 0.1
    testo = np.where(np.isfinite(piv.values),
                     np.vectorize(lambda v: f"{v:.1%}" if np.isfinite(v) else "")(piv.values), "")

    fig = go.Figure(go.Heatmap(
        z=piv.values, x=MESI_IT, y=piv.index.astype(str),
        colorscale=SCALA_DIVERGENTE, zmid=0.0, zmin=-limite, zmax=limite,
        text=testo, texttemplate="%{text}",
        textfont=dict(size=10, family=FONT_MONO),
        xgap=3, ygap=3,
        colorbar=dict(tickformat=".0%", outlinewidth=0, thickness=12,
                      tickfont=dict(color=PALETTE["text_muted"], size=10)),
        hovertemplate="%{y} %{x}: <b>%{z:.2%}</b><extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="")
    fig.update_xaxes(side="top", showgrid=False, title_text="")
    return _layout(fig, f"Rendimenti mensili — {res['label']}",
                   "Rendimento time-weighted mese per mese.",
                   altezza=max(300, 60 + 34 * len(piv)), legenda=False)


# ============================================================================
# 9. Distribuzione dei rendimenti mensili
# ============================================================================
def fig_distribuzione(risultato: Dict[str, Any]) -> go.Figure:
    serie = _serie_varianti(risultato, "twr_mese")
    if not serie:
        return _vuoto("Nessun dato disponibile")
    prezzi = risultato.get("mercato", {}).get("prezzi")

    fig = go.Figure()
    bm = _benchmark(risultato)
    if bm is not None:
        rb = bm["monthly"]["twr_mese"].dropna()
        if not rb.empty:
            fig.add_trace(go.Violin(
                y=rb.values, name=bm["label"], box_visible=True, meanline_visible=True,
                line=dict(color=COLORE_BENCHMARK, width=1.6),
                fillcolor=_rgba(COLORE_BENCHMARK, 0.22), opacity=0.9, points=False,
                hovertemplate=f"{bm['label']}<br>%{{y:.2%}}<extra></extra>"))
    elif isinstance(prezzi, pd.DataFrame) and "rendimento_mese" in prezzi.columns:
        r = prezzi["rendimento_mese"].dropna()
        if not r.empty:
            fig.add_trace(go.Violin(
                y=r.values, name="Sottostante", box_visible=True, meanline_visible=True,
                line=dict(color=PALETTE["prezzo"], width=1.4),
                fillcolor="rgba(203,213,225,0.16)", opacity=0.85, points=False,
                hovertemplate="Sottostante<br>%{y:.2%}<extra></extra>"))

    for chiave, s in serie.items():
        s = s.dropna()
        if s.empty:
            continue
        c = COLORE_VARIANTE.get(chiave, PALETTE["text"])
        fig.add_trace(go.Violin(
            y=s.values, name=risultato["varianti"][chiave]["label"],
            box_visible=True, meanline_visible=True,
            line=dict(color=c, width=1.4), fillcolor=c, opacity=0.30, points=False,
            hovertemplate="%{fullData.name}<br>%{y:.2%}<extra></extra>"))

    fig.add_hline(y=0, line=dict(color=PALETTE["axis"], width=1.1))
    fig.update_layout(hovermode="closest", violingap=0.3)
    fig.update_yaxes(tickformat=".0%", title_text="Rendimento mensile")
    fig.update_xaxes(showgrid=False)
    return _layout(fig, "Distribuzione dei rendimenti mensili",
                   "La covered call taglia la coda destra e lascia quasi intatta la sinistra: "
                   "e' il costo del cap.", altezza=440)


# ============================================================================
# 10. Rolling return
# ============================================================================
def fig_rolling(risultato: Dict[str, Any], finestra: int = 12) -> go.Figure:
    serie = _serie_varianti(risultato, "twr_mese")
    if not serie:
        return _vuoto("Nessun dato disponibile")
    fig = go.Figure()
    for chiave, s in serie.items():
        rr = (1.0 + s).rolling(finestra).apply(np.prod, raw=True) - 1.0
        rr = rr.dropna()
        if rr.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rr.index, y=rr.values, name=risultato["varianti"][chiave]["label"],
            line=dict(color=COLORE_VARIANTE.get(chiave, PALETTE["text"]), width=2.0),
            hovertemplate=HT_PCT))
    fig.add_hline(y=0, line=dict(color=PALETTE["axis"], width=1.1))
    fig.update_yaxes(tickformat=".0%", title_text=f"Rendimento a {finestra} mesi")
    _asse_tempo(fig)
    return _layout(fig, f"Rendimento rolling a {finestra} mesi",
                   "Rendimento composto della finestra mobile, time-weighted.", altezza=400)


# ============================================================================
# 11. Rischio / rendimento
# ============================================================================
def fig_rischio_rendimento(risultato: Dict[str, Any]) -> go.Figure:
    punti = []
    tutte = dict(risultato.get("varianti", {}))
    bm = _benchmark(risultato)
    if bm is not None:
        tutte["benchmark"] = bm
    for chiave, res in tutte.items():
        mt = res.get("metrics") or {}
        if mt.get("rendimento_medio") is not None and mt.get("rendimento_volatilita"):
            punti.append((res["label"], mt["rendimento_volatilita"], mt["rendimento_medio"],
                          mt.get("rendimento_su_rischio"),
                          COLORE_VARIANTE.get(chiave, PALETTE["text"]),
                          chiave == "benchmark"))
    if not punti:
        return _vuoto("Metriche non disponibili")

    fig = go.Figure()
    # isoquante del rapporto rendimento/oscillazione come sfondo
    xmax = max(p[1] for p in punti) * 1.35
    xs = np.linspace(0.001, xmax, 60)
    for sh in (0.25, 0.5, 1.0, 1.5):
        fig.add_trace(go.Scatter(
            x=xs, y=sh * xs, mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(148,163,184,0.16)", width=1, dash="dot")))
        fig.add_annotation(x=xmax * 0.98, y=sh * xmax * 0.98, text=f"rapporto {sh}",
                           showarrow=False, xanchor="right",
                           font=dict(color=PALETTE["text_muted"], size=10))

    for nome, vol, rendimento, rapporto, colore, e_bench in punti:
        fig.add_trace(go.Scatter(
            x=[vol], y=[rendimento], name=nome, mode="markers+text",
            text=[nome], textposition="top center",
            textfont=dict(color=PALETTE["text_muted"], size=11),
            marker=dict(color=colore, size=20 if e_bench else 18,
                        symbol="diamond" if e_bench else "circle",
                        line=dict(color=PALETTE["bg"], width=2)),
            hovertemplate=(f"<b>{nome}</b><br>Oscillazione: %{{x:.1%}}<br>"
                           f"Rendimento: %{{y:.1%}}<br>"
                           f"rapporto {rapporto:.2f}<extra></extra>")))

    fig.add_hline(y=0, line=dict(color=PALETTE["axis"], width=1.1))
    fig.update_xaxes(tickformat=".0%", title_text="Oscillazione dei rendimenti annuali",
                     showgrid=True, range=[0, xmax])
    fig.update_yaxes(tickformat=".0%", title_text="Rendimento medio annuo")
    fig.update_layout(hovermode="closest")
    return _layout(fig, "Rischio e rendimento",
                   "Rendimento medio annuo sul capitale investito contro quanto oscilla da "
                   "un anno all'altro. Il rombo e' il solo sottostante con lo stesso ciclo "
                   "annuale: le varianti sopra e a sinistra di quel punto stanno facendo "
                   "meglio.", altezza=440, legenda=False)


# ============================================================================
# 12. Durata dei drawdown
# ============================================================================
def fig_durata_drawdown(risultato: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    trovato = False
    tutte = dict(risultato.get("varianti", {}))
    bm = _benchmark(risultato)
    if bm is not None:
        tutte["benchmark"] = bm
    for chiave, res in tutte.items():
        df = res.get("monthly")
        if df is None or df.empty:
            continue
        durate = drawdown_durations(df["dd_twr_pct"])
        if not durate:
            continue
        trovato = True
        e_bench = chiave == "benchmark"
        fig.add_trace(go.Histogram(
            x=durate, name=res["label"], xbins=dict(start=0.5, size=1),
            marker=dict(color=COLORE_VARIANTE.get(chiave, PALETTE["text"]),
                        line=dict(width=2 if e_bench else 0, color=COLORE_BENCHMARK)),
            opacity=0.45 if e_bench else 0.72,
            hovertemplate="%{fullData.name}<br>%{x} mesi: %{y} episodi<extra></extra>"))
    if not trovato:
        return _vuoto("Nessun episodio di drawdown registrato")
    fig.update_layout(barmode="overlay", bargap=0.12)
    fig.update_xaxes(title_text="Durata dell'episodio (mesi)", dtick=1, showgrid=False)
    fig.update_yaxes(title_text="Numero di episodi")
    return _layout(fig, "Durata degli episodi di drawdown",
                   "Quanto tempo passa sotto il massimo precedente, confrontato con il solo "
                   "sottostante che segue lo stesso ciclo annuale.", altezza=400)


# ============================================================================
# 13. Premio stimato e volatilita'
# ============================================================================
def fig_premio_stimato(risultato: Dict[str, Any], chiave: str = "premi_cash") -> go.Figure:
    res = risultato.get("varianti", {}).get(chiave)
    if not res or res["monthly"].empty:
        return _vuoto("Nessun dato disponibile")
    df = res["monthly"]
    if df["premio_pct"].fillna(0).eq(0).all():
        return _vuoto("Questa variante non vende opzioni")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.5, 0.5], specs=[[{"secondary_y": True}], [{}]])

    fig.add_trace(go.Scatter(
        x=df.index, y=df["sigma_stimata"], name="Volatilita realizzata",
        line=dict(color=PALETTE["vol"], width=1.7), hovertemplate=HT_PCT),
        row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["sigma_implicita"], name="Volatilita usata per il pricing",
        line=dict(color=PALETTE["strike"], width=1.7, dash="dash"), hovertemplate=HT_PCT),
        row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["premio_pct"], name="Premio (% dello spot)",
        line=dict(color=PALETTE["premio"], width=2.3),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.14)", hovertemplate=HT_PCT),
        row=1, col=1, secondary_y=True)

    fig.add_trace(go.Bar(
        x=df.index, y=df["premio"], name="Premio incassato",
        marker=dict(color=PALETTE["premio"], line=dict(width=0)), opacity=0.85,
        hovertemplate="Premio: <b>$%{y:,.0f}</b><extra></extra>"), row=2, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=-df["intrinseco_pagato"], name="Intrinseco pagato",
        marker=dict(color=PALETTE["intrinseco"], line=dict(width=0)), opacity=0.85,
        hovertemplate="Intrinseco: <b>-$%{y:,.0f}</b><extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["netto_opzione"].cumsum(), name="Netto opzioni cumulato",
        line=dict(color=PALETTE["text"], width=2.0),
        hovertemplate="Netto cumulato: <b>$%{y:,.0f}</b><extra></extra>"), row=2, col=1)

    fig.update_layout(barmode="relative")
    fig.update_yaxes(tickformat=".0%", title_text="Volatilita", row=1, col=1, secondary_y=False)
    fig.update_yaxes(tickformat=".1%", title_text="Premio", row=1, col=1,
                     secondary_y=True, showgrid=False, rangemode="tozero")
    fig.update_yaxes(tickprefix="$", title_text="Valuta", row=2, col=1)
    _asse_tempo(fig, selettore=False, riga=1)
    _asse_tempo(fig, selettore=False, riga=2)
    medio = df.loc[df["premio_pct"] > 0, "premio_pct"].mean()
    return _layout(fig, "Premio stimato e volatilita",
                   f"Premio medio {medio:.2%} dello spot al mese. In basso l'incasso reale contro "
                   f"il costo del cap: la linea chiara e' il risultato netto cumulato.",
                   altezza=580)


# ============================================================================
# 14. Composizione del risultato annuale
# ============================================================================
def fig_composizione_annuale(risultato: Dict[str, Any], chiave: str = "premi_cash") -> go.Figure:
    res = risultato.get("varianti", {}).get(chiave)
    if not res or res.get("yearly") is None or res["yearly"].empty:
        return _vuoto("Nessun dato disponibile")
    y = res["yearly"]
    anni = y.index.astype(str)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=anni, y=y["premi_incassati"], name="Premi incassati",
                         marker=dict(color=PALETTE["premio"], line=dict(width=0)),
                         hovertemplate="Premi: <b>$%{y:,.0f}</b><extra></extra>"))
    fig.add_trace(go.Bar(x=anni, y=-y["intrinseco_pagato"], name="Intrinseco pagato",
                         marker=dict(color=PALETTE["intrinseco"], line=dict(width=0)),
                         hovertemplate="Intrinseco: <b>$%{y:,.0f}</b><extra></extra>"))
    fig.add_trace(go.Bar(x=anni, y=y["btd_investito"], name="Capitale BTD investito",
                         marker=dict(color=PALETTE["btd"], line=dict(width=0)), opacity=0.55,
                         hovertemplate="BTD: <b>$%{y:,.0f}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=anni, y=y["risultato_anno"], name="Risultato dell'anno",
                             mode="markers+lines",
                             line=dict(color=PALETTE["text"], width=2.0),
                             marker=dict(size=10, color=PALETTE["text"],
                                         line=dict(color=PALETTE["bg"], width=2)),
                             hovertemplate="Risultato: <b>$%{y:,.0f}</b><extra></extra>"))
    fig.add_hline(y=0, line=dict(color=PALETTE["axis"], width=1.2))
    fig.update_layout(barmode="relative", bargap=0.3, hovermode="x unified")
    fig.update_yaxes(tickprefix="$", title_text="")
    fig.update_xaxes(showgrid=False, title_text="")
    return _layout(fig, f"Composizione del risultato annuale — {res['label']}",
                   "Da cosa arriva il risultato di ogni ciclo annuale.", altezza=440)


# ============================================================================
# 15. Prezzo, strike e assegnazioni
# ============================================================================
def fig_prezzo_strike(risultato: Dict[str, Any], chiave: str = "premi_cash") -> go.Figure:
    res = risultato.get("varianti", {}).get(chiave)
    if not res or res["monthly"].empty:
        return _vuoto("Nessun dato disponibile")
    df = res["monthly"]
    if df["strike"].isna().all():
        return _vuoto("Questa variante non vende opzioni")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Chiusura del mese",
                             line=dict(color=PALETTE["prezzo"], width=1.9),
                             hovertemplate="Close: <b>%{y:,.2f}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=df.index, y=df["strike"], name="Strike della call venduta",
                             line=dict(color=PALETTE["strike"], width=1.5, dash="dot"),
                             hovertemplate="Strike: <b>%{y:,.2f}</b><extra></extra>"))
    itm = df[df["intrinseco_pagato"] > 0]
    if not itm.empty:
        fig.add_trace(go.Scatter(
            x=itm.index, y=itm["close"], name="Call in-the-money", mode="markers",
            marker=dict(color=PALETTE["intrinseco"], size=8, symbol="triangle-down",
                        line=dict(color=PALETTE["bg"], width=1)),
            customdata=np.c_[itm["intrinseco_pagato"].values],
            hovertemplate="Call ITM<br>intrinseco pagato $%{customdata[0]:,.0f}<extra></extra>"))
    _bande_anni(fig, df.index)
    fig.update_yaxes(title_text="Prezzo")
    _asse_tempo(fig)
    n_itm, n = len(itm), len(df)
    return _layout(fig, "Prezzo del sottostante e strike venduto",
                   f"Call finita in-the-money in {n_itm} mesi su {n} ({n_itm / n:.0%}). "
                   f"A delta 0.50 lo strike sta appena sopra il prezzo di apertura.",
                   altezza=440)


# ============================================================================
# 16. Calibrazione del premio
# ============================================================================
def fig_calibrazione(cal: Dict[str, Any]) -> go.Figure:
    df = cal.get("dettaglio")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return _vuoto("Carica un file di prezzi reali per calibrare")

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.09,
                        subplot_titles=("Andamento nel tempo", "Stimato contro reale"))

    fig.add_trace(go.Scatter(
        x=df["data"], y=df["premio_pct_reale"], name="Premio reale",
        mode="markers", marker=dict(color=PALETTE["prezzo"], size=7),
        hovertemplate="Reale: <b>%{y:.2%}</b><extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["data"], y=df["premio_pct_stimato"], name="Premio stimato",
        mode="markers", marker=dict(color=PALETTE["premio"], size=7, symbol="diamond"),
        hovertemplate="Stimato: <b>%{y:.2%}</b><extra></extra>"), row=1, col=1)

    lim = float(max(df["premio_pct_reale"].max(), df["premio_pct_stimato"].max())) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim], mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color=PALETTE["axis"], width=1.2, dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=df["premio_pct_reale"], y=df["premio_pct_stimato"], name="Osservazioni",
        mode="markers", marker=dict(color=PALETTE["vol"], size=8, opacity=0.75),
        showlegend=False,
        hovertemplate="reale %{x:.2%}<br>stimato %{y:.2%}<extra></extra>"), row=1, col=2)

    fig.update_yaxes(tickformat=".1%", title_text="Premio / spot", row=1, col=1)
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(tickformat=".1%", title_text="Reale", row=1, col=2, range=[0, lim])
    fig.update_yaxes(tickformat=".1%", title_text="Stimato", row=1, col=2, range=[0, lim])
    fig.update_layout(hovermode="closest")

    m = cal.get("metriche", {})
    sub = (f"VRP calibrato {m.get('vrp_calibrato', float('nan')):.3f} · "
           f"MAE {m.get('mae', float('nan')) * 100:.2f} punti di spot · "
           f"bias {m.get('bias', float('nan')) * 100:+.2f} · "
           f"R² {m.get('r2', float('nan')):.3f} · {m.get('n', 0)} osservazioni")
    return _layout(fig, "Calibrazione del premio sui prezzi reali", sub, altezza=440)
