"""Tema grafico della dashboard: palette, template Plotly e CSS.

Un unico posto in cui vivono i colori, cosi' che ogni grafico e ogni KPI
parlino la stessa lingua visiva. La palette e' pensata per fondo scuro con
contrasto sufficiente e serie distinguibili anche in scala di grigi.
"""
from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go
import plotly.io as pio

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
PALETTE: Dict[str, str] = {
    # superfici
    "bg":          "#0B1220",
    "surface":     "#111827",
    "surface_alt": "#161F32",
    "grid":        "rgba(148,163,184,0.13)",
    "axis":        "rgba(148,163,184,0.35)",
    "text":        "#E5E7EB",
    "text_muted":  "#94A3B8",

    # serie principali
    "no_premi":       "#F59E0B",   # ambra
    "premi_cash":     "#38BDF8",   # azzurro
    "premi_reinvest": "#2DD4BF",   # verde acqua
    "bh_semplice":    "#A78BFA",   # viola
    "bh_flussi":      "#94A3B8",   # grigio-blu

    # elementi
    "versamenti":  "#64748B",
    "btd":         "#6366F1",
    "premio":      "#22C55E",
    "intrinseco":  "#EF4444",
    "drawdown":    "#F43F5E",
    "vol":         "#0EA5E9",
    "strike":      "#FBBF24",
    "prezzo":      "#CBD5E1",
    "positivo":    "#22C55E",
    "negativo":    "#F43F5E",
    "neutro":      "#64748B",
}

# Il benchmark a parita' di mandato: colore distinto dalle tre varianti, sempre
# tratteggiato, cosi' si riconosce a colpo d'occhio in ogni grafico.
COLORE_BENCHMARK = PALETTE["bh_semplice"]

# colore per variante del motore
COLORE_VARIANTE: Dict[str, str] = {
    "no_premi": PALETTE["no_premi"],
    "premi_cash": PALETTE["premi_cash"],
    "premi_reinvest": PALETTE["premi_reinvest"],
    "benchmark": COLORE_BENCHMARK,
}

# scala divergente per le heatmap dei rendimenti (rosso -> neutro -> verde)
SCALA_DIVERGENTE = [
    [0.00, "#7F1D1D"], [0.25, "#DC2626"], [0.45, "#4B5563"],
    [0.50, "#374151"], [0.55, "#4B5563"], [0.75, "#16A34A"], [1.00, "#14532D"],
]

FONT = "Inter, Segoe UI, system-ui, -apple-system, sans-serif"
FONT_MONO = "JetBrains Mono, Consolas, ui-monospace, monospace"


# ----------------------------------------------------------------------------
# Template Plotly
# ----------------------------------------------------------------------------
def _build_template() -> go.layout.Template:
    asse = dict(
        gridcolor=PALETTE["grid"],
        zerolinecolor=PALETTE["axis"],
        linecolor=PALETTE["axis"],
        tickcolor=PALETTE["axis"],
        tickfont=dict(color=PALETTE["text_muted"], size=11),
        title=dict(font=dict(color=PALETTE["text_muted"], size=12)),
        showline=True,
        automargin=True,
    )
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT, color=PALETTE["text"], size=13),
            title=dict(font=dict(size=17, color=PALETTE["text"]), x=0.0, xanchor="left",
                       pad=dict(b=14)),
            margin=dict(l=10, r=10, t=64, b=10),
            xaxis={**asse, "showgrid": False},
            yaxis=asse,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0,
                bgcolor="rgba(0,0,0,0)", borderwidth=0,
                font=dict(size=12, color=PALETTE["text_muted"]),
            ),
            hoverlabel=dict(
                bgcolor=PALETTE["surface"], bordercolor=PALETTE["axis"],
                font=dict(family=FONT_MONO, size=12, color=PALETTE["text"]),
                align="left",
            ),
            hovermode="x unified",
            colorway=[PALETTE["premi_reinvest"], PALETTE["premi_cash"], PALETTE["no_premi"],
                      PALETTE["bh_semplice"], PALETTE["btd"], PALETTE["premio"]],
            separators=",.",
        )
    )


TEMPLATE_NAME = "kq_dark"
pio.templates[TEMPLATE_NAME] = _build_template()
pio.templates.default = TEMPLATE_NAME


# ----------------------------------------------------------------------------
# CSS della pagina Streamlit
# ----------------------------------------------------------------------------
CSS = f"""
<style>
  .stApp {{ background: {PALETTE['bg']}; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
  h1, h2, h3 {{ font-family: {FONT}; letter-spacing: -0.02em; }}
  h1 {{ font-weight: 650; }}

  /* riga di KPI */
  .kq-kpi-row {{ display: grid; gap: 12px; margin: 4px 0 18px 0;
                 grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); }}
  .kq-kpi {{ background: {PALETTE['surface']};
             border: 1px solid rgba(148,163,184,0.14);
             border-radius: 12px; padding: 14px 16px; }}
  .kq-kpi .lab {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
                  color: {PALETTE['text_muted']}; margin-bottom: 6px; }}
  .kq-kpi .val {{ font-family: {FONT_MONO}; font-size: 22px; font-weight: 600;
                  color: {PALETTE['text']}; line-height: 1.15; }}
  .kq-kpi .sub {{ font-size: 11px; color: {PALETTE['text_muted']}; margin-top: 5px; }}
  .kq-pos {{ color: {PALETTE['positivo']} !important; }}
  .kq-neg {{ color: {PALETTE['negativo']} !important; }}

  /* bandella informativa */
  .kq-note {{ background: {PALETTE['surface_alt']};
              border-left: 3px solid {PALETTE['vol']};
              border-radius: 6px; padding: 10px 14px; margin: 6px 0 16px 0;
              font-size: 13px; color: {PALETTE['text_muted']}; }}

  section[data-testid="stSidebar"] {{ background: {PALETTE['surface']}; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
  div[data-testid="stMetricValue"] {{ font-family: {FONT_MONO}; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{ border-radius: 8px 8px 0 0; padding: 8px 16px; }}
</style>
"""

# Retrocompatibilita' con il vecchio modulo (alcuni script esterni lo importano)
STYLE_CONFIG = {
    "figure_figsize": (16, 8),
    "colors": {
        "equity_no_prem": PALETTE["no_premi"],
        "equity_prem_accum": PALETTE["premi_cash"],
        "equity_prem_reinvest": PALETTE["premi_reinvest"],
        "investment": PALETTE["versamenti"],
        "reinvest": PALETTE["btd"],
        "drawdown_asset": PALETTE["drawdown"],
        "drawdown_portfolio_usd": PALETTE["drawdown"],
        "buy_hold": PALETTE["bh_semplice"],
    },
    "line_width": {"standard": 2.0, "thin": 1.3, "thick": 2.6},
    "palette": PALETTE,
    "template": TEMPLATE_NAME,
}
