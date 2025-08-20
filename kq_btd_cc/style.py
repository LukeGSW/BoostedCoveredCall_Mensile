
"""Stile e palette per i grafici della dashboard.
Niente dipendenza da seaborn per compatibilità con requirements.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

STYLE_CONFIG = {
    "figure_figsize": (16, 8),
    "colors": {
        "equity_no_prem": "#E67E22",
        "equity_prem_accum": "#F1C40F",
        "equity_prem_reinvest": "#1ABC9C",
        "investment": "#3498DB",
        "reinvest": "#5DADE2",
        "drawdown_asset": "#8E44AD",
        "drawdown_portfolio_usd": "#E74C3C",
        "buy_hold": "#9B59B6",
    },
    "line_width": {"standard": 1.8, "thin": 1.2, "thick": 2.2},
}

# Applica uno stile di base
plt.style.use("seaborn-v0_8-whitegrid")
