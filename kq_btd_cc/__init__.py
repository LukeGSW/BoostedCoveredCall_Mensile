"""kq_btd_cc — Boosted Covered Call mensile con Buy-The-Dip potenziato.

Moduli:
  data_api    download EODHD (mensile/settimanale/giornaliero) con cache
  vol         stimatori di volatilita' realizzata da OHLC
  pricing     Black-Scholes senza scipy e stima del premio a delta 0.50
  engine      motore di backtest (contabilita' a flussi, cap della call reale)
  metrics     metriche di performance e rischio time-weighted
  calibration calibrazione del premio stimato su prezzi di opzioni reali
  charts      grafici Plotly a tema scuro
  export      pacchetto JSON completo del backtest
  style       palette, template Plotly e CSS
  utils       formattazione e helper sui dati
"""

from .style import STYLE_CONFIG, PALETTE, TEMPLATE_NAME, CSS
from .engine import BacktestConfig, run_backtest, VARIANTS
from .pricing import PremiumModel
from .metrics import compute_metrics, metrics_table
from .export import build_export, export_json_bytes

__all__ = [
    "STYLE_CONFIG", "PALETTE", "TEMPLATE_NAME", "CSS",
    "BacktestConfig", "run_backtest", "VARIANTS",
    "PremiumModel", "compute_metrics", "metrics_table",
    "build_export", "export_json_bytes",
]

__version__ = "2.0.0"
