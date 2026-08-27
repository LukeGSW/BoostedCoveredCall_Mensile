"""kq_btd_cc — Boosted Covered Call con Buy-The-Dip potenziato, mensile o settimanale.

La cadenza decide solo la lunghezza del passo: mensile (l'originale) o
settimanale, dove tutto quello che si fa a fine mese si fa a fine settimana.
Il ciclo annuale — capitale deciso a gennaio, tutto liquidato a dicembre — resta
identico nelle due versioni. Le decisioni stanno sulla griglia del periodo, ma il
conto e' valorizzato ogni giorno di borsa: e' li' che si vedono i drawdown veri.

Moduli:
  cadenza     mensile o settimanale: periodi per anno e vocabolario dei testi
  giornaliero rivalutazione giorno per giorno del conto (drawdown veri)
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
from .cadenza import CADENZE, adatta, periodi_anno
from .engine import BacktestConfig, run_backtest, VARIANTS
from .pricing import PremiumModel
from .metrics import compute_metrics, metrics_table
from .export import build_export, export_json_bytes

__all__ = [
    "STYLE_CONFIG", "PALETTE", "TEMPLATE_NAME", "CSS",
    "CADENZE", "adatta", "periodi_anno",
    "BacktestConfig", "run_backtest", "VARIANTS",
    "PremiumModel", "compute_metrics", "metrics_table",
    "build_export", "export_json_bytes",
]

__version__ = "2.2.0"
