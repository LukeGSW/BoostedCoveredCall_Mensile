"""kq_btd_cc
Pacchetto per la strategia Boosted Covered Call (mensile) con BTD.

Contiene:
- data_api: download e preparazione dati EODHD (OHLC aggiustati)
- utils: funzioni di utilità per formattazione e plotting
- style: configurazioni estetiche dei grafici
- core: logica di simulazione e generazione figure
"""

from .style import STYLE_CONFIG
from .core import esegui_analisi_completa

__all__ = ["STYLE_CONFIG", "esegui_analisi_completa"]

__version__ = "1.0.0"
