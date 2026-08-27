"""Premi reali misurati su call ATM mensili, usati come metro di controllo.

Ricavati da 1.666 vendite effettive di call at-the-money con scadenza mensile
esportate da OptionLAB. Per ogni operazione il premio e' stato rapportato al
prezzo del sottostante alla data di apertura, e la volatilita' implicita e'
stata ottenuta invertendo Black-Scholes sul premio.

Servono a rispondere a una domanda pratica: il premio che il modello sta
stimando e' nell'ordine di grandezza giusto per quel sottostante? Non sono
parametri del modello e non influenzano il backtest.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# premio e volatilita' in frazione (0.0164 = 1,64% del prezzo del sottostante)
PREMI_REALI: Dict[str, Dict[str, Any]] = {
    "SPX":  dict(n=312, dal="2000-01", al="2025-12", dte=28, premio_mediano=0.01640,
                 premio_p05=0.00951, premio_p95=0.03197, iv_mediana=0.1387,
                 descrizione="Indice S&P 500"),
    "SPY":  dict(n=252, dal="2005-01", al="2025-12", dte=28, premio_mediano=0.01615,
                 premio_p05=0.00980, premio_p95=0.03210, iv_mediana=0.1365,
                 descrizione="ETF sull'S&P 500"),
    "AAPL": dict(n=305, dal="2000-05", al="2025-12", dte=28, premio_mediano=0.03467,
                 premio_p05=0.01920, premio_p95=0.07072, iv_mediana=0.3074,
                 descrizione="Apple"),
    "PG":   dict(n=306, dal="2000-05", al="2025-12", dte=28, premio_mediano=0.01893,
                 premio_p05=0.00753, premio_p95=0.04487, iv_mediana=0.1534,
                 descrizione="Procter & Gamble"),
    "WMT":  dict(n=307, dal="2000-05", al="2025-12", dte=28, premio_mediano=0.02211,
                 premio_p05=0.00923, premio_p95=0.05420, iv_mediana=0.1798,
                 descrizione="Walmart"),
    "TSLA": dict(n=184, dal="2010-07", al="2025-12", dte=28, premio_mediano=0.05446,
                 premio_p05=0.03598, premio_p95=0.07728, iv_mediana=0.5051,
                 descrizione="Tesla"),
}

# Alias verso i ticker EODHD piu' comuni per gli stessi sottostanti.
ALIAS_TICKER: Dict[str, str] = {
    "GSPC": "SPX", "SPX": "SPX", "SP500": "SPX", "^GSPC": "SPX",
    "SPY": "SPY", "VOO": "SPY", "IVV": "SPY",
    "AAPL": "AAPL", "PG": "PG", "WMT": "WMT", "TSLA": "TSLA",
}


def radice(ticker: str) -> str:
    """'AAPL.US' -> 'AAPL', '$SPX' -> 'SPX', 'BTC-USD.CC' -> 'BTC-USD'."""
    t = str(ticker or "").strip().upper().lstrip("$^")
    return t.split(".")[0]


def riferimento(ticker: str) -> Optional[Dict[str, Any]]:
    """Riferimento sui premi reali per questo ticker, se disponibile."""
    chiave = ALIAS_TICKER.get(radice(ticker))
    if not chiave:
        return None
    return {"simbolo": chiave, **PREMI_REALI[chiave]}


def giudizio(premio_stimato: Optional[float], rif: Optional[Dict[str, Any]]) -> Optional[str]:
    """Confronta il premio medio stimato con l'intervallo osservato sui prezzi reali."""
    if premio_stimato is None or not rif:
        return None
    mediano = rif["premio_mediano"]
    scarto = premio_stimato / mediano - 1.0
    if rif["premio_p05"] <= premio_stimato <= rif["premio_p95"]:
        posizione = "dentro l'intervallo osservato"
    elif premio_stimato < rif["premio_p05"]:
        posizione = "sotto l'intervallo osservato"
    else:
        posizione = "sopra l'intervallo osservato"
    return (f"Premio medio stimato {premio_stimato:.2%} contro un mediano reale di "
            f"{mediano:.2%} su {rif['n']} vendite effettive di call ATM mensili "
            f"({rif['dal']} - {rif['al']}): {scarto:+.0%}, {posizione} "
            f"({rif['premio_p05']:.2%} - {rif['premio_p95']:.2%}).")
