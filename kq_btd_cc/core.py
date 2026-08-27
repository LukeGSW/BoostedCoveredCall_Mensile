"""Punto di ingresso: dai parametri della sidebar alle figure pronte.

`esegui_analisi_completa` scarica i dati, lancia il motore e costruisce solo i
grafici effettivamente richiesti dalle preferenze. I flag di `plot_prefs`
adesso contano davvero: in passato l'unico letto era quello dei grafici
addizionali e tutti gli altri erano decorativi.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import charts
from .cadenza import adatta, normalizza
from .data_api import ChiaveMancante, DatiNonDisponibili, carica_serie
from .engine import BacktestConfig, run_backtest
from .pricing import PremiumModel

# Mappa dai vecchi nomi in MAIUSCOLO ai campi di BacktestConfig.
ALIAS_LEGACY: Dict[str, str] = {
    "EODHD_TICKER": "ticker",
    "START_DATE": "start_date",
    "END_DATE_OVERRIDE": "end_date",
    "INITIAL_CAPITAL": "capitale_iniziale",
    "ADDITIONAL_CAPITAL": "capitale_addizionale",
    "CAPITAL_BOOST_PERCENT": "boost_pct",
    "VAR_CONFIDENCE_LEVEL": "var_confidence",
    "BUY_THE_DIP_DRAWDOWN_LIMIT_PERCENT": "btd_dd_weekly_limit",
}

CAMPI_MODELLO_PREMIO = {"vrp", "vrp_slope", "vrp_ancora", "vrp_add", "target_delta",
                        "r", "q", "prem_floor", "prem_cap", "vrp_min", "vrp_max"}


def costruisci_config(params: Dict[str, Any]) -> BacktestConfig:
    """Accetta sia i nomi nuovi sia quelli del vecchio dizionario `params_gui`."""
    params = dict(params or {})
    for vecchio, nuovo in ALIAS_LEGACY.items():
        if vecchio in params and nuovo not in params:
            params[nuovo] = params.pop(vecchio)

    premio = params.pop("premium_model", None)
    if isinstance(premio, dict):
        premio = PremiumModel(**{k: v for k, v in premio.items() if k in CAMPI_MODELLO_PREMIO})
    elif not isinstance(premio, PremiumModel):
        premio = PremiumModel()
    # parametri del modello passati "piatti" nel dizionario
    piatti = {k: params.pop(k) for k in list(params) if k in CAMPI_MODELLO_PREMIO}
    if piatti:
        premio = PremiumModel(**{**premio.to_dict(), **piatti})

    validi = set(BacktestConfig.__dataclass_fields__)
    puliti = {k: v for k, v in params.items() if k in validi and k != "premium_model"}
    return BacktestConfig(premium_model=premio, **puliti)


# ----------------------------------------------------------------------------
# Catalogo dei grafici
# ----------------------------------------------------------------------------
# (chiave, flag di preferenza, titolo per la UI, costruttore, e' addizionale)
CATALOGO: List[Tuple[str, str, str, Callable[[Dict[str, Any]], Any], bool]] = [
    ("confronto_equity", "mostra_grafico_1", "Confronto delle equity",
     lambda r, **kw: charts.fig_confronto_equity(r, log=kw.get("log", False)), False),
    ("pnl_netto", "mostra_grafico_pnl", "Utile netto dei versamenti",
     lambda r, **kw: charts.fig_pnl_netto(r), False),
    ("verdetto_bh", "mostra_grafico_verdetto", "Verdetto contro il Buy & Hold",
     lambda r, **kw: charts.fig_verdetto_vs_bh(r), False),
    ("eq_dd_no_premi", "mostra_grafici_abc", "BTD No Premi — valore e drawdown",
     lambda r, **kw: charts.fig_equity_drawdown(r, "no_premi"), False),
    ("eq_dd_cash", "mostra_grafici_abc", "BTD + Premi (Cash) — valore e drawdown",
     lambda r, **kw: charts.fig_equity_drawdown(r, "premi_cash"), False),
    ("eq_dd_reinvest", "mostra_grafici_abc", "BTD + Premi (Reinvest) — valore e drawdown",
     lambda r, **kw: charts.fig_equity_drawdown(r, "premi_reinvest"), False),
    ("underwater", "mostra_grafico_underwater", "Drawdown a confronto",
     lambda r, **kw: charts.fig_underwater(r), False),
    ("btd", "mostra_grafico_5", "Acquisti Buy-The-Dip",
     lambda r, **kw: charts.fig_btd(r, kw.get("variante", "premi_cash")), False),
    ("dd_settimanale", "mostra_grafico_6", "Drawdown settimanale e filtro BTD",
     lambda r, **kw: charts.fig_dd_settimanale(r), False),
    ("rendimenti_annuali", "mostra_grafico_rend_annuali", "Rendimenti annuali operativi",
     lambda r, **kw: charts.fig_rendimenti_annuali(r), False),
    ("composizione_annuale", "mostra_grafico_composizione", "Composizione del risultato annuale",
     lambda r, **kw: charts.fig_composizione_annuale(r, kw.get("variante", "premi_cash")), False),
    ("premio", "mostra_grafico_premio", "Premio stimato e volatilita",
     lambda r, **kw: charts.fig_premio_stimato(r, kw.get("variante", "premi_cash")), False),
    ("prezzo_strike", "mostra_grafico_strike", "Prezzo del sottostante e strike",
     lambda r, **kw: charts.fig_prezzo_strike(r, kw.get("variante", "premi_cash")), False),
    ("heatmap", "mostra_grafici_addizionali", "Rendimenti mensili",
     lambda r, **kw: charts.fig_heatmap_mensile(r, kw.get("variante", "premi_reinvest")), True),
    ("distribuzione", "mostra_grafici_addizionali", "Distribuzione dei rendimenti",
     lambda r, **kw: charts.fig_distribuzione(r), True),
    # La finestra e' sempre lunga un anno: dodici barre sul mensile, cinquantadue
    # sul settimanale. La decide il grafico leggendo la cadenza del backtest.
    ("rolling", "mostra_grafici_addizionali", "Rendimento rolling a un anno",
     lambda r, **kw: charts.fig_rolling(r), True),
    ("rischio_rendimento", "mostra_grafici_addizionali", "Rischio e rendimento",
     lambda r, **kw: charts.fig_rischio_rendimento(r), True),
    ("durata_dd", "mostra_grafici_addizionali", "Durata degli episodi di drawdown",
     lambda r, **kw: charts.fig_durata_drawdown(r), True),
]

# Default: tutto acceso tranne i grafici addizionali.
PREFERENZE_DEFAULT: Dict[str, bool] = {
    flag: (flag != "mostra_grafici_addizionali") for _, flag, _, _, _ in CATALOGO
}


def costruisci_figure(
    risultato: Dict[str, Any],
    plot_prefs: Optional[Dict[str, Any]] = None,
    variante_dettaglio: str = "premi_cash",
    log_equity: bool = False,
) -> Dict[str, Any]:
    """Costruisce solo le figure richieste. Un grafico che fallisce non blocca gli altri."""
    prefs = {**PREFERENZE_DEFAULT, **(plot_prefs or {})}
    # I titoli sono scritti al mensile: qui diventano settimanali se serve.
    cadenza = normalizza((risultato.get("config") or {}).get("cadenza"))
    figure: Dict[str, Any] = {}
    titoli: Dict[str, str] = {}
    base: List[Any] = []
    extra: List[Any] = []
    errori: List[str] = []

    for chiave, flag, titolo, costruttore, addizionale in CATALOGO:
        if not prefs.get(flag, False):
            continue
        try:
            fig = costruttore(risultato, variante=variante_dettaglio, log=log_equity)
        except Exception as e:                       # un grafico rotto non ferma la dashboard
            errori.append(f"{adatta(titolo, cadenza)}: {e}")
            continue
        figure[chiave] = fig
        titoli[chiave] = adatta(titolo, cadenza)
        (extra if addizionale else base).append(fig)

    return {"figure": figure, "titoli": titoli, "figures": base,
            "figures_extra": extra, "errori": errori}


# ----------------------------------------------------------------------------
# Analisi completa
# ----------------------------------------------------------------------------
def esegui_analisi_completa(
    params_gui: Dict[str, Any],
    plot_prefs: Optional[Dict[str, Any]] = None,
    dati: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Scarica i dati (se non forniti), esegue il backtest e produce le figure.

    Il valore di ritorno resta un dizionario con le chiavi `figures` e
    `figures_extra` per compatibilita', ma contiene anche `risultato` con
    tutte le serie e le metriche, e `figure` con le figure indicizzate per nome.
    Le figure sono oggetti Plotly, non piu' matplotlib.
    """
    cfg = costruisci_config(params_gui)
    avvisi: List[str] = []

    if dati is None:
        try:
            dati = carica_serie(cfg.ticker, cfg.start_date, cfg.end_date)
        except (ChiaveMancante, DatiNonDisponibili) as e:
            return {"ok": False, "errore": str(e), "risultato": None,
                    "figure": {}, "figures": [], "figures_extra": [], "avvisi": []}
    avvisi += list(dati.get("avvisi", []))

    risultato = run_backtest(
        dati.get("mensile"),
        dati.get("settimanale") if isinstance(dati.get("settimanale"), pd.DataFrame) else None,
        dati.get("giornaliero") if isinstance(dati.get("giornaliero"), pd.DataFrame) else None,
        cfg,
    )
    if not risultato.get("ok"):
        return {"ok": False, "errore": risultato.get("errore", "Backtest fallito"),
                "risultato": risultato, "figure": {}, "figures": [],
                "figures_extra": [], "avvisi": avvisi}

    prefs = plot_prefs or {}
    figure = costruisci_figure(
        risultato, prefs,
        variante_dettaglio=str(prefs.get("variante_dettaglio", "premi_cash")),
        log_equity=bool(prefs.get("log_equity", False)),
    )
    return {"ok": True, "errore": None, "risultato": risultato,
            "avvisi": avvisi + list(risultato.get("warnings", [])), **figure}
