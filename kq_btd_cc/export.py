"""Costruzione del pacchetto JSON scaricabile con l'intero backtest.

Contiene tutto quello che serve a rifare i conti fuori dalla dashboard:
parametri, serie completa di ogni variante, tabelle annuali, metriche, flussi
di cassa, benchmark, volatilita' stimata e calibrazione.

I nomi dei campi restano quelli mensili (`serie_mensile`, `rendimento_mese`)
anche quando il backtest gira a cadenza settimanale: cambiarli avrebbe reso
illeggibili i file salvati in passato. Il campo `parametri.cadenza` dice sempre
di che passo si tratta, e `strategia.cadenza` lo ripete in chiaro.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .cadenza import adatta, label as etichetta_cadenza, normalizza, periodi_anno
from .utils import dataframe_to_records, json_safe

SCHEMA_VERSION = "2.0"

# Descrizione dei campi della serie mensile: rende il JSON leggibile fra sei mesi.
DIZIONARIO_CAMPI: Dict[str, str] = {
    "data": "Data di chiusura della barra mensile",
    "anno": "Anno solare (la strategia si azzera a ogni cambio d'anno)",
    "open": "Apertura del mese, prezzo a cui si impiega il capitale e si fissa lo strike",
    "close": "Chiusura del mese, prezzo di scadenza della call",
    "rendimento_mese": "Rendimento del sottostante nel mese (close su close)",
    "segnale_btd": "True se il mese precedente ha chiuso in negativo",
    "btd_bloccato": "True se il filtro sul drawdown settimanale ha bloccato l'acquisto",
    "dd_weekly": "Drawdown settimanale dell'asset alla data della decisione",
    "btd_importo": "Capitale investito nel Buy-The-Dip in questo mese, boost incluso",
    "btd_quota_calo": "Parte dell'acquisto legata all'entita' del calo del mese precedente",
    "btd_quota_boost": "Parte dell'acquisto dovuta al boost pianificato a inizio anno",
    "btd_residuo_anno": "Capitale ancora disponibile sotto il tetto annuo dei BTD; "
                        "vuoto quando non e' impostato alcun tetto (default)",
    "capitale_impiegato_anno": "Capitale messo al lavoro nel ciclo annuale in corso: "
                               "capitale fisso piu' i Buy-The-Dip dell'anno. Si azzera "
                               "a ogni gennaio, a differenza di versamenti_cum",
    "btd_prezzo": "Prezzo di esecuzione dell'acquisto BTD",
    "sigma_stimata": "Volatilita' realizzata annualizzata nota prima dell'inizio del mese",
    "sigma_implicita": "Volatilita' usata per prezzare la call (realizzata x VRP)",
    "vrp_applicato": "VRP effettivo del mese, funzione del livello di volatilita'",
    "strike": "Strike della call venduta (delta 0.50 o ATM secondo configurazione)",
    "premio_pct": "Premio incassato in frazione del prezzo del sottostante",
    "premio": "Premio incassato in valuta (quote coperte x open x premio_pct)",
    "intrinseco_pagato": "Valore intrinseco pagato a scadenza se la call e' in-the-money",
    "netto_opzione": "Premio meno intrinseco: risultato netto della call nel mese",
    "reinvestito": "Importo reinvestito in quote (solo variante Reinvest)",
    "quote_coperte": "Quote coperte dalla covered call, costanti nell'anno",
    "quote_extra": "Quote non coperte (capitale addizionale, BTD, reinvestimenti)",
    "cassa": "Liquidita' totale del conto, operativa piu' quella dei premi; negativa "
             "quando il riacquisto della call e' finanziato a debito contro le azioni",
    "cassa_opzioni": "Parte della liquidita' che viene dai premi incassati meno "
                     "l'intrinseco pagato. Nella variante Cash resta separata e non "
                     "finanzia gli acquisti sui cali, che si pagano con capitale proprio",
    "interessi": "Interessi del mese: attivi sulla cassa positiva, passivi sul saldo a debito",
    "liquidazione": "Controvalore liquidato al reset di inizio anno",
    "valore_portafoglio": "Valore totale del conto: quote a mercato piu' cassa",
    "versamento_mese": "Denaro entrato dall'esterno in questo mese",
    "versamenti_cum": "Denaro entrato dall'esterno dall'inizio del backtest; non si azzera "
                      "a gennaio perche' e' il metro con cui si misura l'utile",
    "pnl_netto": "Utile vero: valore_portafoglio meno versamenti_cum",
    "bh_stessi_flussi": "Valore di un buy and hold che riceve gli stessi versamenti e non "
                        "liquida mai; su un sottostante che si moltiplica per ordini di "
                        "grandezza esce dalla scala di tutto il resto",
    "ciclo_annuale": "Valore del solo sottostante comprato e liquidato con lo stesso ciclo "
                     "annuale della strategia: il confronto a parita' di mandato",
    "ciclo_annuale_pnl": "Utile netto del solo sottostante con lo stesso ciclo annuale",
    "ciclo_annuale_twr": "Rendimento mensile time-weighted del solo sottostante",
    "ciclo_annuale_dd": "Drawdown percentuale del solo sottostante",
    "twr_mese": "Rendimento time-weighted del mese, ripulito dai flussi",
    "indice_twr": "Indice dei rendimenti time-weighted (base 1)",
    "dd_valore": "Drawdown del valore del conto, in valuta",
    "dd_twr_pct": "Drawdown percentuale della strategia (su indice_twr)",
    "pnl_dd": "Drawdown dell'utile netto, in valuta",
}


def _serie_to_records(s: Optional[pd.Series], nome: str) -> list:
    if s is None or not isinstance(s, pd.Series) or s.empty:
        return []
    df = s.dropna().to_frame(name=nome)
    return dataframe_to_records(df)


def build_export(
    risultato: Dict[str, Any],
    calibrazione: Optional[Dict[str, Any]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Assembla il dizionario completo da serializzare."""
    if not risultato or not risultato.get("ok"):
        return {
            "schema": SCHEMA_VERSION,
            "generato_il": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ok": False,
            "errore": (risultato or {}).get("errore", "Backtest non eseguito"),
        }

    cfg = risultato["config"]
    mercato = risultato.get("mercato", {})

    # Il benchmark viene esportato insieme alle varianti: ha la stessa struttura
    # e serve a rifare qualunque confronto fuori dalla dashboard.
    tutte = dict(risultato.get("varianti", {}))
    if risultato.get("benchmark"):
        tutte["benchmark"] = risultato["benchmark"]

    varianti: Dict[str, Any] = {}
    for chiave, res in tutte.items():
        mdf: pd.DataFrame = res.get("monthly", pd.DataFrame())
        ydf: pd.DataFrame = res.get("yearly", pd.DataFrame())
        if mdf.empty:
            continue
        equity = mdf[["valore_portafoglio", "pnl_netto", "versamenti_cum", "indice_twr"]]
        flussi = mdf.loc[mdf["versamento_mese"] > 0, ["versamento_mese", "versamenti_cum"]]
        varianti[chiave] = {
            "etichetta": res.get("label", chiave),
            "metriche": json_safe(res.get("metrics", {})),
            "equity": dataframe_to_records(equity),
            "serie_mensile": dataframe_to_records(mdf),
            "tabella_annuale": dataframe_to_records(ydf, index_name="anno"),
            "flussi_di_cassa": dataframe_to_records(flussi),
        }

    prezzi = mercato.get("prezzi")
    cadenza = normalizza(cfg.get("cadenza"))
    export: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "generato_il": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": True,
        "note": note,
        "strategia": {
            "nome": f"Boosted Covered Call {cadenza} con Buy-The-Dip",
            "cadenza": cadenza,
            "periodi_per_anno": periodi_anno(cadenza),
            "descrizione": adatta(
                "Capitale fisso impiegato a inizio anno e coperto da una call mensile "
                "venduta a delta 0.50; acquisti Buy-The-Dip dopo ogni mese negativo, "
                "maggiorati di un boost; liquidazione e reset a fine anno.", cadenza
            ),
            "convenzioni": {
                "cap_covered_call": (
                    "La call viene riacquistata al valore intrinseco a scadenza: il costo "
                    "dell'upside tagliato si accumula davvero mese dopo mese."
                ),
                "premio": (
                    "Il premio e' una percentuale del prezzo corrente del sottostante, "
                    "quindi l'incasso in valuta cambia ogni mese."
                ),
                "contabilita": (
                    "versamenti_cum traccia il denaro entrato dall'esterno; "
                    "pnl_netto = valore_portafoglio - versamenti_cum."
                ),
                "rendimenti": (
                    "Il rendimento di ogni anno e' semplice, non composto: risultato "
                    "dell'anno diviso il capitale davvero investito in quel ciclo "
                    "(capitale di gennaio piu' gli acquisti sui cali). I premi "
                    "reinvestiti non entrano al denominatore perche' arrivano dal "
                    "mercato. Il rischio resta misurato sul rendimento time-weighted "
                    "mensile, che neutralizza i flussi di cassa."),
                "benchmark": (
                    "Il confronto corretto e' 'solo sottostante, stesso ciclo annuale': "
                    "stesso capitale fisso impiegato a gennaio e liquidato a dicembre, "
                    "senza opzioni e senza Buy-The-Dip. Il buy and hold classico non "
                    "liquida mai, quindi su un sottostante molto direzionale produce "
                    "valori che non sono confrontabili."),
            },
        },
        "parametri": json_safe(cfg),
        "mercato": {
            "ticker": cfg.get("ticker"),
            "fonte_volatilita": mercato.get("vol_source"),
            "prezzi_mensili": dataframe_to_records(prezzi) if isinstance(prezzi, pd.DataFrame) else [],
            "volatilita_realizzata": _serie_to_records(mercato.get("vol"), "sigma"),
            "drawdown_settimanale": _serie_to_records(mercato.get("dd_weekly"), "dd_weekly"),
            "buy_and_hold_semplice": _serie_to_records(mercato.get("bh_semplice"), "valore"),
        },
        "varianti": varianti,
        "calibrazione_premio": json_safe(calibrazione) if calibrazione else None,
        "avvisi": list(risultato.get("warnings", [])),
        "dizionario_campi": {k: adatta(v, cadenza) for k, v in DIZIONARIO_CAMPI.items()},
    }
    # Le convenzioni sono scritte al mensile: sul settimanale vanno tradotte,
    # altrimenti il file spiega una strategia diversa da quella che ha girato.
    conv = export["strategia"]["convenzioni"]
    export["strategia"]["convenzioni"] = {k: adatta(v, cadenza) for k, v in conv.items()}
    export["strategia"]["cadenza_label"] = etichetta_cadenza(cadenza)
    return export


def export_json_bytes(export: Dict[str, Any], indent: int = 2) -> bytes:
    """Serializza in UTF-8, pronto per st.download_button."""
    return json.dumps(export, ensure_ascii=False, indent=indent, default=str).encode("utf-8")


def nome_file_export(cfg: Dict[str, Any]) -> str:
    ticker = str(cfg.get("ticker", "backtest")).replace(".", "_").replace("/", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"boosted_covered_call_{ticker}_{stamp}.json"
