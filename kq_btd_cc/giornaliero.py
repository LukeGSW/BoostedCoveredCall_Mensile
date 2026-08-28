"""Rivalutazione giorno per giorno del conto costruito sulle barre di periodo.

PERCHE' SERVE. Il motore opera sulle barre della cadenza scelta e valorizza il
conto una volta per barra, alla chiusura. Un crollo del 30% a meta' mese che
rientra prima dell'ultimo giorno non lascia traccia da nessuna parte: la curva
mensile passa da un massimo all'altro e il drawdown misurato resta zero. Su
barre settimanali il buco e' piu' piccolo ma c'e' lo stesso. Il drawdown vero,
quello che si vede sull'estratto conto e che fa scattare le chiamate a margine,
si misura sui prezzi di ogni giorno.

COSA FA. Non cambia nulla di come si opera: le decisioni restano sulla griglia
del periodo, perche' e' quella la strategia. Cambia solo la VALORIZZAZIONE, che
diventa giornaliera. La posizione dentro il periodo e' nota:

  * le quote coperte si comprano all'apertura del periodo e restano ferme;
  * gli acquisti sui cali entrano all'apertura (o alla chiusura, se configurato);
  * i premi si incassano all'apertura;
  * l'intrinseco si paga alla chiusura;
  * i premi si reinvestono alla chiusura, oppure — se il reinvestimento e'
    differito al Buy-The-Dip — nello stesso momento dell'acquisto sui cali.

Fra un estremo e l'altro cambia solo il prezzo. Percio' ogni giorno:

    valore = quote * prezzo + liquidita' - valore della call venduta

La call venduta e' un debito finche' non scade: si valuta con Black-Scholes
sullo strike e sulla volatilita' implicita che il motore ha gia' usato per
incassarne il premio, con il tempo residuo che si consuma giorno dopo giorno.
All'ultimo giorno il tempo residuo e' zero, la formula restituisce il valore
intrinseco e il conto giornaliero RITORNA ESATTAMENTE al valore di fine periodo
calcolato dal motore. E' la verifica che tiene onesto tutto il resto: se i due
non coincidono al centesimo, la ricostruzione e' sbagliata.

IL PEGGIO VISTO IN GIORNATA. Oltre alla chiusura si valorizza anche il minimo
della giornata, con lo stesso metodo. E' un'approssimazione dichiarata — nessuno
sa in che ordine il prezzo abbia toccato i suoi estremi — ma dice quanto in
basso e' arrivato il conto se il minimo fosse stato il prezzo di quel momento.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .cadenza import normalizza
from .pricing import bs_call_price

GIORNI_ANNO = 252.0

# Colonne del periodo che servono a ricostruire la posizione dentro il periodo.
COLONNE_RICHIESTE = ("open", "close", "quote_coperte", "quote_extra", "cassa",
                     "reinvestito", "intrinseco_pagato", "btd_importo",
                     "versamento_mese", "valore_portafoglio")


def _bs_vettoriale(S, K, T, sigma, r: float, q: float) -> np.ndarray:
    """Black-Scholes su array. Con T <= 0 restituisce il valore intrinseco."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    out = np.zeros_like(S)
    valide = np.isfinite(K) & (K > 0) & np.isfinite(S) & (S > 0)
    for i in np.flatnonzero(valide):
        out[i] = bs_call_price(float(S[i]), float(K[i]), float(T[i]),
                               float(sigma[i]) if np.isfinite(sigma[i]) else 0.0, r, q)
    return out


def _confini(periodi: pd.DataFrame, cadenza: str) -> tuple:
    """Primo e ultimo giorno di calendario di ogni periodo."""
    idx = pd.DatetimeIndex(periodi.index)
    if normalizza(cadenza) == "settimanale":
        inizio = idx - pd.to_timedelta(idx.weekday, unit="D")
        fine = inizio + pd.Timedelta(days=6)
    else:
        inizio = pd.DatetimeIndex([pd.Timestamp(year=t.year, month=t.month, day=1) for t in idx])
        fine = inizio + pd.offsets.MonthEnd(0)
    return inizio.normalize(), fine.normalize()


def valorizza_giornaliero(
    periodi: pd.DataFrame,
    giornaliero: Optional[pd.DataFrame],
    cfg: Any,
    r: float = 0.0,
    q: float = 0.0,
) -> pd.DataFrame:
    """Serie giornaliera del valore del conto per una variante gia' simulata.

    `periodi` e' la tabella per barra prodotta da `run_variant`; `giornaliero`
    sono gli OHLC giornalieri dello stesso sottostante. Restituisce un
    DataFrame vuoto se i dati giornalieri non ci sono: la dashboard in quel caso
    mostra soltanto la valorizzazione di fine periodo, dicendolo.
    """
    if periodi is None or periodi.empty:
        return pd.DataFrame()
    if giornaliero is None or not isinstance(giornaliero, pd.DataFrame) or giornaliero.empty:
        return pd.DataFrame()
    if any(c not in periodi.columns for c in COLONNE_RICHIESTE):
        return pd.DataFrame()

    cadenza = normalizza(getattr(cfg, "cadenza", "mensile"))
    applica_cap = bool(getattr(cfg, "applica_cap", True))
    btd_al_close = str(getattr(cfg, "btd_execution", "open")) == "close"

    g = giornaliero.copy()
    g = g[~g.index.duplicated(keep="last")].sort_index()
    g.index = pd.DatetimeIndex(g.index).normalize()
    if "Low" not in g.columns or g["Low"].isna().all():
        g["Low"] = g[["Open", "Close"]].min(axis=1)
    g = g[np.isfinite(g["Close"]) & (g["Close"] > 0)]
    if g.empty:
        return pd.DataFrame()

    inizio, fine = _confini(periodi, cadenza)
    # Ogni giorno finisce nel periodo che lo contiene; quelli fuori si scartano.
    pos = np.searchsorted(inizio.values, g.index.values, side="right") - 1
    dentro = (pos >= 0) & (pos < len(periodi))
    pos = np.where(dentro, pos, 0)
    dentro &= g.index.values <= fine.values[pos]
    g = g[dentro]
    pos = pos[dentro]
    if g.empty:
        return pd.DataFrame()

    P = periodi
    chiusura = P["close"].to_numpy(float)
    qc = P["quote_coperte"].to_numpy(float)
    qe_fine = P["quote_extra"].to_numpy(float)
    cassa_fine = P["cassa"].to_numpy(float)
    # Il reinvestimento puo' cadere in due momenti diversi del periodo: alla
    # chiusura (modalita' "subito") oppure insieme all'acquisto sui cali
    # (modalita' "al_btd"). Solo quello di chiusura va scorporato dalla
    # posizione di meta' periodo; quello al BTD e' gia' dentro dal momento in
    # cui il BTD e' stato eseguito.
    if "reinvestito_a_chiusura" in P:
        reinv_chiusura = P["reinvestito_a_chiusura"].to_numpy(float)
        reinv_btd = P["reinvestito_al_btd"].to_numpy(float)
    else:
        reinv_chiusura = (P["reinvestito"].to_numpy(float) if "reinvestito" in P
                          else np.zeros(len(P)))
        reinv_btd = np.zeros(len(P))
    # Se il BTD si esegue alla chiusura, anche i suoi arretrati cadono li'.
    if btd_al_close:
        reinv_chiusura = reinv_chiusura + reinv_btd
    reinvestito = reinv_chiusura
    intrinseco = P["intrinseco_pagato"].to_numpy(float)
    btd_importo = P["btd_importo"].to_numpy(float)
    btd_prezzo = P["btd_prezzo"].to_numpy(float) if "btd_prezzo" in P else np.full(len(P), np.nan)
    strike = P["strike"].to_numpy(float) if "strike" in P else np.full(len(P), np.nan)
    sigma = (P["sigma_implicita"].to_numpy(float) if "sigma_implicita" in P
             else np.full(len(P), np.nan))
    versamento = P["versamento_mese"].to_numpy(float)

    # Quote e liquidita' DENTRO il periodo, cioe' prima dei movimenti di chiusura.
    quote_reinv = np.where(chiusura > 0, reinvestito / np.where(chiusura > 0, chiusura, 1.0), 0.0)
    quote_btd_close = np.zeros(len(P))
    if btd_al_close:
        ok = np.isfinite(btd_prezzo) & (btd_prezzo > 0)
        quote_btd_close = np.where(ok, btd_importo / np.where(ok, btd_prezzo, 1.0), 0.0)
    qe_intra = qe_fine - quote_reinv - quote_btd_close
    cassa_intra = (cassa_fine + intrinseco + reinvestito
                   + (btd_importo if btd_al_close else 0.0))
    #   (qc + qe_intra) * C + cassa_intra - intrinseco = valore di fine periodo

    # Scadenza della call: l'ultimo giorno di borsa del periodo, cosi' il tempo
    # residuo si annulla esattamente li' e la formula restituisce l'intrinseco.
    ultimo_giorno = pd.Series(g.index.values).groupby(pos).max().reindex(range(len(P)))
    scadenza = ultimo_giorno.to_numpy()
    e_ultimo = g.index.values == scadenza[pos]
    e_primo = np.concatenate([[True], pos[1:] != pos[:-1]])

    # La composizione e' sempre quella di DENTRO il periodo, con la call ancora
    # aperta e segnata a mercato come debito. All'ultimo giorno il tempo residuo
    # si annulla, il debito diventa il valore intrinseco e i conti tornano da
    # soli al valore di fine periodo: e' l'identita' che verifica tutto.
    #   (qc + qe_intra) * C + cassa_intra - intrinseco = (qc + qe_fine) * C + cassa_fine
    # Trattare l'ultimo giorno come gia' liquidato sottrarrebbe l'intrinseco due
    # volte, perche' la cassa di fine periodo l'ha gia' pagato.
    quote = qc[pos] + qe_intra[pos]
    cassa = cassa_intra[pos]

    giorni_residui = (scadenza[pos] - g.index.values) / np.timedelta64(1, "D")
    tau = np.maximum(giorni_residui, 0.0) / 365.0
    # Senza cap la call non viene mai riacquistata: il premio e' incassato e
    # basta, quindi non c'e' nessun debito da segnare a mercato.
    K = strike[pos] if applica_cap else np.full(len(pos), np.nan)
    S = g["Close"].to_numpy(float)
    L = np.minimum(g["Low"].to_numpy(float), S)

    passivo = qc[pos] * _bs_vettoriale(S, K, tau, sigma[pos], r, q)
    passivo_min = qc[pos] * _bs_vettoriale(L, K, tau, sigma[pos], r, q)

    valore = quote * S + cassa - passivo
    valore_min = quote * L + cassa - passivo_min

    # I versamenti entrano all'apertura del periodo, quindi il primo giorno.
    versamento_giorno = np.where(e_primo, versamento[pos], 0.0)

    out = pd.DataFrame({
        "close": S,
        "low": L,
        "anno": P["anno"].to_numpy()[pos] if "anno" in P else g.index.year,
        "periodo": periodi.index.values[pos],
        "fine_periodo": e_ultimo,
        "quote": quote,
        "cassa": cassa,
        "valore_call": passivo,
        "valore_portafoglio": valore,
        "valore_minimo": np.minimum(valore_min, valore),
        "versamento_giorno": versamento_giorno,
    }, index=g.index)
    out.index.name = "data"

    out["versamenti_cum"] = out["versamento_giorno"].cumsum()
    out["pnl_netto"] = out["valore_portafoglio"] - out["versamenti_cum"]

    base = out["valore_portafoglio"].shift(1).fillna(0.0) + out["versamento_giorno"]
    out["twr_giorno"] = np.where(base > 0, out["valore_portafoglio"] / base - 1.0, 0.0)
    out["indice_twr"] = (1.0 + out["twr_giorno"]).cumprod()

    picco = out["valore_portafoglio"].cummax()
    out["dd_valore"] = (out["valore_portafoglio"] - picco).clip(upper=0.0)
    picco_twr = out["indice_twr"].cummax()
    out["dd_twr_pct"] = (out["indice_twr"] / picco_twr - 1.0).clip(upper=0.0)

    # Il minimo di giornata va portato sulla stessa scala time-weighted, altrimenti
    # si confronterebbe un valore gonfiato dai versamenti con un indice che li
    # neutralizza, e il "peggio visto" risulterebbe piu' tenero della chiusura.
    indice_min = (out["indice_twr"].shift(1).fillna(1.0)
                  * np.where(base > 0, out["valore_minimo"] / base.replace(0, np.nan), 1.0))
    out["dd_intraday_pct"] = ((indice_min / picco_twr - 1.0)
                              .clip(upper=0.0).fillna(0.0)
                              .combine(out["dd_twr_pct"], min))
    return out


# ----------------------------------------------------------------------------
# Metriche
# ----------------------------------------------------------------------------
def _durate(dd: pd.Series) -> list:
    out, cur = [], 0
    for v in dd.to_numpy():
        if v < -1e-12:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def metriche_giornaliere(gio: pd.DataFrame, periodi: pd.DataFrame,
                         confidence: float = 0.99) -> Dict[str, Any]:
    """Rischio misurato sui prezzi di ogni giorno, e quanto ne nascondeva il periodo."""
    if gio is None or gio.empty:
        return {}

    dd = gio["dd_twr_pct"]
    durate = _durate(dd)
    rg = gio["twr_giorno"].replace([np.inf, -np.inf], np.nan).dropna()
    var = None
    if len(rg) >= 20:
        var = _f(np.quantile(rg.to_numpy(), 1.0 - float(confidence)))

    dd_periodo = _f(periodi["dd_twr_pct"].min()) if "dd_twr_pct" in periodi else None
    dd_gior = _f(dd.min())
    nascosto = (dd_gior - dd_periodo) if (dd_gior is not None and dd_periodo is not None) else None

    # Riconciliazione: all'ultimo giorno di ogni periodo il conto giornaliero
    # deve tornare esattamente al valore calcolato dal motore.
    fine = gio[gio["fine_periodo"]]
    scarto = None
    if not fine.empty:
        atteso = periodi["valore_portafoglio"].reindex(pd.Index(fine["periodo"]))
        diff = (fine["valore_portafoglio"].to_numpy() - atteso.to_numpy())
        scala = max(1.0, float(np.nanmax(np.abs(atteso.to_numpy()))))
        scarto = _f(np.nanmax(np.abs(diff)) / scala)

    return {
        "giorni": int(len(gio)),
        "max_dd_giornaliero_pct": dd_gior,
        "max_dd_giornaliero_valore": _f(gio["dd_valore"].min()),
        "max_dd_intraday_pct": _f(gio["dd_intraday_pct"].min()),
        "dd_giornaliero_durata_max": int(max(durate)) if durate else 0,
        "dd_giornaliero_durata_media": _f(np.mean(durate)) if durate else None,
        "var_giornaliero": var,
        "peggior_giorno": _f(rg.min()) if len(rg) else None,
        "miglior_giorno": _f(rg.max()) if len(rg) else None,
        "dd_nascosto_dal_periodo": _f(nascosto),
        "riconciliazione_scarto": scarto,
    }
