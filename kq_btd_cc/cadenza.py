"""Cadenza del backtest: mensile o settimanale.

La strategia e' la stessa, cambia solo la lunghezza del passo elementare. Qui
stanno le due cose che servono ovunque:

  * i numeri  -> quanti periodi entrano in un anno, quanti giorni dura la call;
  * le parole -> "mese" o "settimana" nei testi mostrati all'utente.

Il motore usa nomi di colonna storici (`rendimento_mese`, `twr_mese`,
`versamento_mese`) anche in cadenza settimanale: rinominarli avrebbe rotto
export, grafici e file gia' salvati senza aggiungere nulla. E' `adatta()` a
tradurre i testi al momento di scriverli a schermo, cosi' l'utente legge
"settimana" dove il codice dice ancora "mese".
"""
from __future__ import annotations

import re
from typing import Any, Dict

MENSILE = "mensile"
SETTIMANALE = "settimanale"

CADENZE: Dict[str, Dict[str, Any]] = {
    MENSILE: {
        "label": "Mensile",
        "periodi_anno": 12,
        "giorni": None,
        "singolare": "mese",
        "plurale": "mesi",
        "aggettivo": "mensile",
        "periodo_eodhd": "m",
        "descrizione": ("Una call al mese a scadenza mensile e un controllo del segnale "
                        "al mese: e' la versione originale della strategia."),
    },
    SETTIMANALE: {
        "label": "Settimanale",
        "periodi_anno": 52,
        "giorni": 7,
        "singolare": "settimana",
        "plurale": "settimane",
        "aggettivo": "settimanale",
        "periodo_eodhd": "w",
        "descrizione": ("Una call a settimana a scadenza settimanale e un controllo del "
                        "segnale a settimana. Il singolo premio vale circa la meta' di "
                        "quello mensile ma se ne incassano 52 invece di 12, e gli acquisti "
                        "sui cali diventano molto piu' frequenti: e' la versione aggressiva."),
    },
}

PERIODI_ANNO: Dict[str, int] = {k: int(v["periodi_anno"]) for k, v in CADENZE.items()}


def normalizza(cadenza: Any) -> str:
    """Qualunque input diventa una cadenza valida; l'ignoto ricade sul mensile."""
    c = str(cadenza or MENSILE).strip().lower()
    return c if c in CADENZE else MENSILE


def periodi_anno(cadenza: Any) -> int:
    return PERIODI_ANNO[normalizza(cadenza)]


def termini(cadenza: Any) -> Dict[str, Any]:
    return CADENZE[normalizza(cadenza)]


def label(cadenza: Any) -> str:
    return str(CADENZE[normalizza(cadenza)]["label"])


# ----------------------------------------------------------------------------
# Riscrittura dei testi
# ----------------------------------------------------------------------------
# In italiano "mese" e' maschile e "settimana" femminile, quindi non basta
# scambiare la parola: vanno riscritti anche gli articoli, le preposizioni
# articolate e gli aggettivi che le stanno intorno. Si procede a passi, mettendo
# prima un segnaposto al posto del sostantivo: cosi' le regole sul contorno
# riconoscono ancora dove stava la parola dopo che e' stata sostituita. Il
# segnaposto e' fatto di sole lettere, perche' i confini di parola lo vedano.
_S, _P = "qxsettimanaxq", "qxsettimanexq"

# 1. modi di dire che non sono una semplice sostituzione di sostantivo
_FRASI = [
    (r"ogni dodici mesi", "ogni anno"),
    (r"dodici mesi", "un anno"),
    (r"dodici volte l'anno", "cinquantadue volte l'anno"),
]

# 2. il sostantivo diventa segnaposto
_NOMI = [(r"mese", _S), (r"mesi", _P)]

# 3. aggettivi maschili che seguono il sostantivo
_CODA = [
    (_S + r" scorso", _S + " scorsa"),
    (_S + r" prossimo", _S + " prossima"),
    (_S + r" negativo", _S + " negativa"),
    (_S + r" positivo", _S + " positiva"),
    (_S + r" successivo", _S + " successiva"),
    (_P + r" negativi", _P + " negative"),
    (_P + r" positivi", _P + " positive"),
    (_P + r" consecutivi", _P + " consecutive"),
    (_P + r" pieni", _P + " piene"),
    (_P + r" trascorsi", _P + " trascorse"),
]

# 4. determinanti maschili che precedono il sostantivo
_TESTA = [
    (r"gli ultimi (\S+ )?" + _P, r"le ultime \1" + _P),
    (r"i primi (\S+ )?" + _P, r"le prime \1" + _P),
    (r"tutti i " + _P, "tutte le " + _P),
    (r"ogni singolo " + _S, "ogni singola " + _S),
    (r"ultimo " + _S, "ultima " + _S),
    (r"penultimo " + _S, "penultima " + _S),
    (r"primo " + _S, "prima " + _S),
    (r"stesso " + _S, "stessa " + _S),
    (r"questo " + _S, "questa " + _S),
    (r"quello " + _S, "quella " + _S),
    (r"quanti " + _P, "quante " + _P),
    (r"molti " + _P, "molte " + _P),
    (r"pochi " + _P, "poche " + _P),
    (r"ultimi " + _P, "ultime " + _P),
    (r"primi " + _P, "prime " + _P),
    (r"dello " + _S, "della " + _S),
    (r"nello " + _S, "nella " + _S),
    (r"allo " + _S, "alla " + _S),
    (r"del " + _S, "della " + _S),
    (r"dal " + _S, "dalla " + _S),
    (r"nel " + _S, "nella " + _S),
    (r"sul " + _S, "sulla " + _S),
    (r"col " + _S, "con la " + _S),
    (r"al " + _S, "alla " + _S),
    (r"il " + _S, "la " + _S),
    (r"un " + _S, "una " + _S),
    (r"lo " + _S, "la " + _S),
    (r"dei " + _P, "delle " + _P),
    (r"nei " + _P, "nelle " + _P),
    (r"sui " + _P, "sulle " + _P),
    (r"coi " + _P, "con le " + _P),
    (r"dai " + _P, "dalle " + _P),
    (r"ai " + _P, "alle " + _P),
    (r"gli " + _P, "le " + _P),
    (r"i " + _P, "le " + _P),
]

# 5. aggettivi derivati, che non hanno problemi di genere
_AGGETTIVI = [
    (r"mensilmente", "settimanalmente"),
    (r"mensili", "settimanali"),
    (r"mensile", "settimanale"),
]


def _compila(regole):
    return [(re.compile(r"\b" + p + r"\b", re.IGNORECASE), r) for p, r in regole]


_C_FRASI, _C_NOMI = _compila(_FRASI), _compila(_NOMI)
_C_CODA, _C_TESTA, _C_AGG = _compila(_CODA), _compila(_TESTA), _compila(_AGGETTIVI)
_RX_S, _RX_P = re.compile(_S, re.IGNORECASE), re.compile(_P, re.IGNORECASE)


def _sub(rx, nuovo: str, testo: str) -> str:
    """Sostituisce espandendo i gruppi e conservando la maiuscola iniziale."""
    def rimpiazza(m):
        out = m.expand(nuovo)
        return out[0].upper() + out[1:] if m.group(0)[:1].isupper() else out
    return rx.sub(rimpiazza, testo)


def _espandi(rx, parola: str, testo: str) -> str:
    return rx.sub(lambda m: (parola.capitalize() if m.group(0)[:1].isupper() else parola), testo)


def adatta(testo: Any, cadenza: Any = MENSILE) -> str:
    """Riscrive un testo scritto al mensile perche' parli della cadenza scelta.

    Sul mensile non tocca nulla. Va usata SOLO sui testi che parlano del passo
    della strategia: una frase come "gli ultimi sei mesi" riferita alla finestra
    di stima della volatilita' non va tradotta, perche' quella finestra si misura
    in giorni di borsa e non dipende dalla cadenza.
    """
    if testo is None:
        return testo
    s = str(testo)
    if normalizza(cadenza) != SETTIMANALE:
        return s
    for gruppo in (_C_FRASI, _C_NOMI, _C_CODA, _C_TESTA, _C_AGG):
        for rx, nuovo in gruppo:
            s = _sub(rx, nuovo, s)
    return _espandi(_RX_P, "settimane", _espandi(_RX_S, "settimana", s))


def adatta_dizionario(d: Dict[str, Any], cadenza: Any = MENSILE) -> Dict[str, Any]:
    """`adatta` applicata ai soli valori testuali di un dizionario di etichette."""
    if normalizza(cadenza) != SETTIMANALE:
        return dict(d)
    return {k: (adatta(v, cadenza) if isinstance(v, str) else v) for k, v in d.items()}
