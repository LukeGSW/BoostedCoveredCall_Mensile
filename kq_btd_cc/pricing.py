"""Pricing Black-Scholes senza dipendenze esterne (niente scipy).

Serve a stimare il premio di una call venduta a delta ~0.50 quando NON si
dispone della volatilita' implicita: la vol viene stimata dai prezzi storici
(vedi `vol.py`) e corretta con un Volatility Risk Premium calibrabile.

Tutte le funzioni ragionano su valori annualizzati; T e' in anni.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import erf, exp, log, sqrt
from typing import Dict

# ----------------------------------------------------------------------------
# Normale standard
# ----------------------------------------------------------------------------
_INV_SQRT_2PI = 0.3989422804014327


def norm_cdf(x: float) -> float:
    """CDF della normale standard (via math.erf, niente scipy)."""
    return 0.5 * (1.0 + erf(x / 1.4142135623730951))


def norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * exp(-0.5 * x * x)


# ----------------------------------------------------------------------------
# Black-Scholes
# ----------------------------------------------------------------------------
def _d1_d2(S: float, K: float, T: float, sigma: float, r: float, q: float):
    vt = sigma * sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vt
    return d1, d1 - vt


def bs_call_price(S: float, K: float, T: float, sigma: float,
                  r: float = 0.0, q: float = 0.0) -> float:
    """Prezzo di una call europea."""
    if not (S > 0 and K > 0):
        return 0.0
    if T <= 0 or sigma <= 0:
        return max(0.0, S * exp(-q * max(T, 0.0)) - K * exp(-r * max(T, 0.0)))
    d1, d2 = _d1_d2(S, K, T, sigma, r, q)
    return S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


def bs_call_delta(S: float, K: float, T: float, sigma: float,
                  r: float = 0.0, q: float = 0.0) -> float:
    """Delta (spot) di una call europea."""
    if not (S > 0 and K > 0):
        return 0.0
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1, _ = _d1_d2(S, K, T, sigma, r, q)
    return exp(-q * T) * norm_cdf(d1)


def strike_for_delta(S: float, T: float, sigma: float, target_delta: float = 0.50,
                     r: float = 0.0, q: float = 0.0, tol: float = 1e-10) -> float:
    """Strike che rende il delta della call pari a `target_delta`.

    Forma chiusa: delta = e^{-qT} N(d1) => d1 = N^{-1}(delta * e^{qT}).
    Qui si usa una bisezione, piu' robusta e senza inversa della normale.
    Nota: a delta 0.50 lo strike e' leggermente SOPRA lo spot (cresce con la vol).
    """
    if not (S > 0) or T <= 0 or sigma <= 0:
        return S
    lo, hi = S * 0.2, S * 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_call_delta(S, mid, T, sigma, r, q) > target_delta:
            lo = mid          # strike troppo basso -> delta troppo alto
        else:
            hi = mid
        if hi - lo < tol * S:
            break
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# Stimatore del premio
# ----------------------------------------------------------------------------
@dataclass
class PremiumModel:
    """Parametri dello stimatore di premio (nessun dato di IV richiesto).

    Il Volatility Risk Premium non e' una costante: misurato su 1.630 vendite
    reali di call ATM mensili su sei sottostanti, scende al crescere della
    volatilita' del sottostante. Vale circa 1.03 su un titolo che oscilla al 12%
    annuo e circa 0.83 su uno al 60%. Modellare questa pendenza riduce di un
    terzo l'errore sui ticker per cui non si hanno prezzi reali.

        vrp_effettivo(sigma) = vrp + vrp_slope * ln(sigma / vrp_ancora)

    vrp        : livello del VRP alla volatilita' di ancoraggio.
    vrp_slope  : pendenza rispetto al logaritmo della volatilita' (negativa).
                 Metterla a zero riporta al VRP costante.
    vrp_ancora : volatilita' a cui `vrp` e' riferito.
    vrp_add    : addendo in punti di vol annua (puo' essere negativo).
    target_delta: delta della call venduta (0.50 = "a delta 0.5").
    r, q       : tasso privo di rischio e dividend/convenience yield annui.
    prem_floor / prem_cap : limiti di sicurezza sul premio in frazione di spot.

    Attenzione: il LIVELLO dipende anche da quale stimatore di volatilita' lo
    alimenta, quindi resta un punto di partenza. La pendenza, che descrive una
    differenza fra sottostanti, e' molto piu' stabile.
    """
    vrp: float = 0.96
    vrp_slope: float = -0.125
    vrp_ancora: float = 0.20
    vrp_add: float = 0.0
    target_delta: float = 0.50
    r: float = 0.04
    q: float = 0.0
    prem_floor: float = 0.0005
    prem_cap: float = 0.35
    vrp_min: float = 0.30
    vrp_max: float = 3.00

    def vrp_effettivo(self, realized_sigma: float) -> float:
        """VRP applicato a questo livello di volatilita' realizzata."""
        if not self.vrp_slope:
            return self.vrp
        s = max(float(realized_sigma), 1e-6)
        ancora = max(self.vrp_ancora, 1e-6)
        return min(self.vrp_max, max(self.vrp_min,
                                     self.vrp + self.vrp_slope * log(s / ancora)))

    def implied_sigma(self, realized_sigma: float) -> float:
        """Da vol realizzata a vol implicita stimata."""
        if realized_sigma is None or realized_sigma != realized_sigma or realized_sigma <= 0:
            return 0.0
        return max(1e-6, realized_sigma * self.vrp_effettivo(realized_sigma) + self.vrp_add)

    def quote(self, spot: float, realized_sigma: float, T: float) -> Dict[str, float]:
        """Restituisce strike, sigma usata, premio in valuta e in frazione di spot.

        Il premio in valuta e' sempre riferito allo spot corrente: usandolo su
        N azioni si ottiene N * spot * premium_pct, cioe' un incasso diverso
        ogni mese al variare del prezzo del sottostante.
        """
        sigma = self.implied_sigma(realized_sigma)
        if not (spot > 0) or T <= 0 or sigma <= 0:
            return {"strike": float(spot), "sigma": 0.0, "premium": 0.0,
                    "premium_pct": 0.0, "moneyness": 1.0, "vrp": 0.0}
        K = strike_for_delta(spot, T, sigma, self.target_delta, self.r, self.q)
        c = bs_call_price(spot, K, T, sigma, self.r, self.q)
        pct = min(self.prem_cap, max(self.prem_floor, c / spot))
        return {"strike": float(K), "sigma": float(sigma),
                "premium": float(pct * spot), "premium_pct": float(pct),
                "moneyness": float(K / spot),
                "vrp": float(self.vrp_effettivo(realized_sigma))}

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def implied_vol_from_call(price: float, S: float, K: float, T: float,
                          r: float = 0.0, q: float = 0.0,
                          lo: float = 1e-4, hi: float = 5.0) -> float:
    """Volatilita' implicita di una call, per bisezione sul prezzo.

    Serve a leggere la volatilita' che il mercato stava effettivamente
    prezzando nei premi reali, per poi confrontarla con quella realizzata.
    Restituisce NaN se il prezzo e' fuori dai limiti di non arbitraggio.
    """
    if not (price > 0 and S > 0 and K > 0 and T > 0):
        return float("nan")
    minimo = max(0.0, S * exp(-q * T) - K * exp(-r * T))
    massimo = S * exp(-q * T)
    if price <= minimo + 1e-12 or price >= massimo:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_call_price(S, K, T, mid, r, q) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break
    return 0.5 * (lo + hi)


def atm_premium_pct_approx(sigma: float, T: float) -> float:
    """Approssimazione di Brenner-Subrahmanyam: C/S ~= 0.4 * sigma * sqrt(T).

    Tenuta solo come riferimento diagnostico: sovrastima il prezzo esatto
    dal +2% (vol 12%) al +19% (vol 100%) su scadenza mensile, quindi il
    motore usa sempre il Black-Scholes completo.
    """
    return 0.3989422804014327 * sigma * sqrt(max(T, 0.0))
