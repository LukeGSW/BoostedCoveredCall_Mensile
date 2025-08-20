from __future__ import annotations
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

class AnsiColors:
    """
    Classe contenitore per i codici di escape ANSI per la colorazione del testo in console.
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    # Aggiunto GRAY per completezza, sebbene non usato nell'originale Cella 1
    GRAY = "\033[90m"


def colorize(text: str, color_code: str) -> str:
    """
    Applica un codice colore ANSI a una stringa di testo.

    Args:
        text: La stringa di testo da colorare.
        color_code: Il codice colore ANSI (es. AnsiColors.GREEN).

    Returns:
        La stringa di testo formattata con il codice colore.
    """
    return f"{color_code}{text}{AnsiColors.RESET}"


def strip_ansi(text: any) -> any:
    """
    Rimuove i codici di escape ANSI da una stringa di testo.
    Se l'input non è una stringa, lo restituisce invariato.

    Args:
        text: La stringa da cui rimuovere i codici ANSI, o un altro tipo di dato.

    Returns:
        La stringa senza codici ANSI, oppure l'input originale se non è una stringa.
    """
    if not isinstance(text, str):
        return text
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def format_currency(value: float) -> str:
    """
    Converte un numero in una stringa formattata come valuta in USD, con due decimali.

    Args:
        value: Il numero da formattare.

    Returns:
        La stringa formattata come valuta.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "$0.00"
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    """
    Converte un numero decimale in una stringa percentuale con due decimali.

    Args:
        value: Il numero decimale da formattare (es. 0.1234 -> "12.34%").

    Returns:
        La stringa in formato percentuale.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "0.00%"
    return f"{value*100:,.2f}%"


def format_ratio(value: float) -> str:
    """
    Converte un numero in stringa con due decimali per rapporti (es. Sharpe/Sortino).

    Args:
        value: Il numero da formattare.

    Returns:
        Stringa con due decimali.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "0.00"
    return f"{value:,.2f}"


def currency_formatter(x: float, pos: int) -> str:
    """
    Formatter per gli assi di Matplotlib per visualizzare valori come valuta (senza decimali).

    Args:
        x: Il valore dell'asse.
        pos: La posizione del tick (non usata).

    Returns:
        Stringa formattata come valuta.
    """
    return f'${x:,.0f}'


def percentage_formatter(x: float, pos: int) -> str:
    """
    Formatter per gli assi di Matplotlib per visualizzare valori percentuali.

    Args:
        x: Il valore dell'asse (in decimali, es. 0.15 per 15%).
        pos: La posizione del tick (non usata).

    Returns:
        Stringa formattata come percentuale.
    """
    return f'{x*100:.0f}%'


def add_watermark(fig: plt.Figure, text: str, fontsize: int = 8, alpha: float = 0.3):
    """
    Aggiunge un watermark al grafico specificato.

    Args:
        fig: L'oggetto Matplotlib Figure.
        text: Il testo del watermark.
        fontsize: La dimensione del font del watermark.
        alpha: La trasparenza del watermark.
    """
    fig.text(0.99, 0.01, text, ha='right', va='bottom', fontsize=fontsize, alpha=alpha)


def setup_common_axis_elements(ax: plt.Axes, title: str, xlabel: str, ylabel: str, grid_alpha: float = 0.4, fontsize_title: int = 16, fontsize_axis_label: int = 14):
    """
    Imposta elementi comuni dell'asse, come titolo, etichette e griglia.

    Args:
        ax: L'asse Matplotlib da configurare.
        title: Il titolo del grafico.
        xlabel: L'etichetta dell'asse x.
        ylabel: L'etichetta dell'asse y.
        grid_alpha: Trasparenza della griglia.
        fontsize_title: Dimensione del font del titolo.
        fontsize_axis_label: Dimensione del font delle etichette degli assi.
    """
    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlabel(xlabel, fontsize=fontsize_axis_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis_label)
    ax.grid(alpha=grid_alpha)


def setup_date_axis(ax: plt.Axes, series_index: pd.Index):
    """
    Configura l'asse delle date, impostando formattazione e limiti basati sull'indice della serie.

    Args:
        ax: L'asse Matplotlib.
        series_index: L'indice della serie (DateTimeIndex).
    """
    if series_index is not None and len(series_index) > 0:
        ax.set_xlim(series_index.min(), series_index.max())
    ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax.tick_params(axis='x', rotation=0)


def check_plot_data(series: pd.Series) -> bool:
    """
    Controlla se una serie Pandas contiene dati validi e sufficienti per il plotting.
    Un dato è considerato plottabile se non è None, non è vuoto e ha più di un valore unico.

    Args:
        series: La serie Pandas da controllare.

    Returns:
        True se la serie è plottabile, False altrimenti.
    """
    return series is not None and not series.empty and series.nunique() > 1


def compute_drawdown_monetary(equity_series: pd.Series) -> pd.Series:
    """
    Calcola il drawdown monetario di una serie di equity.
    Il drawdown è la differenza tra il picco corrente e il valore attuale dell'equity.

    Args:
        equity_series: Una serie Pandas rappresentante la curva di equity.

    Returns:
        Una serie Pandas rappresentante il drawdown monetario (valori <= 0).
    """
    if equity_series is None or equity_series.empty:
        return pd.Series(dtype=float)
    running_max = equity_series.cummax()
    drawdown_dollars = equity_series - running_max
    return drawdown_dollars.clip(upper=0)  # Assicura che i valori siano <= 0
