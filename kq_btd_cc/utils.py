"""Utilità generali per la strategia BTD + Covered Call.

Nota: Niente I/O su disco, niente dipendenze non presenti in requirements.txt.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.dates import MonthLocator, YearLocator, DateFormatter


# ----------------------------------------------------------------------------
# ANSI / Console helpers (facoltativi)
# ----------------------------------------------------------------------------
class AnsiColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def colorize(text: str, color_code: str) -> str:
    return f"{color_code}{text}{AnsiColors.RESET}"


def strip_ansi(text: object) -> object:
    if isinstance(text, str):
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)
    return text


# ----------------------------------------------------------------------------
# Formatters
# ----------------------------------------------------------------------------

def format_currency(value: float) -> str:
    return f"${value:,.2f}" if pd.notna(value) else "N/A"


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value == np.inf:
        return "+Inf%"
    if value == -np.inf:
        return "-Inf%"
    return f"{value * 100:.2f}%"


def format_ratio(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    if value == np.inf:
        return "+Inf"
    if value == -np.inf:
        return "-Inf"
    return f"{value:.{decimals}f}"


def currency_formatter(x: float, pos: int) -> str:
    return f"${x:,.0f}"


def percentage_formatter(x: float, pos: int) -> str:
    return f"{x:.1f}%"


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------

def add_watermark(fig: plt.Figure, text: str) -> None:
    fig.text(0.99, 0.01, text, fontsize=8, color="gray", ha="right", va="bottom", alpha=0.7)


def setup_common_axis_elements(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    y_formatter: Optional[FuncFormatter] = None,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_formatter:
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")


def setup_date_axis(
    ax: plt.Axes,
    major_locator_base: int = 1,
    minor_locator_interval: Optional[int] = None,
    minor_format: Optional[str] = None,
) -> None:
    ax.xaxis.set_major_locator(YearLocator(base=major_locator_base))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))

    if minor_locator_interval:
        ax.xaxis.set_minor_locator(MonthLocator(interval=minor_locator_interval))
        if minor_format == "null" or minor_format is None:
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_formatter(DateFormatter(minor_format))
            ax.tick_params(axis="x", which="minor", labelrotation=45)
    else:
        ax.xaxis.set_minor_locator(plt.NullLocator())

    plt.setp(ax.get_xticklabels(which="major"), rotation=0, ha="center")


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------

def check_plot_data(series: pd.Series) -> bool:
    return series is not None and not series.empty and series.nunique() > 1


def compute_drawdown_monetary(equity_series: pd.Series) -> pd.Series:
    if equity_series is None or equity_series.empty:
        return pd.Series(dtype=float)
    running_max = equity_series.cummax()
    dd = equity_series - running_max
    return dd.clip(upper=0)


def calculate_drawdown_durations(equity_series: pd.Series) -> list[int]:
    if not check_plot_data(equity_series):
        return []
    dd_series = compute_drawdown_monetary(equity_series)
    durations = []
    cur = 0
    in_dd = False
    for v in dd_series:
        if v < -1e-9:
            cur += 1
            in_dd = True
        else:
            if in_dd and cur > 0:
                durations.append(cur)
            cur = 0
            in_dd = False
    if in_dd and cur > 0:
        durations.append(cur)
    return durations
