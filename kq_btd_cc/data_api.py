from __future__ import annotations
import pandas as pd
import requests

# Cache Streamlit se disponibile (niente side-effect in ambienti non-Streamlit)
try:
    import streamlit as st
    _cache = st.cache_data(show_spinner=False)
except Exception:
    def _cache(func):
        return func

@_cache
def _download_and_prepare_eodhd_data(ticker: str, start_date: str, end_date: str,
                                     time_period: str, api_key: str) -> pd.DataFrame:
    """
    [Portata dal notebook] Scarica e prepara i dati storici (OHLCV) da EODHD.
    Logica originale preservata.
    """
    # --- INIZIO: codice originale invariato ---
    # (La funzione completa è stata portata così com'è dal notebook.)
    import json
    import datetime as dt
    from pandas import to_datetime

    print(f"Tentativo di download dati EODHD (periodo: {time_period}) per {ticker} da {start_date} a {end_date}...")
    api_url = f"https://eodhd.com/api/eod/{ticker}"
    params = {
        'api_token': api_key, 'from': start_date, 'to': end_date,
        'period': time_period, 'fmt': 'json', 'order': 'a'  # Ascendente
    }
    df_result = pd.DataFrame()

    try:
        response = requests.get(api_url, params=params, timeout=45)
        response.raise_for_status()
        json_data = response.json()

        if not json_data or not isinstance(json_data, list) or len(json_data) == 0:
            print("  -> Nessun dato valido scaricato da EODHD.")
            return df_result

        df = pd.DataFrame(json_data)
        # Normalizzazione colonne e conversioni come nel notebook
        rename_map = {
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'adjusted_close': 'adj_close', 'volume': 'volume', 'date': 'date'
        }
        df = df.rename(columns=rename_map)
        df['date'] = to_datetime(df['date']).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)

        # Aggiustamento OHLC in base ad adjusted_close
        if 'adj_close' in df.columns and 'close' in df.columns:
            ratio = df['adj_close'] / df['close']
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = df[col] * ratio

        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.set_index('date')

        df_result = df
    except requests.HTTPError as e:
        print(f"[EODHD] HTTPError: {e}")
    except requests.RequestException as e:
        print(f"[EODHD] RequestException: {e}")
    except Exception as e:
        print(f"[EODHD] Errore generico: {e}")

    return df_result
    # --- FINE: codice originale invariato ---
