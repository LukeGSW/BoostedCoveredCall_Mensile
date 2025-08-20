# kq_btd_cc/core.py
from __future__ import annotations
import typing as _t

# NOTA: la app imposta questa variabile prima di chiamare la funzione
EODHD_API_KEY: str | None = None

def esegui_analisi_completa(
    params_gui: dict,
    plot_prefs: dict,
    export_enabled: bool = False,
    export_base_path: str = "kriterion_export_output",
    export_plot_format: str = "png",
    export_table_format: str = "html",
    export_html_report: bool = False,
    html_report_filename: str = "Kriterion_Report_Completo.html",
    num_columns_html_report: int = 2,
    *,
    # estensione per Streamlit: non chiudere le figure e ritornarle
    streamlit_mode: bool = True,
) -> dict:
    """
    Ponte per la funzione principale.
    In questa versione stub NON esegue la strategia: serve solo a sbloccare l'integrazione con l'app.
    Nella prossima commit sostituiamo il corpo con il porting integrale della logica esatta del notebook,
    rimuovendo definitivamente CSV/HTML e restituendo la lista di figure.
    """
    # Validazioni minime, così l'app fornisce errori chiari
    if not isinstance(params_gui, dict) or not isinstance(plot_prefs, dict):
        raise ValueError("params_gui e plot_prefs devono essere dizionari.")
    if not EODHD_API_KEY:
        raise RuntimeError("EODHD_API_KEY non impostata. La app la passa via st.secrets prima della chiamata.")

    # Struttura di ritorno attesa dall'app
    return {
        "df_simulazione_reinv_export": None,
        "df_simulazione_cash_export": None,
        "df_simulazione_np_export": None,
        "params_simulazione": params_gui,
        "path_report_html": None,
        "figures": [],  # nella versione completa conterrà i matplotlib.figure.Figure in ordine logico
    }

