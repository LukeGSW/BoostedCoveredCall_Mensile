STYLE_CONFIG = {
    'figure_figsize': (16, 8),
    'plot_style': 'seaborn-v0_8-whitegrid',
    'palette': 'tab10',
    'colors': {
        'equity_no_prem': '#E67E22',
        'equity_prem_accum': '#F1C40F',
        'equity_prem_reinvest': '#1ABC9C',
        'equity_with_prem': '#2ECC71',  # Non usata attivamente nel Codice 1, ma presente
        'investment': '#3498DB',
        'reinvest': '#5DADE2',         # Usato per barre BTD nel Grafico 5 originale
        'drawdown_asset': '#8E44AD',
        'drawdown_portfolio_usd': '#E74C3C',
        'buy_hold': '#9B59B6',
        'annotation': '#C0392B',
    },
    'font_sizes': {
        'title': 16, 'suptitle': 18, 'axis_label': 14, 'tick_label': 12,
        'legend': 11, 'annotation': 11, 'watermark': 8
    },
    'line_width': {
        'standard': 1.75, 'thin': 1.25, 'thick': 2.0
    },
    'grid_alpha': 0.4,
    'table_format': 'fancy_grid',
    # Il watermark_text verrà aggiornato dinamicamente nella funzione principale (Cella 2)
    # e non è più definito staticamente qui per evitare dipendenze da variabili non ancora note.
}
