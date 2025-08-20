# Boosted Covered Call — Studio Mensile (BTC-USD.CC di default)

Dashboard Streamlit per presentare la strategia **BTD + Covered Call mensile (boosted)** con tre varianti:
- **BTD No Premi**
- **BTD + Premi (Cash)**
- **BTD + Premi (Reinvest)**

La logica replica fedelmente il notebook originale, **senza** salvataggi CSV/HTML: l’app genera solo le **figure** richieste.

---

## 🚀 Avvio rapido (Streamlit Cloud)

1. **Fork/Import** di questo repo nel tuo GitHub.
2. Su **Streamlit Cloud** → *New app* → collega questo repo e seleziona `app.py`.
3. In **Settings → Secrets** incolla:
   ```toml
   EODHD_API_KEY = "la-tua-api-key"
