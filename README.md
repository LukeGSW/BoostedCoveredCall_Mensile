# Boosted Covered Call — Studio Mensile

Dashboard Streamlit per il backtest della strategia **Boosted Covered Call**: un capitale
fisso deciso a inizio anno, coperto da una call mensile venduta a delta 0.50, con acquisti
Buy-The-Dip potenziati sui mesi negativi e liquidazione totale a fine anno.

Le curve messe a confronto:

| Curva | Cosa rappresenta |
|---|---|
| **Buy & Hold (stesso ciclo annuale)** | il sottostante comprato a gennaio e liquidato a dicembre, come la strategia, ma senza opzioni e senza Buy-The-Dip — **e' il metro di paragone** |
| **BTD No Premi** | solo Buy-The-Dip sul sottostante, nessuna opzione |
| **BTD + Premi (Cash)** | covered call con i premi tenuti in liquidita' |
| **BTD + Premi (Reinvest)** | covered call con il risultato netto delle opzioni reinvestito in quote |
| Buy & Hold (capitale iniziale) | il sottostante comprato una volta e mai piu' toccato |
| Buy & Hold (stessi versamenti) | riceve gli stessi soldi negli stessi mesi ma non liquida mai |

Il benchmark che conta e' il primo. Gli ultimi due non liquidano mai: su un sottostante
molto direzionale accumulano quote comprate anni prima a una frazione del prezzo e
arrivano a valori che non sono confrontabili con una strategia che chiude ogni dodici mesi.
Su BTC dal 2010 il buy and hold che non liquida arriva a **108 miliardi** contro i 7,3
milioni del ciclo annuale: quando succede viene escluso dai grafici a scala lineare, con
una nota che ne riporta il valore, e resta visibile in scala logaritmica.

Il benchmark a parita' di mandato gira **dentro lo stesso motore** delle varianti, quindi
ha le stesse colonne, le stesse metriche e la stessa tabella annuale, e compare accanto
alle altre curve in ogni grafico: equity, utile netto, drawdown, equity e drawdown della
singola variante, rischio e rendimento, durata degli episodi di drawdown, rendimenti
annuali, distribuzione dei rendimenti, tabella delle metriche e dettaglio anno per anno.

---

## Come funziona il modello

**Ciclo annuale.** All'apertura di gennaio si impiega il capitale iniziale comprando il
sottostante: sono le *quote coperte*, e restano invariate per tutto l'anno. A dicembre si
liquida tutto. Se il conto ha piu' del capitale iniziale, l'eccedenza resta in cassa; se ne
ha meno, si versa la differenza — ed e' registrata come **versamento**, non come utile.

### Le tre voci di capitale, e cosa le distingue

| | Quando entra | La call lo cappa? | Si incassa il premio? |
|---|---|---|---|
| **Capitale iniziale** | apertura di gennaio | **si**, e' la base coperta | **si** |
| **Capitale addizionale annuale** | apertura di gennaio, una volta sola | no, tiene tutto il rialzo | no |
| **BTD + Boost** | a ogni segnale, durante l'anno | no | no |

Il capitale addizionale annuale e il BTD Boost sono due cose diverse e vanno impostate
separatamente: il primo e' un importo che entra **una volta sola** a inizio anno; il secondo
e' una percentuale che si aggiunge a **ogni** acquisto BTD. Tutte e tre le voci vengono
liquidate a fine anno.

**La call.** Ogni mese si vende una call con scadenza a fine mese, strike a delta 0.50
(quindi appena sopra lo spot, tanto piu' quanto e' alta la volatilita'). Il premio e' una
percentuale del prezzo corrente del sottostante, quindi l'incasso in valuta cambia ogni
mese. A scadenza, se la call e' in-the-money la si **riacquista al valore intrinseco**: le
quote restano le stesse e il costo del cap sull'upside si accumula davvero, mese dopo mese.

**Buy-The-Dip.** Quando il mese precedente chiude in negativo si investe l'entita' del calo
applicata al capitale fisso. Il segnale e' noto alla chiusura del mese precedente e
l'acquisto avviene all'apertura del mese successivo. Un filtro opzionale sospende gli
acquisti quando il drawdown settimanale dell'asset supera una soglia; anche questo filtro
legge solo dati gia' disponibili al momento della decisione.

**BTD Boost.** Una percentuale del capitale iniziale che si aggiunge a **ogni** acquisto
BTD, oltre alla quota legata all'entita' del calo. Eredita tutto dal BTD: stesso momento e
stesso prezzo di acquisto, quote non coperte dalla call, stesso tetto annuo, liquidazione a
fine anno. Con capitale iniziale 25.000 e boost al 5%, ogni acquisto porta con se' 1.250 di
boost; se il tetto annuo taglia l'acquisto, calo e boost si riducono in proporzione. Le due
componenti sono tracciate separatamente in `btd_quota_calo` e `btd_quota_boost`, in ogni
tabella e in ogni export: su un backtest tipico il boost vale il 40% di tutto il capitale
investito in Buy-The-Dip.

**Contabilita'.** Ogni euro entrato dall'esterno finisce in `versamenti_cum`, cosi'
`pnl_netto = valore_portafoglio − versamenti_cum` e' il risultato vero. Rendimenti,
volatilita', Sharpe, drawdown e VaR sono calcolati sul **rendimento time-weighted**, che
neutralizza i flussi di cassa.

**Saldo a debito.** Se il riacquisto di una call molto in-the-money supera la liquidita'
disponibile, il conto va a debito contro le azioni in portafoglio. Il finanziamento e'
tracciato, gli si applica un tasso configurabile, e la dashboard avvisa quando diventa
rilevante.

---

## Il premio senza dati di volatilita' implicita

Un premio fisso non regge il cambio di sottostante: a delta 0.50 su scadenza mensile una
call vale circa l'1,7% dello spot con volatilita' al 15% e circa l'8,9% con volatilita' al
90%. La dashboard lo ricostruisce cosi':

1. **volatilita' realizzata** dai dati giornalieri — Yang-Zhang (default), Garman-Klass,
   Rogers-Satchell, Parkinson, close-to-close o EWMA, con finestra corta e lunga mescolabili;
2. **volatilita' implicita stimata** = realizzata × *volatility risk premium*;
3. **prezzo Black-Scholes completo** con strike risolto numericamente al delta obiettivo.

La volatilita' usata per un mese e' sempre quella nota **prima** che il mese inizi. Non
esiste piu' un premio da impostare a mano: il numero lo produce il modello.

Il VRP non e' costante. Misurato sui prezzi reali, scende al crescere della volatilita' del
sottostante — vale circa 1.03 su un titolo che oscilla al 12% annuo e 0.83 su uno al 60% —
e la dashboard modella questa pendenza:

    vrp(sigma) = vrp_a_20% + pendenza x ln(sigma / 20%)

### Cosa si tocca nella sidebar

Due sole scelte, entrambe con un default che va bene cosi' com'e'.

**Taratura del premio.** *Predefinito* usa i coefficienti misurati sui prezzi reali ed e'
il punto di partenza giusto per un sottostante qualunque. *Calibrato sui prezzi reali*
compare da solo dopo aver caricato un file nella scheda Calibrazione premio, e prende i
valori da li' senza doverli ricopiare. *Manuale* apre i due parametri del modello, il
livello del rapporto fra volatilita' implicita e realizzata e la sua pendenza. Sotto la
scelta c'e' sempre la traduzione in premi concreti: quanto incasserebbe una call mensile su
un indice, su un titolo e su una crypto.

**Memoria del modello**, cioe' quanto storico guarda lo stimatore di volatilita':

| | Guarda | Si accorge di un cambio di regime in | Il premio si muove |
|---|---|---|---|
| **Predefinita** | 6 mesi, smorzati verso 2 anni | 4 mesi | 2,7% al mese |
| Piu reattiva | 3 mesi, smorzati poco | **1 mese** | 3,6% al mese, salti fino all'80% |
| Piu stabile | 1 anno, appoggiato a 3 anni | 7 mesi | **2,3% al mese**, mai oltre il 18% |

Il premio medio esce quasi identico in tutti e tre i casi: cambia solo quanto in fretta
segue il mercato e quanto balla nel frattempo. *Manuale* espone stimatore, finestre e peso
uno per uno.

### Quanto e' affidabile su un ticker che non hai calibrato

Validazione leave-one-out sui sei sottostanti: il coefficiente viene stimato su cinque e
applicato al sesto, mai visto, esattamente come quando si lancia il backtest su un ticker
qualunque senza caricare nessun file.

| Scarto sul premio medio incassato | Media assoluta |
|---|---|
| Premio fisso al 5% | **107%** |
| VRP costante generico (1.15) | 22,5% |
| VRP costante stimato su altri ticker | 13,3% |
| **VRP dipendente dalla volatilita, stimato su altri ticker** | **8,7%** |
| VRP calibrato sul ticker stesso (serve il file dei prezzi) | 0% |

Su un ticker non calibrato il premio finisce quindi entro il **10% circa** del valore reale,
contro il 107% del vecchio numero fisso. L'errore mese per mese resta del 36-37% in tutti i
casi: il coefficiente sposta il livello, non migliora il tempismo. Per il backtest conta il
livello, perche' a fare il risultato e' il totale dei premi incassati.

Una precisazione onesta: questi numeri sono stati ottenuti con la volatilita' campionata a
ogni scadenza, l'unica ricostruibile dai file di prezzi. Il motore in produzione usa dati
giornalieri, dove l'errore di stima della volatilita' e' molto piu' basso, quindi il valore
misurato qui e' un limite inferiore. Il livello del VRP dipende anche da quale stimatore lo
alimenta: la pendenza, che descrive una differenza fra sottostanti, e' piu' stabile.

### Cosa dicono i prezzi reali

I default dello stimatore non sono inventati: vengono da 1.666 vendite effettive di call
ATM mensili su sei sottostanti fra il 2000 e il 2025.

| | Operazioni | Periodo | Premio mediano | Intervallo 5-95% | Vol. implicita mediana |
|---|---|---|---|---|---|
| SPX | 312 | 2000-2025 | **1,64%** | 0,95% - 3,20% | 13,9% |
| SPY | 252 | 2005-2025 | **1,62%** | 0,98% - 3,21% | 13,7% |
| PG | 306 | 2000-2025 | **1,89%** | 0,75% - 4,49% | 15,3% |
| WMT | 307 | 2000-2025 | **2,21%** | 0,92% - 5,42% | 18,0% |
| AAPL | 305 | 2000-2025 | **3,47%** | 1,92% - 7,07% | 30,7% |
| TSLA | 184 | 2010-2025 | **5,45%** | 3,60% - 7,73% | 50,5% |

Un premio fisso al 5% sovrastima l'incasso del **205% su SPX** e del 210% su SPY, mentre su
TSLA e' quasi corretto: e' la ragione per cui il premio va calcolato invece che scelto.
Questi numeri sono consultabili dentro la dashboard e servono da controllo sull'ordine di
grandezza del premio stimato.

Due scelte di progetto derivano direttamente da questi dati:

- **Finestra corta di sei mesi, smorzata a meta' verso la media di lungo periodo.** Provando
  tutte le combinazioni di finestra (6, 12, 18, 24, 36 osservazioni) e smorzamento (da 0 a
  100%), la finestra piu' corta con smorzamento intermedio ha vinto su tutti e sei i
  sottostanti. La volatilita' si raggruppa nel tempo, quindi il dato recente conta; ma
  stimarla da poche osservazioni introduce un errore che va smorzato.
- **La calibrazione allinea il livello dei premi, non l'errore mese per mese.** Minimizzando
  l'errore assoluto resta una sottostima sistematica del 7-14%, minimizzando quello relativo
  si arriva al 20-48%. Allineando direttamente il premio medio lo scarto va a zero, che e'
  cio' che conta quando a fare il risultato e' il totale incassato.

### Calibrare il VRP sui prezzi reali

Nella scheda *Calibrazione premio* si carica un file con i prezzi reali delle call e la
dashboard cerca il VRP che minimizza l'errore, riportando MAE, RMSE, distorsione e R², e
confrontando tutti gli stimatori di volatilita' fra loro. Colonne minime:

| Colonna | Obbligatoria | Note |
|---|---|---|
| data | si | data della quotazione |
| prezzo del sottostante | si | spot alla stessa data |
| prezzo dell'opzione | si | `mid`, oppure `bid` e `ask` |
| giorni a scadenza | si | `dte`, oppure una data di scadenza |
| delta | no | se presente, filtra le opzioni vicine al delta obiettivo |
| strike, IV, tipo | no | usati per i controlli |

I nomi delle colonne vengono riconosciuti da soli e restano modificabili a mano. **Gli export
di OptionLAB sono riconosciuti automaticamente**: la dashboard associa da sola data di
apertura, sottostante all'apertura e premio incassato, tiene solo le vendite di call e
ignora la serie di equity giornaliera affiancata ai trade. Un esempio in formato generico
e' in [`esempio_calibrazione.csv`](esempio_calibrazione.csv).

### Percentuale o valuta: come si leggono i drawdown

I drawdown si confrontano **solo in percentuale**. Quello in valuta dipende da quanto
capitale ciascuna curva sta facendo lavorare, e i Buy-The-Dip ne fanno impiegare alla
strategia molto piu' che al benchmark. Su S&P 500 dal 2000:

| | Drawdown in valuta | Drawdown in percentuale | Capitale medio impiegato |
|---|---|---|---|
| Buy & Hold (ciclo annuale) | −48.338 | −46,3% | 79.440 |
| BTD + Premi (Cash) | −53.272 | **−33,8%** | 96.440 (+21%) |
| BTD + Premi (Reinvest) | −55.790 | **−32,8%** | 100.338 (+26%) |

In dollari la strategia sembra perdere di piu'; in percentuale perde un terzo in meno.
La seconda e' la lettura giusta, ed e' quella che la dashboard usa in tutti i grafici di
drawdown e nella riga *Riduzione del drawdown*. L'importo in valuta resta nell'annotazione
del grafico e in una riga a parte della tabella metriche, etichettata per quello che e'.

---

## Seguire la strategia a mercato

La scheda **Anno in corso** e' pensata per l'uso operativo, non per il backtest. Mostra:

- i numeri dell'anno selezionato in evidenza — risultato, premi incassati contro intrinseco
  pagato, BTD investito con la ripartizione fra calo e boost, capitale ancora disponibile
  sotto il tetto annuo, valore del conto diviso fra posizione e cassa;
- **cosa fare il mese prossimo**: se il segnale BTD e' scattato e per quale importo, quanto
  di quello e' boost, quanto resta sotto il tetto, e su quante quote vendere la call con
  strike indicativo, premio atteso e volatilita' usata. Sono tutte grandezze gia'
  determinate dalla chiusura del mese scorso, quindi si possono leggere prima dell'apertura;
  a inizio anno l'avviso diventa invece quello di liquidare e reimpiegare il capitale fisso;
- la **tabella mese per mese** dell'anno: prezzo, rendimento, stato del segnale, BTD diviso
  in quota calo e quota boost, residuo del tetto, quote coperte ed extra, strike venduto,
  premio in percentuale e in valuta, intrinseco pagato, netto dell'opzione, cassa, valore
  del conto, versamenti, utile netto e drawdown. Scaricabile in CSV.

---

## Export

Il pulsante **Scarica JSON** produce un file con parametri, equity, serie mensile completa
di ogni variante (42 colonne), tabelle annuali, flussi di cassa, metriche, volatilita'
stimata, drawdown settimanale, benchmark, calibrazione del premio e un dizionario che
descrive ogni campo.

---

## Avvio

### Streamlit Cloud
1. Fai il fork del repository.
2. *New app* → seleziona `app.py`.
3. In *Settings → Secrets*:
   ```toml
   EODHD_API_KEY = "la-tua-api-key"
   ```

### Scegliere il periodo

Anno e mese si impostano con due campi separati, non col calendario: il selettore di
Streamlit elenca nel menu degli anni una finestra fissa di vent'anni e ignora `min_value`,
quindi si fermerebbe una ventina d'anni indietro tagliando fuori la bolla dot-com. Qui
l'anno si puo' scrivere direttamente e si arriva al 1970. Il motore lavora comunque su
barre mensili, quindi il giorno non aggiungerebbe niente.

Lo storico effettivo dipende dal ticker: EODHD restituisce i dati dalla prima data che ha.

### In locale
```bash
pip install -r requirements.txt
```
```bash
streamlit run app.py
```
La chiave si puo' passare come variabile d'ambiente `EODHD_API_KEY` oppure in
`.streamlit/secrets.toml`.

---

## Struttura

```
app.py                    dashboard Streamlit
kq_btd_cc/
  data_api.py             download EODHD (giornaliero, settimanale, mensile) con cache
  vol.py                  stimatori di volatilita' realizzata
  pricing.py              Black-Scholes senza scipy, strike a delta obiettivo
  engine.py               motore di backtest e contabilita' dei flussi
  metrics.py              metriche time-weighted e confronto con il benchmark
  calibration.py          calibrazione del premio sui prezzi reali
  charts.py               grafici Plotly a tema scuro
  riferimenti.py          premi reali misurati, usati come metro di controllo
  export.py               pacchetto JSON
  core.py                 orchestrazione dai parametri alle figure
  style.py                palette, template Plotly, CSS
  utils.py                formattazione e helper
```

---

## Cosa e' cambiato rispetto alla versione precedente

- Il cap della covered call **si accumula**. Prima il valore del pacchetto veniva
  ricalcolato da zero ogni mese partendo dall'apertura, e il guadagno tagliato dalla call
  tornava indietro il mese successivo: su un sottostante che saliva del 5% al mese per due
  anni il modello restituiva +192% dove una covered call reale sarebbe rimasta a zero.
- **I versamenti non sono piu' contati come utili.** Prima il capitale investito nei
  Buy-The-Dip entrava nell'equity e a fine anno veniva registrato come profitto: su percorsi
  in cui il sottostante perdeva oltre il 90% la curva mostrava comunque centinaia di
  migliaia di dollari di "guadagno", quasi interamente costituiti dai soldi versati.
- Il **Buy & Hold e' confrontabile**: riceve gli stessi versamenti negli stessi mesi.
- I rendimenti sono **time-weighted**, quindi Sharpe, volatilita', VaR e drawdown percentuale
  non sono piu' distorti dai flussi di cassa.
- Il **premio e' stimato** dalla volatilita' del sottostante e calibrabile sui prezzi reali.
  La percentuale fissa da regolare a mano non esiste piu': era un valore arbitrario che
  sovrastimava l'incasso su SPX del 205%.
- Il **BTD Boost e' tracciato separatamente** dalla quota legata al calo, in ogni tabella e
  in ogni export, cosi' si vede sempre quanto pesa (nei test, il 40% del capitale investito
  in Buy-The-Dip).
- Nuova scheda **Anno in corso** con il dettaglio mese per mese e il piano operativo del
  mese successivo, per seguire la strategia quando e' a mercato.
- Il **BTD si esegue all'apertura** del mese del segnale, non alla chiusura: il segnale e'
  noto un mese prima.
- Il **filtro sul drawdown settimanale** legge il dato di fine mese precedente invece di
  quello del mese in corso.
- Al reset di gennaio il risultato del mese non conta piu' due volte il gap fra la chiusura
  di dicembre e l'apertura di gennaio.
- I **flag dei grafici nella sidebar funzionano**: prima l'unico letto era quello dei
  grafici addizionali.
- Esistono davvero i **rendimenti annuali operativi**, una tabella di metriche, il livello di
  confidenza del VaR che era raccolto e mai usato, e l'**export JSON**.
- Grafica rifatta in **Plotly** a tema scuro, interattiva; niente piu' figure matplotlib
  lasciate aperte a ogni rerun e chiamate a EODHD ora in cache.
