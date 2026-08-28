# Boosted Covered Call — Studio Mensile e Settimanale

Dashboard Streamlit per il backtest della strategia **Boosted Covered Call**: un capitale
fisso deciso a inizio anno, coperto da una call venduta a delta 0.50, con acquisti
Buy-The-Dip potenziati dopo ogni periodo negativo e liquidazione totale a fine anno.

Uno **switch in cima alla sidebar** sceglie il passo della strategia — **Mensile** (il
default, la versione originale) oppure **Settimanale** — e tutto il resto lo segue.

Le decisioni stanno sulla griglia del periodo, ma il conto e' **valorizzato ogni giorno di
borsa**: e' li' che si vedono i drawdown veri, quelli che una valorizzazione a fine barra
nasconde (vedi *[Il conto valorizzato ogni giorno](#il-conto-valorizzato-ogni-giorno-non-a-fine-barra)*).

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
arrivano a valori che non sono confrontabili con una strategia che chiude ogni anno.
Su BTC dal 2010 il buy and hold che non liquida arriva a **108 miliardi** contro i 7,3
milioni del ciclo annuale: quando succede viene escluso dai grafici a scala lineare, con
una nota che ne riporta il valore, e resta visibile in scala logaritmica.

Il benchmark a parita' di mandato gira **dentro lo stesso motore** delle varianti, quindi
ha le stesse colonne, le stesse metriche e la stessa tabella annuale, e compare accanto
alle altre curve in ogni grafico: equity, utile netto, drawdown, equity e drawdown della
singola variante, rischio e rendimento, durata degli episodi di drawdown, rendimenti
annuali, distribuzione dei rendimenti, tabella delle metriche e dettaglio anno per anno.

---

## Le due cadenze

Il passo elementare del backtest e' una barra: un mese di calendario oppure una settimana.
Cambia quello, e basta.

| | Mensile | Settimanale |
|---|---|---|
| Barre in un anno | 12 | ~52 |
| Scadenza della call venduta | fine mese | fine settimana |
| Vita dell'opzione usata nel prezzo | i giorni del mese | 7 giorni |
| Segnale Buy-The-Dip | il mese precedente ha chiuso in negativo | la settimana precedente ha chiuso in negativo |
| BTD Boost | su ogni acquisto | su ogni acquisto, quindi molte volte di piu' |
| Interessi su cassa e debito | tasso annuo / 12 | tasso annuo / 52 |
| **Ciclo annuale** | **identico** | **identico** |

Quello che **non** cambia e' tutto il resto: il capitale si decide a gennaio con le stesse
regole (fisso o crescente), le quote coperte restano invariate per l'anno, il capitale
addizionale entra una volta sola, si liquida a dicembre e si ricomincia. Anche il modo di
misurare e' lo stesso: il rendimento resta quello semplice **annuo** sul capitale
investito, e i drawdown restano time-weighted sulla singola barra.

**Cosa aspettarsi dalla settimanale.** La call dura sette giorni invece di trenta, quindi
il singolo premio vale circa la meta' di quello mensile — il valore temporale cresce con
la radice del tempo, non con il tempo — ma se ne incassano 52 invece di 12: l'incasso
lordo dell'anno **circa raddoppia**. Cresce pero' anche il numero di volte in cui la call
finisce in-the-money e va riacquistata a intrinseco, e soprattutto esplode il numero di
Buy-The-Dip: su otto anni di test sintetici, 178 acquisti contro 40. E' la versione
aggressiva: piu' premio incassato, ma anche molto piu' capitale da tirare fuori per i cali
e piu' occasioni di farsi tagliare il rialzo. Con il **boost** conviene tenerne conto —
lo stesso 10% che sul mensile vale al massimo 1,2 volte il capitale iniziale in un anno,
sul settimanale ne vale fino a 5,2.

**Dati.** La cadenza settimanale usa le barre settimanali di EODHD, che vengono comunque
gia' scaricate per il filtro sul drawdown. La volatilita' continua a stimarsi sui dati
**giornalieri**: le finestre della sidebar restano in giorni di borsa e non vanno toccate
quando si cambia passo. Il taglio anti-look-ahead si sposta di conseguenza — il premio
della settimana usa la volatilita' nota **prima del lunedi**, mai quella della settimana
in corso.

**Le parole.** I nomi delle colonne restano quelli storici (`rendimento_mese`, `twr_mese`,
`versamento_mese`) anche in cadenza settimanale, cosi' export e file salvati in passato
restano leggibili: vanno letti come "del periodo". A schermo invece i testi si adattano, e
il campo `parametri.cadenza` dell'export dice sempre con che passo ha girato il backtest.

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

**Nessun tetto agli acquisti.** Di default il cumulato degli acquisti BTD di un anno non
ha limite: ogni segnale viene eseguito per intero. Un tetto e' ancora impostabile da
configurazione (`btd_cap_annuo_pct`), ma non e' esposto nella dashboard perche' cambia la
natura della strategia in un modo controintuitivo: il budget si esaurisce sui cali
superficiali di inizio anno e lascia scoperti quelli profondi che arrivano dopo. Su BTC nel
2022, con un tetto al 100%, boost 0% e boost 15% investivano gli stessi 35.000 ma il secondo
finiva il budget a maggio e saltava il crollo del −37,8% di giugno, pagando le quote il 36%
in piu'. Senza tetto il boost si comporta come deve, in modo proporzionale:

| Boost | BTD investito | Quote | Acquisti | Utile netto |
|---|---|---|---|---|
| 0% | 140.508 | 5,52 | 35 | 101.726 |
| 5% | 201.758 | 7,72 | 35 | 122.900 |
| 15% | 324.258 | 12,11 | 35 | 165.249 |
| 30% | 508.008 | 18,70 | 35 | 228.832 |

Quando un tetto e' impostato, la dashboard lo segnala: i segnali saltati sono marcati nel
grafico dei Buy-The-Dip, le metriche riportano quanto e' stato tagliato e qual e' il calo
piu' profondo rimasto scoperto, e compare un avviso.

**Contabilita'.** Ogni euro entrato dall'esterno finisce in `versamenti_cum`, cosi'
`pnl_netto = valore_portafoglio − versamenti_cum` e' il risultato vero. Rendimenti,
il rendimento e' quello semplice annuo sul capitale investito, mentre drawdown e VaR sono
calcolati sul **rendimento time-weighted** mensile, che neutralizza i flussi di cassa.

### Il capitale di gennaio: fisso o crescente

E' la scelta che decide il risultato piu' di ogni altra, e va fatta consapevolmente.

Con **capitale fisso** ogni gennaio si rimette al lavoro sempre lo stesso importo e i
profitti restano in cassa. La strategia non compone: guadagna ogni anno piu' o meno la
stessa cifra in valuta, la curva cresce in **linea retta** invece che esponenzialmente, e il
rendimento si diluisce anno dopo anno perche' il conto cresce mentre il capitale al lavoro
no. Su S&P 500 dal 2000 con 75.000 impiegati: il capitale di gennaio passa da 75.000 a
78.763 in ventisei anni (**+5%**) mentre il conto cresce del **+327%**.

Con **capitale crescente** gli utili tornano al lavoro ogni gennaio, mantenendo la
proporzione fra parte coperta e parte scoperta. Una quota resta liquida per finanziare gli
acquisti sui cali durante l'anno: senza quella riserva ogni Buy-The-Dip richiederebbe denaro
fresco e il capitale versato crescerebbe insieme al conto.

Stessi identici parametri, su S&P 500 dal 2000:

| | Capitale fisso | Capitale crescente |
|---|---|---|
| Rendimento medio annuo | 2,90% | **4,25%** |
| Capitale versato | 130.684 | **75.000** (solo quello iniziale) |
| Max drawdown | −40,2% | −38,2% |
| Rendimento / oscillazione | 0,32 | **0,44** |
| Quota del conto investita | 59% | 69% |

Il capitale crescente **si autofinanzia**: non serve un euro oltre il capitale iniziale.

### Audit: da dove viene il risultato

Ogni mese il P&L viene ricostruito dalle sue componenti elementari, e la somma coincide al
centesimo con la variazione del conto. Su S&P 500 dal 2000, capitale coperto 25.000 piu'
50.000 non coperti, boost 15%:

| Voce | |
|---|---|
| Movimento delle quote | **+180.131** |
| Premi incassati | +132.484 |
| Intrinseco pagato sulle call | **−139.077** |
| Interessi su cassa e debito | −100 |
| **Utile netto** | **178.018** |

**Le opzioni nettano −6.593 in ventisei anni**: i premi incassati sono quasi esattamente
pari all'intrinseco pagato. Non e' un difetto del modello. I 312 prezzi reali di call ATM
mensili su SPX dello stesso periodo dicono la stessa cosa, anzi peggio: 1.225.538 di premi
contro 1.486.082 pagati alla scadenza, cioe' **−260.544**, pari a −0,29% del sottostante al
mese. Il modello e' se mai leggermente ottimista, perche' lo strike a delta 0.50 sta un
filo sopra lo spot mentre le operazioni reali usavano lo strike quotato piu' vicino.

Il risultato reale, per periodo:

| | 2000-2004 | 2005-2009 | 2010-2014 | 2015-2019 | 2020-2024 |
|---|---|---|---|---|---|
| Netto della call venduta | **+23,8%** | +4,8% | −39,6% | −26,8% | **−42,8%** |

Vendere call vicino al denaro funzionava nel decennio laterale e costa caro da quando il
mercato sale con decisione. La dashboard mostra questa decomposizione nella scheda *Opzione
e premio*.

Il salto annuale, invece, non c'entra: liquidare a dicembre e ricomprare all'apertura di
gennaio e' costato in tutto 1.387 su ventisei passaggi d'anno, una cinquantina di dollari
per volta.

### Le tre leve che cambiano il verdetto

Misurate sullo stesso backtest:

| Leva | Effetto |
|---|---|
| **Capitale crescente** | rendimento medio da 2,90% a 4,25%, versamenti da 130.684 a 75.000, rendimento/oscillazione da 0,32 a 0,44 |
| **Delta della call** | a 0.50 le opzioni tolgono 12.366; a 0.30 ne aggiungono 1.030, a 0.20 ne aggiungono 3.281, e il drawdown resta comunque sotto quello del solo sottostante |
| **Remunerazione della cassa** | da 0% a 4% l'utile passa da 172.268 a 295.040 e il rendimento medio da 2,93% a 4,29% |
| **Dividendi** | `GSPC.INDX` e' un indice di prezzo e li esclude, circa due punti l'anno. Su `SPY.US` il motore usa i prezzi rettificati e ci sono |

Combinando delta 0.30 e cassa al 4%: rendimento medio 4,39% contro 4,14% del solo
sottostante, drawdown
−41,7% contro −46,3%, rendimento/oscillazione 0,50 contro 0,43. L'edge esiste ma sta quasi tutto nel rischio,
non nel rendimento.

### Come si misura il rendimento

La strategia liquida tutto a dicembre e riparte a gennaio: ogni anno e' un ciclo chiuso a
se' stante. Il numero naturale non e' un tasso composto, ma il **rendimento semplice di
ogni anno sul capitale davvero investito in quel ciclo**:

    rendimento dell'anno = risultato dell'anno / (capitale di gennaio + acquisti sui cali)

Al denominatore c'e' solo il denaro che esce dalle tasche di chi investe. I **premi
reinvestiti non entrano**, perche' arrivano dal mercato e non sono capitale conferito.

Un tasso composto sul conto intero direbbe tutt'altro, e sarebbe fuorviante: il conto
include la cassa ferma, che dopo qualche anno puo' essere meta' del totale e diluisce
qualunque percentuale. Su S&P 500 dal 2000 la stessa simulazione da' **7,8% di rendimento
medio annuo** contro un tasso composto sul conto del 2,9%.

Le metriche di rendimento sono quindi: rendimento medio e mediano, oscillazione dei
rendimenti annuali, rapporto fra i due (che sostituisce lo Sharpe), rendimento diviso max
drawdown, anni chiusi in utile, miglior e peggior anno. Il **rischio** resta misurato sul
rendimento time-weighted mensile, che neutralizza i flussi di cassa.

### I premi restano in cassa

Nella variante **Premi (Cash)** i premi incassati finiscono in un conto separato che **non
finanzia gli acquisti sui cali**: quelli si pagano con capitale proprio. Nel campo
`cassa_opzioni` si vede in ogni momento quanto della liquidita' viene dalle opzioni. A fine
anno, con la liquidazione, i due conti si riuniscono.

Il risultato non cambia, cambia la lettura: senza la separazione i premi riducevano i
versamenti necessari e non si capiva piu' quanto capitale la strategia richiedesse davvero.

Lo stesso vale per la variante **Reinvest**: anche li' i premi stanno in un conto separato.
Prima finivano nella cassa generale e riducevano di nascosto il capitale versato per i
Buy-The-Dip — la stessa cosa che si era voluta togliere alla variante Cash.

### Quando i premi tornano al lavoro

Solo per la variante **Reinvest**, e scelto da un selettore nella sidebar sotto *Buy-The-Dip*:

| | Cosa succede |
|---|---|
| **Subito** (default) | il risultato netto delle opzioni compra quote alla chiusura di ogni periodo, al prezzo che c'e' in quel momento |
| **Al prossimo acquisto sui cali** | restano fermi in un conto a parte e rientrano tutti insieme, **al lordo**, quando scatta un Buy-The-Dip, **allo stesso prezzo del BTD** |

L'idea del secondo modo: i premi si accumulano mentre il mercato sale e rientrano su un
ribasso, invece che al prezzo corrente qualunque esso sia. L'esempio, in cadenza mensile:
si incassa il premio di gennaio, il mercato sale, si incassa quello di febbraio; febbraio
chiude in negativo, quindi a marzo scatta il BTD, e in quel momento entrano nel mercato il
premio di gennaio, quello di febbraio **e anche quello di marzo**, al prezzo di apertura di
marzo. Verificato al centesimo in un test costruito apposta.

**Il premio entra lordo.** Anche quello del periodo in corso, incassato all'apertura poche
ore prima del BTD. E' quello che succede su un conto vero: quando vendi la call il premio
e' cassa disponibile subito, e se il ribasso arriva prima della scadenza lo spendi senza
aspettare di sapere quanto ti costera' il riacquisto. L'intrinseco si paga dopo, a
scadenza, e si scala dal conto delle opzioni — che puo' andare a debito, e in quel caso
paga interessi come qualunque altro saldo negativo, al tasso impostato nella sidebar. Il
vantaggio e' che nel frattempo quel premio ha comprato quote e ha lavorato; il costo e' che
si e' speso denaro prima di sapere quanto ne sarebbe rimasto. La metrica *Saldo piu basso
toccato dal conto delle opzioni* dice quanto in profondita' si e' arrivati.

L'unico limite e' la cassa vera: se le scadenze precedenti hanno gia' prosciugato il conto
delle opzioni non si compra a debito, si versa quello che c'e'.

Altre due cose da sapere:

- **Si rientra solo se il BTD e' avvenuto davvero.** Un segnale bloccato dal filtro sul
  drawdown, o azzerato dal tetto annuo, non apre la porta: non c'e' acquisto, non c'e'
  rientro.
- **Il salvadanaio si azzera a gennaio.** I premi incassati a novembre e dicembre senza
  piu' un calo davanti restano in cassa e vengono liquidati con tutto il resto, come nella
  variante Cash. La metrica *Premi liquidati a dicembre senza essere reinvestiti* dice
  quanto e' successo.

**Non e' automaticamente meglio.** Comprare su un ribasso e' un vantaggio, ma aspettare in
un mercato che sale e' uno svantaggio, e quale dei due prevalga dipende da quanto spesso
scattano i BTD. Nei test sintetici a otto anni, in cadenza mensile i premi rientrano a un
prezzo medio **0,5% piu' basso** e il conto finisce piu' in alto; in cadenza settimanale i
BTD sono cosi' frequenti che l'attesa media e' di due centesimi di settimana, lo sconto non
fa in tempo a maturare e si finisce per pagare **poco piu' in alto**. In entrambi i casi il
drawdown vero peggiora di qualche decimo, perche' si e' investito prima e piu': e' la
stessa aggressivita' che dovrebbe pagare. Il selettore serve a misurarlo sul proprio
sottostante, non a dare per scontato il verdetto.

La scheda *Opzione e premio* ha un grafico dedicato — il salvadanaio, i rientri e le due
curve cumulate di quanto le opzioni hanno prodotto e quanto e' tornato al lavoro — e le
metriche riportano attesa media, quota rientrata e premi mai reinvestiti.

### Il vero freno: meta' del conto sta ferma

Il reset annuale reimpiega solo il capitale fisso e lascia in cassa tutti i profitti
accumulati. Dopo qualche anno la cassa diventa la parte piu' grossa del conto: su S&P 500
dal 2000, con 100.000 di capitale fisso, a fine periodo il **62,5%** del conto e' liquidita'
e sul periodo intero la media e' del **48%**.

Questo ha due conseguenze che vanno tenute a mente leggendo i risultati.

**Il rendimento va misurato sul capitale investito.** Un tasso calcolato sul conto intero
darebbe il 2,51%, ma meta' di quel conto non lavora. La dashboard usa il rendimento medio
annuo sul capitale davvero impiegato, e riporta accanto la *Quota media del conto
investita* perche' si veda quanta liquidita' resta ferma.

**La remunerazione della cassa conta piu' di quanto sembri.** Il default e' 0%, cioe' liquidita'
ferma sul conto che non rende nulla per ventisei anni. Con un tasso realistico il quadro
cambia in modo sostanziale:

| Cassa remunerata al | Valore finale | Utile netto | Rendimento | Rend./oscill. |
|---|---|---|---|---|
| 0% (default) | 518.546 | 267.625 | 2,51% | 0,34 |
| 2% | 628.771 | 380.044 | 3,31% | 0,46 |
| 4% | 789.357 | **542.868** | **4,24%** | 0,62 |

Passando da 0% a 4% l'utile piu' che raddoppia. E' lo stesso motivo per cui il buy and hold
che non liquida mai sembra irraggiungibile: quello tiene il 100% investito e capitalizza
senza interruzioni, la strategia ne tiene circa la meta'. La dashboard avvisa quando la quota
investita scende sotto l'80%.

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

### Due grandezze diverse che si chiamano entrambe "capitale"

| | Si azzera a gennaio? | A cosa serve |
|---|---|---|
| **Capitale impiegato nell'anno** | **si** | quanto la strategia sta facendo lavorare adesso: capitale fisso piu' i Buy-The-Dip accumulati nel ciclo in corso |
| **Capitale versato (cumulato)** | **no, mai** | quanto denaro e' entrato dall'esterno da quando e' partito il backtest |

Il primo segue il reset annuale come tutto il resto: su S&P 500 riparte da 77.000 ogni
gennaio, cresce durante l'anno con gli acquisti sui cali fino a un massimo di 153.338, e a
gennaio successivo torna a 77.000. Nei grafici e' la linea a gradini.

Il secondo **non deve** azzerarsi, ed e' importante capire perche': e' il metro con cui si
misura l'utile, visto che `pnl_netto = valore del conto − capitale versato`. Azzerandolo a
ogni gennaio, il denaro versato negli anni precedenti ricomparirebbe come guadagno — che e'
esattamente il difetto della versione originale della dashboard, dove i versamenti dei
Buy-The-Dip finivano contati come profitto.

Nei grafici le due curve ci sono entrambe, con nomi distinti, e nella tabella mensile
dell'anno in corso c'e' la colonna *Capitale impiegato*.

### Perche' il capitale versato si appiattisce

La curva del capitale versato cresce **solo quando la strategia chiede denaro**: al primo
impiego, quando a gennaio il conto non arriva al capitale fisso, e quando un acquisto sui
cali non trova liquidita'. Appena i profitti bastano a coprire entrambe le cose, la linea
diventa orizzontale e resta li' per sempre.

Su S&P 500 dal 2000, con 77.000 di capitale fisso, i versamenti sono stati:

| Anno | Versato | Cumulato |
|---|---|---|
| 2000 | 89.372 | 89.372 |
| 2001 | 10.289 | 99.661 |
| 2002 | 4.400 | 104.062 |
| 2003 | 875 | **104.937** |
| dal 2004 in poi | 0 | 104.937 |

Dieci mesi su 320 hanno richiesto denaro, tutti nel crollo delle dot-com. Da ottobre 2003
la strategia si autofinanzia: il valore del conto non e' mai piu' sceso sotto i 77.000 che
servono a gennaio, con un minimo di 96.546 nel 2008. La linea piatta a 104.937 contro un
conto arrivato a 268.000 dice esattamente questo: di quei 268.000, poco piu' di 100.000
sono soldi tuoi e il resto e' utile. Nei grafici il punto dell'ultimo versamento e'
annotato, cosi' la linea orizzontale non si scambia per un errore.

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

### Il conto valorizzato ogni giorno, non a fine barra

Il motore **opera** sulla griglia del periodo — una decisione al mese, o alla settimana —
ma il conto viene **valorizzato ogni giorno di borsa**. Sono due cose diverse, e tenerle
separate e' quello che rende onesto il numero del rischio.

Se si guarda il conto solo alla chiusura della barra, tutto quello che succede in mezzo
sparisce. Un crollo del 30% a meta' mese che rientra entro il 31 non lascia traccia: la
curva mensile passa da un massimo all'altro e il drawdown misurato resta zero. Non e' un
caso di scuola — e' quello che succede, in piccolo, in ogni backtest valorizzato a fine
barra, e va sempre nella stessa direzione: **il drawdown misurato cosi' e' sistematicamente
piu' tenero di quello vero**.

Il caso limite, costruito apposta come test: un sottostante che chiude ogni mese esattamente
a 100 ma a meta' mese scende a 65 e risale.

| | Drawdown misurato |
|---|---|
| Valorizzando a fine mese | **0,00%** su cinque anni |
| Valorizzando ogni giorno | **−35,0%** |

Su un percorso realistico la differenza e' meno teatrale ma tutt'altro che trascurabile: su
otto anni al 55% di volatilita', −15,4% a fine mese contro **−28,7%** valorizzando ogni
giorno, quasi il doppio. E cambia anche il verdetto: la riduzione del drawdown rispetto al
solo sottostante scende dal 19,3% al **15,6%**, perche' prima si confrontavano due numeri
entrambi sottostimati, e non nello stesso modo.

**Come si costruisce.** Dentro il periodo la posizione e' nota: le quote coperte si comprano
all'apertura e non si muovono, gli acquisti sui cali entrano all'apertura, i premi si
incassano all'apertura, l'intrinseco si paga e i premi si reinvestono alla chiusura. Fra un
estremo e l'altro cambia solo il prezzo, quindi ogni giorno:

    valore = quote x prezzo + liquidita' - valore della call venduta

La call venduta e' un **debito finche' non scade**, e si segna a mercato con Black-Scholes
sullo stesso strike e sulla stessa volatilita' implicita con cui il motore ne ha incassato
il premio, con il tempo residuo che si consuma giorno dopo giorno. All'ultimo giorno il
tempo residuo e' zero, la formula restituisce il valore intrinseco e la serie giornaliera
**ritorna esattamente al valore di fine periodo** calcolato dal motore. E' la verifica che
tiene onesto tutto il resto: nei test lo scarto massimo e' dell'ordine di 10^-10 dollari su
conti da centomila, su entrambe le cadenze, con e senza cap, con il capitale fisso e con
quello composto.

**Il peggio visto in giornata.** Oltre alla chiusura si valorizza anche il **minimo** di
ogni giornata, con lo stesso metodo. E' un'approssimazione dichiarata — nessuno sa in che
ordine il prezzo abbia toccato i suoi estremi — ma dice quanto in basso e' arrivato il conto
se in quel momento il prezzo fosse stato il minimo. Nella tabella e' la riga *Peggio visto
in giornata*.

**Quali numeri guardare.** `max_dd_giornaliero_pct` e' il drawdown vero ed e' quello che la
dashboard mostra nelle KPI, nel pannello del drawdown di ogni variante, nel confronto
underwater e nel verdetto contro il benchmark. `max_dd_pct` resta in tabella, etichettato
*alla chiusura di periodo*, e `dd_nascosto_dal_periodo` dice quanti punti passavano
inosservati. Un grafico dedicato mette le due curve una sopra l'altra, e un secondo grafico
confronta le tre frequenze di valorizzazione curva per curva.

**Senza dati giornalieri** — ticker che EODHD non copre al giorno — la dashboard non
inventa: resta alla valorizzazione di fine periodo e lo dice con un avviso in cima e una
nota nella scheda del rischio.

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
  cadenza.py              mensile o settimanale: periodi per anno e vocabolario dei testi
  giornaliero.py          rivalutazione giorno per giorno, call venduta segnata a mercato
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

- Nella variante Reinvest si puo' scegliere **quando** i premi tornano al lavoro: subito
  alla chiusura del periodo, oppure fermi in un salvadanaio finche' non scatta un acquisto
  sui cali, per rientrare **al lordo** e allo stesso prezzo del BTD, con l'intrinseco pagato
  poi a scadenza. E in entrambi i modi i premi non
  finanziano piu' di nascosto i Buy-The-Dip.
- Il conto e' **valorizzato ogni giorno di borsa**, non solo alla chiusura della barra.
  Prima un crollo rientrato prima di fine mese non compariva da nessuna parte e il drawdown
  misurato era sistematicamente piu' tenero del vero: su otto anni al 55% di volatilita',
  −15,4% invece di −28,7%. La call venduta viene segnata a mercato con Black-Scholes finche'
  non scade, e a ogni fine periodo le due serie coincidono al centesimo.
- Esiste la **cadenza settimanale**, scelta con lo switch in cima alla sidebar: stessa
  strategia con passo di sette giorni invece di trenta. Piu' premi incassati e molti piu'
  acquisti sui cali; il ciclo annuale non cambia di una virgola.
- Il cap della covered call **si accumula**. Prima il valore del pacchetto veniva
  ricalcolato da zero ogni mese partendo dall'apertura, e il guadagno tagliato dalla call
  tornava indietro il mese successivo: su un sottostante che saliva del 5% al mese per due
  anni il modello restituiva +192% dove una covered call reale sarebbe rimasta a zero.
- **I versamenti non sono piu' contati come utili.** Prima il capitale investito nei
  Buy-The-Dip entrava nell'equity e a fine anno veniva registrato come profitto: su percorsi
  in cui il sottostante perdeva oltre il 90% la curva mostrava comunque centinaia di
  migliaia di dollari di "guadagno", quasi interamente costituiti dai soldi versati.
- Il **Buy & Hold e' confrontabile**: riceve gli stessi versamenti negli stessi mesi.
- Il **rendimento e' quello semplice annuo sul capitale investito**, non un tasso composto
  sul conto: il conto include la cassa ferma e diluiva ogni percentuale. Drawdown e VaR
  restano time-weighted, quindi non distorti dai flussi di cassa.
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
