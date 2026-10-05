# Allocazione adattiva con una catena di Markov sui regimi di volatilità

In questo progetto costruisco un portafoglio virtuale che riduce il rischio quando la volatilità del mercato fa pensare a uno stress imminente. La probabilità di stress viene da una catena di Markov stimata su regimi di volatilità osservabili. Poi la confronto, fuori campione, con regole molto più semplici (vol targeting, esposizione costante, un HMM, una regressione logistica, un filtro di tendenza).

La domanda è: **la catena aggiunge qualcosa rispetto a regole più semplici?** Il README descrive come l'ho verificato, in modo che il risultato si possa controllare anche se la risposta è no.

## Indice

1. [Risultato in breve](#1-risultato-in-breve)
2. [Come ho lavorato](#2-come-ho-lavorato)
3. [Dati](#3-dati)
4. [Il metodo in quattro passi](#4-il-metodo-in-quattro-passi)
5. [Cosa confronto](#5-cosa-confronto)
6. [Risultati sul test](#6-risultati-sul-test)
7. [Limiti](#7-limiti)
8. [Come riprodurre i numeri](#8-come-riprodurre-i-numeri)
9. [Struttura del repository](#9-struttura-del-repository)
10. [Riferimenti](#10-riferimenti)

I dettagli tecnici (scelta dei parametri, misure statistiche, risultati sullo sviluppo, parte previsiva, controlli e limiti completi) sono in [docs/metodo.md](docs/metodo.md). In [docs/domande_e_risposte.md](docs/domande_e_risposte.md) ci sono le risposte brevi alle domande più probabili.

---

## 1. Risultato in breve

Sul periodo di test (gennaio 2015 – giugno 2026, 2.889 giorni di borsa, costi di 5 punti base):

- **Il drawdown è molto più basso di mercato, 60/40 e nucleo statico.** Il drawdown massimo è del 6,6%, contro il 34,2% del mercato, il 20,9% del 60/40 e il 15,3% del nucleo statico (gli stessi strumenti sempre investiti). Contro questi tre confronti l'intervallo di confidenza al 95% sulla differenza è tutto a favore della strategia. Contro il vol targeting (8,3%) e l'esposizione costante (10,2%) la differenza non è distinguibile da zero.
- **Lo Sharpe non migliora.** 0,753 per la strategia, contro 0,697 del mercato, 0,671 del 60/40 e 0,792 del nucleo statico. La stima è più alta di mercato e 60/40, più bassa del nucleo statico; nessuna differenza è distinguibile da zero.
- **La catena di Markov non aggiunge valore dimostrabile rispetto a regole più semplici.** Contro il vol targeting con la stessa esposizione media la differenza di Sharpe è −0,012, con intervallo al 95% [−0,106; +0,082]. Anche catena base, stato corrente, HMM e regressione logistica non si distinguono.
- **Il test ha una potenza limitata.** In una simulazione con la stessa lunghezza, un vantaggio vero di 0,10 di Sharpe viene riconosciuto nel 43% dei casi (circa ±3 punti) e uno di 0,15 nel 78%. L'estremo superiore dell'intervallo è +0,082: un vantaggio sul vol targeting, se esiste, è piccolo, ma non posso escludere che esista.

In sintesi: la riduzione del drawdown viene dal tenere meno rischio quando la volatilità sale, cosa che fa anche il vol targeting. Non vedo un contributo specifico della catena di Markov.

![Crescita del capitale nel test](img/equity_test.png)

## 2. Come ho lavorato

Per non ingannarmi da solo ho diviso il periodo in due parti e fissato le regole prima di guardare i risultati.

- **Sviluppo**, fino al 31 dicembre 2014: qui ho provato le idee e scelto i parametri.
- **Test**, dal 1° gennaio 2015 a giugno 2026: l'analisi principale (`06_inferenza.py test`) l'ho lanciata **una sola volta**, dopo aver congelato le configurazioni in `config_congelata.json`. Lo script si rifiuta di partire se il file non c'è, se non coincide con le scelte fatte sullo sviluppo o se esiste già un esito completo. Parte previsiva, grafici e notebook li ho fatti dopo, sugli stessi dati e senza cambiare nessuna scelta: sono letture aggiuntive, non una seconda occasione.
- Prima del test ho scritto che cosa avrei considerato un risultato positivo ([docs/criteri_del_verdetto.md](docs/criteri_del_verdetto.md); il repository non prova la data di scrittura, quindi è una mia dichiarazione). Livello 1: la strategia è utile? (drawdown più basso e Sharpe non peggiore di mercato, 60/40 e nucleo statico, al netto dei costi). Livello 2: la catena aggiunge valore? Misura unica: differenza di Sharpe contro il vol targeting con la stessa esposizione media, con intervallo al 95%. Evidenza se l'intervallo è tutto favorevole e il segno regge con costi a 10 punti base e nei sottoperiodi 2015-2019 e 2020-2026; indicazione debole se la stima è favorevole ma l'intervallo include lo zero; altrimenti nessun valore aggiunto.
- Esito: **nessun valore aggiunto dimostrabile** (sezione 6). Livello 1: soddisfatto sul drawdown; sullo Sharpe la stima contro il nucleo statico è più bassa (−0,039), quindi non soddisfatto sulla stima puntuale.

La **variante multi-scala** è nata dopo aver visto, nello sviluppo, che la catena semplice non prevedeva la volatilità meglio di EWMA e HAR: non era un'idea precedente ai risultati. Il test però non era ancora stato toccato.

## 3. Dati

Tutti i dati sono gratuiti.

| Serie | Fonte | Note |
|---|---|---|
| Mercato (Mkt) | Kenneth R. French Data Library, fattori giornalieri | Mkt-RF + RF, rendimenti a valore ponderato con dividendi reinvestiti |
| NoDur (beni di consumo non durevoli) | French, 10 portafogli per industria | HiTec viene letta ma serve solo per i controlli sui dati |
| Chips (semiconduttori) | French, 49 portafogli per industria | |
| Liquidità (RF) | French, stesso file dei fattori | tasso giornaliero, equivalente al tasso a 1 mese dei Treasury |
| Obbligazioni (Bond7) | rendimenti a scadenza costante a 7 anni pubblicati dal Dipartimento del Tesoro USA (file annuali CSV dal 2000, già scaricati per un altro mio progetto; `data/treasury/cmt.csv`) | rendimento totale stimato di un titolo alla pari: cedola maturata, durata e convessità; il rullaggio sulla curva è ignorato |
| Oro | GLD, file giornaliero in formato Stooq | rendimento del prezzo (GLD non paga dividendi); le spese di gestione dell'ETF sono già nel prezzo, a differenza degli indici |

Il calendario è quello di French (26.274 giorni dal luglio 1926 al 30 giugno 2026). L'allocazione parte dal 28 febbraio 2005 (primo giorno con tutte le serie, perché c'è l'oro) e le metriche partono dopo 504 giorni di avviamento (1° marzo 2007).

Uso gli indici di French e non gli ETF perché questi ultimi non sono rendimenti totali (nei giorni di stacco dividendo il prezzo scende del dividendo). I dati non sono nel repository, tranne il piccolo file del Tesoro: i file di French si scaricano con `scripts/00_scarica_dati.py` (controlla il robots.txt e aspetta 2 secondi tra le richieste), il file di GLD va messo a mano in `data/raw/gld.us.txt`. Controlli sui dati e dettagli in [docs/metodo.md](docs/metodo.md#1-dati-controlli-e-scelte).

## 4. Il metodo in quattro passi

**Passo 1, regimi di volatilità.** Dai rendimenti logaritmici di Mkt calcolo la volatilità EWMA (λ = 0,94, annualizzata). Tre stati: calma sotto il quantile 0,40, stress sopra il quantile 0,90, normale in mezzo. I quantili sono calcolati solo sul passato (finestra espandibile, minimo 252 giorni) e lo stato di un giorno passato **non viene mai riclassificato** con soglie successive: altrimenti la matrice userebbe informazione del futuro.

**Passo 2, catena di Markov.** Conto le transizioni tra stati già osservate, con lisciamento di Laplace (alfa = 1), e ottengo la matrice di transizione. La probabilità di stress a 5 giorni è p(t) = elemento (stato di oggi, stress) della matrice alla quinta; cinque giorni sono una settimana di borsa, la frequenza con cui ribilancio. Un test di ordine rifiuta la catena di ordine 1 (i regimi hanno più memoria di un giorno), quindi la considero un'approssimazione.

**Passo 3, variante multi-scala.** Tre catene con memorie della volatilità di 5, 15 e 42 giorni, combinate con pesi che premiano le catene che hanno previsto meglio negli ultimi giorni (η = 10, sconto 0,99, fissati in anticipo). Al giorno t uso solo esiti già noti. Dettagli e limiti di questa variante in [docs/metodo.md](docs/metodo.md#2-regimi-catena-e-multi-scala-dettagli).

![Pesi delle tre catene](img/pesi_scale_test.png)

**Passo 4, allocazione.** Il nucleo rischioso è NoDur, Mkt, Chips, Bond7 e oro, con pesi proporzionali all'inverso della volatilità a 60 giorni; il resto sta in liquidità (RF). L'esposizione al nucleo è

```
L(t) = min(1, m · σ_rif · (1 − (1 − r) · p(t)) / σ_nucleo(t))
```

dove σ_nucleo è la volatilità a 60 giorni del nucleo, `σ_rif` la sua mediana nel periodo di avviamento (0,0532), `m` scala il rischio obiettivo e `r` è la quota di rischio che tengo anche con p = 1. Niente leva (L ≤ 1); con p = 0 la regola diventa il semplice vol targeting.

Decido l'ultimo giorno di borsa di ogni settimana (giorno d) con i dati fino alla chiusura; l'ordine viene eseguito alla chiusura di d+1 e il portafoglio guadagna dal giorno d+2. Costi: 5 punti base sul turnover delle sole gambe rischiose (sensibilità a 0, 10 e 25 punti base nella sezione 6). La probabilità guarda 5 giorni avanti da d, ma la posizione si tiene da d+2: c'è uno scarto di due giorni che non ho corretto.

I parametri `q_alto`, `m` e `r` li scelgo su una griglia di 12 configurazioni, sul solo sviluppo (criterio: Sharpe netto più alto, a parità entro 0,02 il turnover più basso). Quelli scelti per la strategia, `m` = 0,8 e `r` = 0,25, stanno al **bordo della griglia**: l'ottimo potrebbe essere fuori e non ho allargato la griglia dopo aver visto i risultati. Dettagli in [docs/metodo.md](docs/metodo.md#3-come-ho-scelto-i-parametri).

## 5. Cosa confronto

Se la catena è utile, deve battere regole più semplici che usano la stessa informazione. Per questo la strategia (multi-scala) è confrontata con:

| Nome nei risultati | Che cos'è |
|---|---|
| mercato | indice Mkt, comprato e tenuto |
| 60_40 | 60% Mkt e 40% Bond7, ribilanciato ogni mese |
| nucleo_statico | gli stessi 5 strumenti con gli stessi pesi, sempre investito (L = 1) |
| vol_target | stessa formula con p = 0, cioè solo vol targeting, m scelto sullo sviluppo |
| vol_target_pari | vol targeting con m scelto perché l'esposizione media coincida con quella della strategia nella finestra in esame |
| esposizione_costante | nucleo statico moltiplicato per l'esposizione media della strategia |
| stato | la stessa formula con p = 1 se oggi lo stato è stress, 0 altrimenti: isola l'effetto della matrice da quello di aver discretizzato la volatilità |
| base | catena a un giorno di memoria |
| hmm | modello di Markov nascosto gaussiano a 3 stati, EM scritto in numpy, solo filtro in avanti, riaddestrato ogni anno (dal 1998) su dati passati |
| logistica | regressione logistica L2 (IRLS) con 5 variabili: log EWMA, log volatilità realizzata a 5/22/66 giorni, log(1 + giorni di permanenza nello stato), riaddestrata ogni 252 giorni |
| trend | uno strumento resta in portafoglio se il suo prezzo è sopra la media mobile a 200 giorni |
| multi | la strategia principale: catena multi-scala |

I confronti «vol targeting a esposizione pari» ed «esposizione costante» usano l'esposizione media della strategia, nota solo a posteriori: sono favorevoli ai benchmark.

Misure usate: Sharpe sull'eccesso rispetto a RF, rendimento composto annuo, volatilità, drawdown massimo, turnover e, per le differenze, un bootstrap stazionario (10.000 ricampionamenti, blocco medio 20 giorni) con intervallo al 95%. Le altre misure (commissione di performance, alfa, Deflated Sharpe, placebo, potenza) sono descritte in [docs/metodo.md](docs/metodo.md#4-come-misuro-i-risultati).

## 6. Risultati sul test

Periodo 1° gennaio 2015 – 30 giugno 2026, 2.889 giorni, costi 5 punti base. Output integrale in [docs/esito_test.txt](docs/esito_test.txt) (copia di quello prodotto da `06_inferenza.py test`).

| | Sharpe | Rendimento | Volatilità | Drawdown | Turnover annuo | Esposizione media |
|---|---|---|---|---|---|---|
| mercato | 0,697 | 14,0% | 18,2% | −34,2% | 0,09 | 1,00 |
| 60/40 | 0,671 | 9,0% | 10,6% | −20,9% | 0,23 | 1,00 |
| nucleo statico | 0,792 | 7,9% | 7,3% | −15,3% | 1,55 | 1,00 |
| esposizione costante | 0,793 | 6,0% | 4,9% | −10,2% | 1,04 | 0,67 |
| vol targeting (m = 0,8) | 0,770 | 5,8% | 4,8% | −8,7% | 1,70 | 0,71 |
| vol targeting a esposizione pari | 0,765 | 5,5% | 4,5% | −8,3% | 1,63 | 0,67 |
| trend (SMA 200) | 0,665 | 5,6% | 5,3% | −8,4% | 4,99 | 0,71 |
| HMM | 0,766 | 6,1% | 5,2% | −8,3% | 2,42 | 0,80 |
| logistica | 0,768 | 5,5% | 4,4% | −6,6% | 2,06 | 0,68 |
| stato corrente | 0,769 | 5,5% | 4,4% | −7,3% | 2,28 | 0,67 |
| catena base | 0,775 | 5,5% | 4,4% | −7,1% | 2,12 | 0,67 |
| **multi-scala** | 0,753 | 5,4% | 4,4% | −6,6% | 2,16 | 0,67 |

La strategia rende meno del mercato per due motivi: il nucleo (con obbligazioni e oro) rende già meno (nucleo statico 7,9% contro 14,0%) e la strategia ne investe in media il 67%.

![Drawdown nel test](img/drawdown_test.png)

![Perdita massima dal picco](img/drawdown_massimo_test.png)

### Differenze con intervallo al 95%

Differenza «multi-scala meno confronto». Per il drawdown la differenza è positiva quando la multi-scala perde meno.

| Confronto | Δ Sharpe | IC 95% | Δ drawdown | IC 95% |
|---|---|---|---|---|
| vol targeting a esposizione pari | −0,012 | [−0,106; +0,082] | +0,017 | [−0,008; +0,023] |
| esposizione costante | −0,039 | [−0,274; +0,211] | +0,037 | [−0,021; +0,048] |
| vol targeting | −0,016 | [−0,110; +0,080] | +0,021 | [−0,004; +0,031] |
| stato corrente | −0,016 | [−0,136; +0,094] | +0,007 | [−0,026; +0,012] |
| catena base | −0,021 | [−0,087; +0,041] | +0,006 | [−0,017; +0,006] |
| HMM | −0,013 | [−0,083; +0,057] | +0,017 | [+0,003; +0,030] |
| logistica | −0,014 | [−0,068; +0,035] | 0,000 | [−0,013; +0,005] |
| trend | +0,088 | [−0,182; +0,382] | +0,018 | [−0,021; +0,044] |
| nucleo statico | −0,039 | [−0,275; +0,212] | +0,087 | [+0,015; +0,119] |
| mercato | +0,057 | [−0,411; +0,492] | +0,276 | [+0,115; +0,423] |
| 60/40 | +0,082 | [−0,322; +0,474] | +0,144 | [+0,047; +0,230] |

![Differenze con intervallo](img/differenze_bootstrap.png)

Sul drawdown l'intervallo esclude lo zero a favore della multi-scala contro mercato, 60/40, nucleo statico e, per poco, HMM. Non lo esclude contro vol targeting, catena base, stato corrente e regressione logistica: la riduzione del rischio la danno anche loro.

### Altri controlli (tutti contro il vol targeting a esposizione pari, se non indicato)

- **Sottoperiodi** (divisione al 1° gennaio 2020): differenza di Sharpe −0,041 fino al 2019 e −0,009 dal 2020. Contro l'esposizione costante: −0,006 e −0,165. Il segno è negativo in entrambe le parti.
- **Costi**: differenza di Sharpe −0,005 con 0 bp, −0,018 con 10 bp, −0,038 con 25 bp.
- **Commissione di performance**: −13,7 / −11,6 / −8,9 punti base annui per γ = 1 / 5 / 10, cioè negativa.
- **Alfa sul nucleo statico**: +0,13% annuo, t = 0,24 (20 e 60 ritardi), beta 0,55.
- **Placebo**: il 42,7% delle traslazioni cicliche di p ha Sharpe pari o superiore a quello della strategia: il tempismo del segnale non si distingue da un segnale casuale con le stesse caratteristiche.
- **Deflated Sharpe**: 0,982 con 60 prove e 0,979 con 192. Lo Sharpe resta positivo dopo la selezione, ma questo non dice nulla sul valore aggiunto della catena.
- **Per regime** (stato del giorno prima): nei giorni di stress (10,3% del totale) la multi-scala ha eccesso annuo +1,6% con volatilità 4,7%, il nucleo statico +13,8% con volatilità 13,0%: la strategia ha ridotto il rischio e ha rinunciato ai rimbalzi.

### Quanto poteva vedere il test

![Potenza del test](img/potenza.png)

| Vantaggio di Sharpe vero | 0,00 | 0,05 | 0,10 | 0,15 | 0,20 | 0,30 |
|---|---|---|---|---|---|---|
| Intervallo tutto positivo | 1,0% | 9,3% | 42,7% | 77,7% | 97,3% | 100% |

(300 ripetizioni per punto, quindi un errore di circa ±3 punti percentuali sui valori intermedi; lunghezza uguale al test.) Con vantaggio nullo l'intervallo risulta tutto positivo in 3 casi su 300. In pratica: un vantaggio grande lo avrei visto, uno piccolo (fino a 0,10) spesso no.

![Sharpe in sviluppo e nel test](img/sharpe_sviluppo_test.png)

Nel test lo Sharpe delle strategie adattive scende rispetto allo sviluppo, mentre mercato e 60/40 salgono (periodo più favorevole alle azioni). Il leggero vantaggio della multi-scala sul vol targeting che c'era nello sviluppo (+0,046) sparisce (−0,012).

Sullo sviluppo, la parte previsiva da sola e le tabelle complete sono in [docs/metodo.md](docs/metodo.md#5-risultati-sullo-sviluppo).

## 7. Limiti

- **Un solo periodo di test**, 11,5 anni con pochi episodi di stress veri. La potenza è limitata: l'assenza di evidenza non è evidenza di assenza.
- **Parametri al bordo della griglia** (`m` = 0,8, `r` = 0,25).
- **Confronti favorevoli ai benchmark**: l'esposizione media della strategia è nota solo a posteriori.
- **Regimi definiti dalla volatilità**, che è persistente: prevedere che domani sarà ancora volatile è facile e non dimostra valore economico. Ridurre l'esposizione abbassa il drawdown anche senza segnale.
- **Rendimenti obbligazionari stimati** (non un indice vero) e **oro** solo come prezzo di GLD, con un giorno mancante.
- **Scelta del nucleo** (NoDur, Chips, oro) senza una motivazione forte oltre ai dati disponibili; fatta conoscendo il periodo 2015-2026.
- **Nessuna correzione per confronti multipli**; costi semplificati; portafoglio virtuale, non eseguibile.
- **Analisi del test fatte dopo l'esito** (parte previsiva, grafici, notebook): letture post hoc.

Elenco completo in [docs/metodo.md](docs/metodo.md#8-limiti). Non è un consiglio di investimento.

## 8. Come riprodurre i numeri

Serve Python 3.10 o superiore.

```
pip install -e ".[dev,dati,notebook]"
python scripts/00_scarica_dati.py        # file di French (poi mettere a mano data/raw/gld.us.txt)
python scripts/01_controlla_dati.py      # qualità dei dati
python -m pytest -q -m "not slow"         # test veloci (senza -m si eseguono anche i 2 lenti, circa 5 minuti)
```

Sviluppo, in ordine (le tabelle vanno in `output/`, non versionata):

```
python scripts/02_regimi_sviluppo.py     # regimi, matrice, test di ordine, previsione (catena)
python scripts/03_multiscala_sviluppo.py # variante multi-scala
python scripts/04_allocazione_sviluppo.py
python scripts/05_concorrenti_sviluppo.py
python scripts/06_inferenza.py sviluppo  # prova generale dell'analisi sullo sviluppo
python scripts/07_potenza.py             # potenza (circa 8 minuti), scrive output/potenza.csv
python scripts/08_walk_forward_sviluppo.py
```

Test e risultati finali:

```
python scripts/06_inferenza.py test      # scrive output/esito_test.txt (vedi nota sotto)
python scripts/10_analisi_descrittive.py # parte previsiva (docs/metodo.md, sezione 6)
python scripts/11_grafici.py             # scrive i grafici in img/ (circa 3 minuti)
```

Il notebook `notebooks/risultati.ipynb` riesegue tutto il calcolo del test (alcuni minuti) e mostra tabelle e grafici.

Nota sul test: lo script si rifiuta di ripartire se `output/esito_test.txt` contiene già un'analisi completa. Serve per non guardare il test più volte per distrazione. In un clone nuovo, dopo aver generato le tabelle dello sviluppo (script 04 e 05), parte una volta sola. `config_congelata.json` è versionato; `09_congela_configurazioni.py` lo ha creato dalle tabelle dello sviluppo e non lo sovrascrive.

## 9. Struttura del repository

```
src/markov_regime_allocation/
    dati.py          lettura e preparazione dei dati
    regimi.py        volatilità EWMA, stati senza futuro, matrici, probabilità di stress
    previsione.py    previsioni di varianza e di stress, QLIKE, Diebold-Mariano, test di ordine
    multiscala.py    catene con memorie diverse e pesi adattivi
    motore.py        nucleo, esposizione, simulazione con esecuzione ritardata e costi, metriche
    hmm.py           HMM gaussiano (EM in numpy, solo filtro in avanti)
    logistica.py     regressione logistica L2 (IRLS) con memoria
    trend.py         filtro di tendenza
    esperimento.py   griglie, scelta delle configurazioni
    inferenza.py     bootstrap, commissione di performance, alfa, Deflated Sharpe, tabella per regime
    esame.py         calcoli del test condivisi da script, grafici e notebook
    grafici.py       grafici
scripts/             dal 00 all'11, nell'ordine della sezione 8
tests/               test (anche contro l'uso del futuro, su dati sintetici e reali)
notebooks/           risultati.ipynb
img/                 grafici del README
docs/                criteri del verdetto scritti prima del test, esito registrato del test, dettagli del metodo (metodo.md), domande e risposte
data/treasury/       rendimenti del Tesoro (file piccolo, versionato)
config_congelata.json  configurazioni scelte sullo sviluppo
```

## 10. Riferimenti

- Bailey, D. H., López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management* 40(5), 94 e seguenti.
- Cederburg, S., O'Doherty, M. S., Wang, F., Yan, X. S. (2020). On the performance of volatility-managed portfolios. *Journal of Financial Economics* 138(1), 95-117.
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics* 7(2), 174-196.
- Diebold, F. X., Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics* 13(3), 253-263.
- Fleming, J., Kirby, C., Ostdiek, B. (2001). The Economic Value of Volatility Timing. *Journal of Finance* 56(1), 329-352.
- J.P. Morgan/Reuters (1996). *RiskMetrics Technical Document*, quarta edizione.
- Moreira, A., Muir, T. (2017). Volatility-Managed Portfolios. *Journal of Finance* 72(4), 1611-1644.
- Newey, W. K., West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica* 55(3), 703-708.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics* 160(1), 246-256.
- Politis, D. N., Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association* 89(428), 1303-1313.
- Kenneth R. French, Data Library (rendimenti giornalieri dei fattori e dei portafogli per industria).