# Dettagli del metodo e dei controlli

Qui ci sono i dettagli che nel [README](../README.md) ho lasciato fuori per non appesantirlo: controlli sui dati, scelta dei parametri, misure statistiche, risultati sullo sviluppo, parte previsiva, controlli contro l'uso del futuro e limiti completi. I numeri sono gli stessi del README.

## Indice

1. [Dati: controlli e scelte](#1-dati-controlli-e-scelte)
2. [Regimi, catena e multi-scala: dettagli](#2-regimi-catena-e-multi-scala-dettagli)
3. [Come ho scelto i parametri](#3-come-ho-scelto-i-parametri)
4. [Come misuro i risultati](#4-come-misuro-i-risultati)
5. [Risultati sullo sviluppo](#5-risultati-sullo-sviluppo)
6. [La parte previsiva da sola](#6-la-parte-previsiva-da-sola)
7. [Controlli contro l'uso del futuro e test del codice](#7-controlli-contro-luso-del-futuro-e-test-del-codice)
8. [Limiti](#8-limiti)

---

## 1. Dati: controlli e scelte

**Perché indici e non ETF.** Avevo i file giornalieri di alcuni ETF (SPY, IEF, ecc.), ma nei giorni di stacco dividendo il prezzo scendeva del dividendo (per SPY −0,32% in media contro +0,04% negli altri giorni), cioè non erano rendimenti totali. Ho quindi usato gli indici di French, che sono a rendimento totale. Gli ETF li ho usati solo come controllo di coerenza (correlazione giornaliera con Mkt 0,986 per SPY e con Bond7 0,931 per IEF, nel 2005-2014). Questi controlli li ho fatti una tantum con i file locali degli ETF, che non sono nel repository: i numeri su ETF di questo paragrafo non si riproducono con gli script.

Controlli sui dati (`scripts/01_controlla_dati.py`): nessun valore mancante nelle serie azionarie e in RF; dal 2005 manca solo il primo rendimento dell'oro e il 17 febbraio 2011 nel file GLD (il rendimento del giorno dopo copre due giorni); 39 giorni di borsa dal 2005 senza dato del Tesoro (festivi del mercato obbligazionario), dove uso l'ultimo valore.

**I dati non sono nel repository**, tranne il file piccolo del Tesoro. I file di French si scaricano con `scripts/00_scarica_dati.py`, che controlla il robots.txt del sito prima di ogni richiesta e aspetta 2 secondi tra una richiesta e l'altra. Il file di GLD va messo a mano in `data/raw/gld.us.txt` (non ho letto i termini di Stooq per la ridistribuzione, quindi non lo includo).

## 2. Regimi, catena e multi-scala: dettagli

**Soglie solo dal passato.** I quantili sono calcolati su tutta la storia fino a quel giorno (finestra espandibile, minimo 252 giorni). Lo stato di ogni giorno passato è quello assegnato quel giorno e **non viene mai riclassificato** con soglie successive. Se lo facessi, la matrice di transizione userebbe informazione del futuro.

Soglie, matrici e modelli usano tutta la storia di Mkt disponibile (dal 1926), non solo dal 2005. Nel periodo di sviluppo (2005-2014) gli stati sono: calma 21,9%, normale 63,1%, stress 14,9% dei giorni.

- **Matrice di transizione**: conto, giorno per giorno, quante volte si è passati da uno stato all'altro tra i giorni già osservati, con lisciamento di Laplace (alfa = 1) per evitare probabilità esattamente zero. Alla fine dello sviluppo la diagonale è 0,962 / 0,961 / 0,941: i regimi sono molto persistenti.
- **Probabilità di stress a 5 giorni**: p(t) = elemento (stato di oggi, stress) della matrice elevata alla quinta. Cinque giorni sono una settimana di borsa, la frequenza con cui ribilancio.

Un test di ordine (rapporto di verosimiglianza tra catena di ordine 1 e di ordine 2, 12 gradi di libertà) **rifiuta** l'ipotesi di Markov di ordine 1: LR = 119,6 nello sviluppo (p < 10⁻¹⁴) e 34,9 nel test (p = 0,0005). I regimi hanno quindi più memoria di un giorno. Il confronto tra la durata dei periodi osservata e quella geometrica implicita nella matrice conferma la cosa (sezione 6 di questo file). Le osservazioni sono dipendenti, quindi il chi quadro è solo approssimato.

Dato che la memoria di un giorno non basta, ho provato a usare tre catene con memorie diverse della volatilità: 5, 15 e 42 giorni di borsa (λ = 1 − 2/(N+1)). Ogni catena ha i suoi stati e la sua probabilità di stress. Le tre probabilità vengono combinate con pesi proporzionali a exp(−η · perdita media scontata), con η = 10 e sconto 0,99. La perdita è il log-score su un evento comune (lo stress della catena a 15 giorni, tra 5 giorni). Al giorno t uso solo perdite già note, cioè fino a t − 5, perché l'esito del giorno u si conosce al giorno u + 5.

η e lo sconto non sono ottimizzati: li ho fissati in anticipo. La regola per passare all'allocazione (errore QLIKE sulla varianza inferiore alla catena base, statistica di Diebold-Mariano ≤ −2) e la sensibilità a η e allo sconto (QLIKE tra 0,4317 e 0,4365) riguardano la versione che combina le **previsioni di varianza** con pesi da QLIKE. La strategia usa invece la versione che combina le **probabilità di stress**, con pesi dal log-score: per questa non ho una regola di passaggio né una sensibilità separata. Nello sviluppo la regola sulla varianza è superata: differenza −0,031, t = −3,58. Contro la sola catena a 15 giorni la differenza è più piccola (−0,013, t = −1,89), quindi parte del miglioramento sulla catena base può venire dalla memoria più lunga e non dalla combinazione.

## 3. Come ho scelto i parametri

Per ogni strategia con probabilità di stress provo una griglia di 12 configurazioni: `q_alto` in {0,80; 0,90}, `m` in {0,8; 1,0; 1,2}, `r` in {0,25; 0,50}. Sullo sviluppo scelgo lo Sharpe netto più alto; se due configurazioni sono entro 0,02 vince quella con meno turnover. Lo stesso criterio vale per tutte le strategie, così nessuna ha più «sforzo di taratura» delle altre. Il vol targeting prova 3 valori di `m`, l'HMM 6 configurazioni, il trend ha la finestra fissata a 200 giorni (100 e 250 solo come sensibilità).

Configurazioni scelte (file `config_congelata.json`): multi, base, stato e logistica scelgono `q_alto` = 0,90, `m` = 0,8, `r` = 0,25; il vol targeting `m` = 0,8; l'HMM `m` = 1,0 e `r` = 0,50.

`m` = 0,8 e `r` = 0,25 sono i **valori più bassi della griglia**: l'ottimo potrebbe stare fuori. Non ho allargato la griglia dopo aver visto i risultati.

Tentativi contati per il Deflated Sharpe: 57 celle di griglia più 3 finestre del trend, cioè 60. Non includono gli iperparametri della multi-scala (memorie, η, sconto, λ, quantili) né la scelta della variante; 192 è un valore prudenziale scelto da me, senza una derivazione.

## 4. Come misuro i risultati

- **Previsione della varianza**: confronto con EWMA e con un HAR (regressione sulla varianza realizzata a 1, 5 e 22 giorni, Corsi 2009), con perdita QLIKE (Patton 2011) e test di Diebold-Mariano.
- **Metriche**: Sharpe sull'eccesso rispetto a RF (media/deviazione standard × √252), rendimento composto annuo, volatilità, drawdown massimo (con il capitale iniziale 1 come punto di partenza), turnover annuo, esposizione media.
- **Bootstrap stazionario** (Politis e Romano), 10.000 ricampionamenti, blocco medio 20 giorni, con gli stessi indici per strategia e confronto, per le differenze di Sharpe e di drawdown massimo (intervallo al 95%). Il drawdown dipende dal percorso: con i blocchi è poco stabile, e lo tengo in conto nella lettura.
- **Commissione di performance** (Fleming, Kirby e Ostdiek): quanto, in punti base annui, un investitore con utilità quadratica e avversione al rischio γ = 1, 5, 10 sarebbe disposto a pagare per passare dal vol targeting alla strategia. Calcolata sui rendimenti semplici (1 + r), al netto dei costi di transazione.
- **Alfa** della strategia rispetto al nucleo statico (stile Moreira e Muir; Cederburg e coautori mettono in dubbio la tenuta fuori campione di queste strategie), con errori standard di Newey-West a 20 e 60 ritardi.
- **Deflated Sharpe** (Bailey e López de Prado): probabilità che lo Sharpe vero superi quello massimo atteso tra N prove con Sharpe vero nullo. Lo applico ai rendimenti del test, dove la selezione non è avvenuta, quindi lo uso come controllo e non come prova. Non misura se la catena aggiunge valore.
- **Placebo**: la serie p(t) viene traslata ciclicamente di un ritardo casuale di almeno un anno (1.000 traslazioni). Conserva distribuzione e persistenza di p ma rompe il legame con i rendimenti. Il valore è la quota di traslazioni con Sharpe pari o superiore a quello della strategia.
- **Potenza** (Monte Carlo): ricampiono a blocchi le coppie di rendimenti dello sviluppo (strategia, vol targeting a esposizione pari), sposto la strategia di una costante in modo che la differenza di Sharpe vera sia nota, e conto quante volte la regola di decisione (bootstrap con intervallo al 95%) la riconosce. Il bootstrap interno usa 400 ricampionamenti (non 10.000) per tenere i tempi accettabili; i 1.975 giorni di sviluppo sono ricampionati fino alla lunghezza del test. Non simula dalla catena (sarebbe circolare): conserva solo dipendenza temporale e code dello sviluppo.
- **Semi del generatore casuale**: fissi (bootstrap, placebo, potenza); non ho verificato quanto cambino gli intervalli con semi diversi.

## 5. Risultati sullo sviluppo

Periodo 1° marzo 2007 – 31 dicembre 2014 (comprende il 2008), costi 5 punti base. Questi sono i numeri su cui ho scelto le configurazioni, quindi sono ottimistici per costruzione.

| | Sharpe | Rendimento | Volatilità | Drawdown | Turnover annuo | Esposizione media |
|---|---|---|---|---|---|---|
| mercato | 0,418 | 7,9% | 22,4% | −54,6% | 0,13 | 1,00 |
| 60/40 | 0,556 | 7,0% | 12,2% | −32,3% | 0,42 | 0,99 |
| nucleo statico | 0,949 | 8,6% | 8,3% | −18,7% | 1,65 | 1,00 |
| esposizione costante | 0,950 | 5,3% | 4,8% | −10,8% | 0,95 | 0,58 |
| vol targeting (m = 0,8) | 0,979 | 5,3% | 4,6% | −9,1% | 1,54 | 0,63 |
| vol targeting a esposizione pari | 0,971 | 4,9% | 4,3% | −8,3% | 1,44 | 0,58 |
| trend (SMA 200) | 0,981 | 6,6% | 6,0% | −8,6% | 4,58 | 0,75 |
| HMM | 1,027 | 6,0% | 5,1% | −8,7% | 2,34 | 0,73 |
| logistica | 1,057 | 5,1% | 4,1% | −5,7% | 1,87 | 0,59 |
| stato corrente | 1,073 | 5,1% | 4,1% | −4,5% | 1,98 | 0,58 |
| catena base | 1,050 | 5,0% | 4,1% | −5,7% | 1,84 | 0,58 |
| **multi-scala** | 1,017 | 4,9% | 4,1% | −6,7% | 1,82 | 0,58 |

Tutti i segnali basati sulla volatilità (multi, base, stato, HMM, logistica) stanno tra 1,02 e 1,07 di Sharpe e sono appena sopra il vol targeting (0,97-0,98). Le loro esposizioni sono quasi la stessa serie: la correlazione tra L(t) della multi-scala e quella del vol targeting è 0,968, con lo stato corrente 0,973, con la catena base 0,987. Con correlazioni così, le differenze di rendimento non si possono attribuire alla matrice con sicurezza. La multi-scala non batte né la catena base né lo stato corrente.

Prima del test ho anche fatto una prova generale della stessa analisi sullo sviluppo (`06_inferenza.py sviluppo`): nessun confronto «della stessa famiglia» aveva una differenza di Sharpe distinguibile da zero. Una verifica a parte con ri-selezione annuale dei parametri (`08_walk_forward_sviluppo.py`, 2009-2014) dà Sharpe 1,307 contro 1,401 della configurazione fissa e 1,412 del vol targeting con m = 0,8: in quel sottoperiodo il vol targeting semplice eguaglia la strategia.

## 6. La parte previsiva da sola

Separo la domanda «la catena prevede bene?» da «l'allocazione rende?». `scripts/10_analisi_descrittive.py` valuta la parte previsiva sullo sviluppo e sul test separatamente (output in `output/descrittive.txt`).

**Varianza a 5 giorni** (errore QLIKE, più basso è meglio; test di Diebold-Mariano con errori standard di Newey-West a 10 ritardi):

| | Sviluppo | Test |
|---|---|---|
| multi-scala | 0,4327 | 0,5319 |
| catena base | 0,4638 | 0,5789 |
| EWMA | 0,3838 | 0,5330 |
| HAR | 0,3724 | 0,4625 |
| multi − base (t) | −0,031 (−3,58) | −0,047 (−3,81) |
| multi − EWMA (t) | +0,049 (+2,01) | −0,001 (−0,03) |
| multi − HAR (t) | +0,060 (+3,68) | +0,069 (+3,72) |

La multi-scala prevede meglio della catena base nei due periodi, ma non meglio di una semplice EWMA (peggio nello sviluppo, pari nel test) e peggio di un HAR. Il bersaglio, la media dei quadrati dei rendimenti giornalieri nei 5 giorni successivi, è molto rumoroso (non uso dati intragiornalieri).

**Probabilità di stress a 5 giorni** (evento: stress della scala a 15 giorni), punteggio di Brier e log-score, più basso è meglio:

| | Brier sviluppo | Brier test | Log-score sviluppo | Log-score test |
|---|---|---|---|---|
| multi-scala | 0,0475 | 0,0554 | 0,1751 | 0,2016 |
| persistenza («domani come oggi») | 0,0436 | 0,0645 | 0,2106 | 0,3064 |
| frequenza storica | 0,1335 | 0,0950 | 0,4565 | 0,3409 |

Nel test la multi-scala batte la persistenza su entrambe le misure; nello sviluppo la persistenza ha il Brier migliore e la multi-scala il log-score migliore. Il log-score taglia le probabilità a [0,01; 0,99] e la persistenza vale 0 o 1, quindi il confronto sul log-score dipende da quel taglio: il Brier è la misura più affidabile.

**Durata dei regimi**: quota di periodi consecutivi nello stesso stato con durata almeno 5, 20 e 60 giorni, osservata contro quella attesa se la catena di ordine 1 fosse vera.

| Stato | Periodo | ≥ 5 giorni | ≥ 20 giorni | ≥ 60 giorni |
|---|---|---|---|---|
| stress | sviluppo, osservata | 0,190 | 0,143 | 0,095 |
| stress | sviluppo, attesa | 0,792 | 0,329 | 0,032 |
| stress | test, osservata | 0,667 | 0,417 | 0,083 |
| stress | test, attesa | 0,849 | 0,459 | 0,089 |

Nello sviluppo molti periodi di stress durano pochi giorni ma qualcuno dura a lungo: la distribuzione non è geometrica. Una spiegazione plausibile, che non ho verificato, è che la volatilità oscilli attorno alla soglia. Nel test l'accordo è migliore. Tabelle complete per i tre stati in `output/descrittive.txt`.

![Matrice di transizione e volatilità](../img/regimi.png)

## 7. Controlli contro l'uso del futuro e test del codice

Il rischio principale di un lavoro di questo tipo è l'uso inconsapevole di informazione futura. Ho controllato così:

- **Test di troncamento** su tutti i segnali (stati, soglie, matrice, p, previsioni, pesi della multi-scala, σ_nucleo, σ_rif, probabilità dell'HMM e della logistica) e sulle simulazioni: se si alterano o si tagliano i dati dopo una certa data, tutto ciò che precede resta identico. `tests/test_assenza_futuro_dati_reali.py` lo fa sui dati veri (lento, circa 5 minuti, si salta se mancano i dati grezzi).
- **Esecuzione ritardata**: un test verifica che dopo una decisione nel giorno d il portafoglio resti in liquidità fino al giorno d+1 e guadagni dal rendimento del giorno d+2.
- **Controlli sulle formule**: in una revisione ho confrontato Deflated Sharpe, Newey-West e bootstrap stazionario con implementazioni scritte a parte (stesso risultato); quei confronti non sono nel repository e non sono un confronto con il testo degli articoli. I test contengono calcoli fatti a mano per drawdown, durata dei regimi e test di ordine; per la commissione di performance verificano che l'equazione sia risolta e che chi ha più avversione al rischio paghi di più.
- **Riproducibilità**: rigenerando le tabelle con il codice, i numeri dello sviluppo coincidono byte per byte; `11_grafici.py` verifica che gli Sharpe ricalcolati sul test coincidano con quelli salvati.
- **Test**: 61 veloci più 2 lenti sui dati reali (circa 5 minuti, si saltano se mancano i dati grezzi, quindi non girano nella CI), con i warning trasformati in errori (`filterwarnings = error` in `pytest.ini`). I veloci girano con Python 3.10 e 3.13 su GitHub Actions; controllo di stile con ruff (regole E, F, B).

Una correzione fatta prima del test: nella prima versione la commissione di performance usava i rendimenti r invece dei rendimenti 1 + r nell'utilità quadratica, e γ risultava quasi ininfluente.

## 8. Limiti

- **Un solo periodo di test**, 11,5 anni con pochi episodi di stress veri (fine 2015, inizio 2019, 2020, 2022, 2025). Il test non ha molta potenza (README, «Risultati sul test»): l'assenza di evidenza non è evidenza di assenza.
- **Parametri al bordo della griglia** (`m` = 0,8, `r` = 0,25): l'ottimo potrebbe stare fuori.
- **Confronti favorevoli ai benchmark**: vol targeting a esposizione pari ed esposizione costante usano l'esposizione media della strategia, nota solo a posteriori.
- **Regimi definiti dalla volatilità**, che è persistente: prevedere che domani sarà ancora volatile è facile e non dimostra valore economico. Ridurre l'esposizione abbassa il drawdown anche senza segnale.
- **Rendimenti obbligazionari stimati** (non un indice vero): rullaggio ignorato, orario di rilevazione del Tesoro diverso dalla chiusura azionaria. L'accordo con IEF è buono, non perfetto.
- **Oro**: solo rendimento del prezzo di GLD, con un giorno mancante nel file (17 febbraio 2011).
- **Nessuna correzione per confronti multipli.** Il Deflated Sharpe copre solo la selezione della configurazione. Gli intervalli e i p-value sono molti e non corretti.
- **Test di ordine con osservazioni dipendenti**: il chi quadro è approssimato.
- **Formule non verificate sul testo originale**: le formule della commissione di performance e del Deflated Sharpe sono scritte come le conosco dagli articoli; ho confrontato il mio codice con altre implementazioni mie, non con le equazioni degli articoli riga per riga.
- **Costi semplificati**: 5 punti base fissi sul turnover; niente impatto di mercato, tasse, scarti denaro-lettera diversi per strumento.
- **Portafoglio virtuale**: nessuna esecuzione reale, nessuna considerazione di liquidità o di vincoli operativi. Gli indici di French non sono strumenti acquistabili.
- **Scelta del nucleo**: NoDur, Chips e oro non hanno una motivazione forte oltre alla disponibilità dei dati; sceglierli dopo aver visto come sono andati tra il 2015 e il 2026 poteva influenzare il confronto con il nucleo statico e con il 60/40.
- **Un solo schema di segnale**: soglie, orizzonte di 5 giorni, finestre di 60 giorni sono scelte fisse, non testate in alternativa.
- **Analisi del test fatte dopo l'esito**: la sezione 6 di questo file, i grafici e il notebook sono letture post hoc.
- **Placebo**: trasla solo p e ha anch'esso poca potenza.
- **Nessuna analisi su altri mercati o altre definizioni di regime** (per esempio basate su altri indici o sul VIX).

Non è un consiglio di investimento.
