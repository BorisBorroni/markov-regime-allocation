# Criteri del verdetto

Testo scritto prima di guardare il periodo di test (dal 1° gennaio 2015). Il repository non contiene una prova della data di scrittura: è il testo che avevo fissato allora, riportato qui senza modifiche di sostanza.

## Cosa considero un risultato positivo

C'è una sola misura primaria. Tutto il resto è descrittivo e non decide il verdetto.

**Livello 1: la strategia è utile?** Contro mercato, 60/40 e nucleo statico, al netto dei costi (5 punti base): drawdown massimo più basso e Sharpe non peggiore.

**Livello 2: la catena aggiunge valore?** Misura primaria: differenza di Sharpe netto (5 punti base) tra la strategia principale e il vol targeting a esposizione media uguale, con intervallo di confidenza bootstrap al 95% (blocchi stazionari, blocco medio 20). Controllo ulteriore: la stessa differenza contro l'esposizione costante, per escludere che conti solo il livello medio di rischio.

Esiti possibili:

- **Evidenza di valore aggiunto**: intervallo al 95% interamente favorevole, segno invariato con costi a 10 punti base, segno invariato in 2015-2019 e in 2020-2026.
- **Indicazione debole**: stima favorevole, intervallo che include lo zero.
- **Nessun valore aggiunto**: tutti gli altri casi. È un esito valido e lo riporto come tale.

Misure descrittive, che non decidono: drawdown, commissione di performance (γ = 1, 5, 10), alfa stile Moreira e Muir, Deflated Sharpe, tabella per regime, HMM, regressione logistica, trend, placebo, sensibilità ai costi.

## Scelte collegate

- Strategia principale: la multi-scala, perché ha superato in sviluppo la regola fissata in anticipo (QLIKE sulla varianza inferiore alla catena base, statistica di Diebold-Mariano ≤ −2). La catena base resta come confronto.
- Divisione invariata: sviluppo fino al 31 dicembre 2014, test dal 1° gennaio 2015. La ri-selezione annuale dei parametri (walk-forward) serve solo come robustezza.
- Il test si guarda una volta sola. La variante multi-scala l'ho introdotta dopo aver visto, nello sviluppo, che la catena semplice non prevedeva meglio di EWMA e HAR.
- Il Monte Carlo serve a misurare taglia e potenza della regola di decisione, non a simulare rendimenti dalla catena.
- Tentativi da dichiarare per il Deflated Sharpe: 57 celle di griglia più 3 finestre del trend (60), e 192 come valore prudenziale.
