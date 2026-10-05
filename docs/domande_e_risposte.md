# Domande che potrebbero farmi

Risposte brevi, con i numeri del [README](../README.md). I dettagli sono in [metodo.md](metodo.md).

**1. Qual è il risultato in una frase?**
La strategia riduce molto il drawdown (6,6% contro 34,2% del mercato), ma non ho trovato prova che la catena di Markov aggiunga qualcosa a un semplice vol targeting: la differenza di Sharpe è −0,012, con intervallo al 95% [−0,106; +0,082].

**2. Perché un risultato negativo è un buon risultato?**
Perché la domanda era se la catena serve davvero, e ho fissato in anticipo come l'avrei giudicata. La risposta onesta è «non si dimostra». Un test che poteva dire no e dice no è più credibile di uno che dice sempre sì.

**3. Perché hai diviso sviluppo e test?**
Per non scegliere i parametri guardando i dati su cui poi giudico il risultato. I parametri li ho scelti fino al 2014 e li ho congelati; il test (dal 2015) l'ho lanciato una volta sola.

**4. Cosa significa che la catena di Markov è «di ordine 1» e perché la usi se il test la rifiuta?**
Ordine 1 vuol dire che la probabilità di domani dipende solo dallo stato di oggi. Il test di ordine rifiuta questa ipotesi: i regimi hanno più memoria. La uso come approssimazione semplice e per questo ho provato anche la variante multi-scala, che però non cambia il verdetto.

**5. Perché la multi-scala, se non batte le alternative?**
Perché dopo lo sviluppo la catena semplice non prevedeva la volatilità meglio di EWMA e HAR, e volevo vedere se la memoria più lunga aiutava. L'ho introdotta dopo aver visto quel risultato, e lo dichiaro nel README. Sul test non migliora lo Sharpe.

**6. Come eviti di usare informazione del futuro?**
Soglie dei regimi calcolate solo sul passato e mai riclassificate; ordini eseguiti il giorno dopo la decisione; test di troncamento: se tolgo i dati dopo una data, tutto ciò che precede resta identico.

**7. Perché il test non ti permette di concludere «la catena è inutile»?**
Perché ha potenza limitata: un vantaggio vero di 0,10 di Sharpe lo avrei riconosciuto nel 43% dei casi, uno di 0,15 nel 78%. Posso dire che non c'è evidenza di valore aggiunto, non che sia escluso.

**8. Perché la strategia rende meno del mercato?**
Il nucleo (con obbligazioni e oro) rende già meno: nucleo statico 7,9% annuo contro 14,0% del mercato. In più la strategia è investita in media al 67%. Il vantaggio è sul rischio, non sul rendimento.

**9. Quali sono i limiti più importanti?**
Un solo periodo di test; parametri al bordo della griglia; benchmark costruiti con l'esposizione media della strategia, nota solo a posteriori; nucleo di strumenti scelto conoscendo il periodo; costi semplificati. L'elenco completo è in [metodo.md](metodo.md#8-limiti).

**10. Cosa faresti per proseguire?**
Allargare la griglia dei parametri, provare altre definizioni di regime (per esempio con il VIX) e altri mercati, e correggere per i confronti multipli.
