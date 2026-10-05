"""Studio Monte Carlo di controllo: quante volte la regola di decisione riconosce un vantaggio di Sharpe noto.

Si parte dalla coppia di rendimenti giornalieri (strategia multi-scala, vol targeting a esposizione pari) dello
sviluppo; si ricampionano a blocchi (stazionario, medio 20) fino alla lunghezza del test e si sposta la strategia di
una costante, in modo che la differenza di Sharpe vera sia esattamente delta. Per ogni delta si ripete la regola di
decisione (bootstrap con intervallo al 95%) e si conta quante volte l'estremo inferiore e' sopra zero.
Limite: la dipendenza temporale e le code vengono dallo sviluppo; non e' una simulazione dalla catena.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markov_regime_allocation import dati, inferenza, motore, multiscala  # noqa: E402
from markov_regime_allocation.esperimento import config_da_tabelle, esegui, m_per_esposizione, prepara  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output"
SVILUPPO_FINE = "2014-12-31"
DELTA = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
RIPETIZIONI = 300
B_INTERNO = 400


def main():
    cfg = config_da_tabelle(pd.read_csv(OUT / "griglia_sviluppo.csv"), pd.read_csv(OUT / "concorrenti_sviluppo.csv"))
    mkt, R, rf, w, vol, sigma_rif, dec = prepara(SVILUPPO_FINE)
    inizio = R.index[motore.AVVIAMENTO]
    p = multiscala.probabilita_stress_multiscala(mkt, q_alto=cfg["multi"]["q_alto"])[0].reindex(R.index)
    sim_a = esegui(R, rf, w, dec, motore.esposizione(p, vol, sigma_rif, cfg["multi"]["m"], cfg["multi"]["r"]))
    media = sim_a["esposizione"].loc[inizio:].mean()
    _, sim_b = m_per_esposizione(R, rf, w, vol, sigma_rif, dec, media, inizio, R.index[-1])
    a = (sim_a["ritorno"].loc[inizio:] - rf.loc[inizio:]).to_numpy()
    b = (sim_b["ritorno"].loc[inizio:] - rf.loc[inizio:]).to_numpy()
    # solo il numero di giorni del test (dal calendario), nessun rendimento
    n_test = int((dati.carica_serie().index >= "2015-01-01").sum())
    print(f"Giorni di sviluppo usati: {len(a)}; lunghezza simulata (circa il test): {n_test}")
    sr_b = inferenza.sharpe(b)
    rng = np.random.default_rng(2024)
    print("\ndelta vero | quota con estremo inferiore IC95 > 0 | quota con stima > 0")
    righe = []
    for delta in DELTA:
        c = (sr_b + delta) * a.std(ddof=1) / np.sqrt(252) - a.mean()
        ok_ic = ok_pt = 0
        for _ in range(RIPETIZIONI):
            idx = next(inferenza.indici_bootstrap(n_test, 1, 20, rng, chunk=1, modulo=len(a)))[0]
            aa, bb = a[idx] + c, b[idx]
            r = inferenza.differenza_bootstrap(aa, bb, np.zeros(n_test), B=B_INTERNO, seed=int(rng.integers(1 << 30)))
            ok_ic += r["sharpe_ic"][0] > 0
            ok_pt += r["sharpe_diff"] > 0
        righe.append({"delta": delta, "potenza_ic": ok_ic / RIPETIZIONI, "quota_stima_positiva": ok_pt / RIPETIZIONI})
        print(f"  {delta:4.2f}      |  {ok_ic / RIPETIZIONI:5.3f}                               |  {ok_pt / RIPETIZIONI:5.3f}")
    pd.DataFrame(righe).to_csv(OUT / "potenza.csv", index=False)


if __name__ == "__main__":
    main()
