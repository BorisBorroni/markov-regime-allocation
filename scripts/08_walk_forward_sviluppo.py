"""Robustezza: ri-selezione annuale della configurazione su finestra espandibile, solo sviluppo.

Per ogni anno Y dal 2009 al 2014 si sceglie la cella della griglia (stesso criterio: Sharpe netto piu' alto, entro 0,02
turnover minimo) usando solo i rendimenti dal primo giorno operativo al 31 dicembre di Y-1, e si usano i rendimenti di
quella cella nell'anno Y. Semplificazione dichiarata: i rendimenti di ogni cella vengono dalla simulazione continua
della cella; il costo dell'eventuale passaggio da una cella all'altra non e' incluso.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markov_regime_allocation import inferenza, motore, multiscala  # noqa: E402
from markov_regime_allocation.esperimento import GRIGLIA, esegui, prepara  # noqa: E402

SVILUPPO_FINE = "2014-12-31"


def main():
    mkt, R, rf, w, vol, sigma_rif, dec = prepara(SVILUPPO_FINE)
    inizio = R.index[motore.AVVIAMENTO]
    p = {q: multiscala.probabilita_stress_multiscala(mkt, q_alto=q)[0].reindex(R.index) for q in (0.80, 0.90)}
    sims = {cella: esegui(R, rf, w, dec, motore.esposizione(p[cella[0]], vol, sigma_rif, cella[1], cella[2])) for cella in GRIGLIA}
    vt = esegui(R, rf, w, dec, motore.esposizione(pd.Series(0.0, index=R.index), vol, sigma_rif, 0.8, 0.0))
    scelte, pezzi, pezzi_vt, pezzi_fisso = {}, [], [], []
    fissa = (0.90, 0.8, 0.25)
    for anno in range(2009, 2015):
        passato = (R.index >= inizio) & (R.index < pd.Timestamp(year=anno, month=1, day=1))
        righe = []
        for cella, s in sims.items():
            m = motore.metriche(s.loc[passato], rf.loc[passato])
            righe.append({"cella": cella, "sharpe": m["sharpe"], "turnover": m["turnover_annuo"]})
        t = pd.DataFrame(righe)
        buone = t[t["sharpe"] >= t["sharpe"].max() - 0.02]
        scelta = buone.sort_values("turnover").iloc[0]["cella"]
        scelte[anno] = scelta
        anno_idx = R.index.year == anno
        pezzi.append(sims[scelta]["ritorno"][anno_idx])
        pezzi_vt.append(vt["ritorno"][anno_idx])
        pezzi_fisso.append(sims[fissa]["ritorno"][anno_idx])
    print("Celle scelte (q_alto, m, r) per anno:")
    for a, c in scelte.items():
        print(f"  {a}: {c}")
    wf, vtr, fx = pd.concat(pezzi), pd.concat(pezzi_vt), pd.concat(pezzi_fisso)
    rfw = rf.reindex(wf.index).to_numpy()
    for nome, serie in (("walk-forward", wf), ("cella fissa scelta su tutto lo sviluppo", fx), ("vol targeting m=0,8", vtr)):
        e = serie.to_numpy() - rfw
        valore = (1 + serie).cumprod()
        print(f"{nome:42s} Sharpe {inferenza.sharpe(e):.3f}  drawdown {(valore / valore.cummax() - 1).min():.3f}")
    print("Nota: gli anni 2009-2014 non coincidono con tutta la finestra di sviluppo (2007-2014); la cella fissa e' stata scelta usando anche questi anni.")


if __name__ == "__main__":
    main()
