"""Concorrenti del segnale Markov sul solo sviluppo (fino al 2014-12-31): HMM, logistica con memoria, trend.

Stessa regola di esposizione, stessi costi e stesso nucleo della strategia. HMM: griglia di 6 celle (m, r),
perche' non usa q_alto. Logistica: 12 celle. Trend: media a 200 giorni fissa; 100 e 250 solo sensibilita'.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markov_regime_allocation import hmm, logistica, motore, multiscala, trend  # noqa: E402
from markov_regime_allocation.esperimento import GRIGLIA, esegui, prepara, scegli  # noqa: E402

SVILUPPO_FINE = "2014-12-31"
OUT = Path(__file__).resolve().parents[1] / "output"


def main():
    mkt, R, rf, w, vol, sigma_rif, dec = prepara(SVILUPPO_FINE)
    inizio = R.index[motore.AVVIAMENTO]
    righe = []

    p_hmm = hmm.probabilita_stress_hmm(mkt).reindex(R.index)
    print("HMM: probabilita' di stress disponibile da", p_hmm.first_valid_index().date(), "media in sviluppo", round(p_hmm.loc[inizio:].mean(), 3))
    cache_log = {q: logistica.probabilita_stress_logistica(mkt, q_alto=q).reindex(R.index) for q in (0.80, 0.90)}
    print("Logistica: disponibile da", cache_log[0.90].first_valid_index().date())

    uscite = {}
    for q, m, r in GRIGLIA:
        if q == 0.90:  # l'HMM non usa q_alto: una sola copia della griglia (m, r)
            L = motore.esposizione(p_hmm, vol, sigma_rif, m, r)
            sim = esegui(R, rf, w, dec, L)
            righe.append({"strategia": "hmm", "q_alto": float("nan"), "m": m, "r": r, **motore.metriche(sim, rf, inizio)})
            uscite[("hmm", m, r)] = L
        L = motore.esposizione(cache_log[q], vol, sigma_rif, m, r)
        sim = esegui(R, rf, w, dec, L)
        righe.append({"strategia": "logistica", "q_alto": q, "m": m, "r": r, **motore.metriche(sim, rf, inizio)})
        uscite[("logistica", q, m, r)] = L
    tabella = pd.DataFrame(righe)
    tabella.to_csv(OUT / "concorrenti_sviluppo.csv", index=False)
    pd.options.display.float_format = "{:.3f}".format
    print("\nGriglie (sviluppo, 5 bp):")
    print(tabella.to_string(index=False))
    print("\nEsposizione media massima:", tabella.groupby("strategia")["esposizione_media"].max().round(3).to_dict())
    scelte = {n: scegli(tabella, n) for n in ("hmm", "logistica")}
    print("\nScelte sullo sviluppo:")
    print(pd.DataFrame(scelte).T.to_string())

    confronto = {n: {k: v for k, v in s.items() if k in ("sharpe", "rendimento", "volatilita", "drawdown", "turnover_annuo", "esposizione_media")} for n, s in scelte.items()}
    for finestra in (200, 100, 250):
        pt = trend.pesi_trend(R, w, finestra)
        sim = motore.simula(R, rf, pt, dec, 5.0)
        confronto[f"trend {finestra}"] = motore.metriche(sim, rf, inizio)
    print("\nConfronto (sviluppo, 5 bp):")
    print(pd.DataFrame(confronto).T.to_string())
    pd.DataFrame(confronto).T.to_csv(OUT / "confronto_concorrenti_sviluppo.csv")

    Lm = motore.esposizione(multiscala.probabilita_stress_multiscala(mkt, q_alto=0.90)[0].reindex(R.index), vol, sigma_rif, 0.8, 0.25)
    for n, chiave in (("hmm", ("hmm", scelte["hmm"]["m"], scelte["hmm"]["r"])), ("logistica", ("logistica", scelte["logistica"]["q_alto"], scelte["logistica"]["m"], scelte["logistica"]["r"]))):
        c = pd.concat([Lm, uscite[chiave]], axis=1).loc[inizio:].dropna().corr().iloc[0, 1]
        print(f"Correlazione di L(t) tra multi (scelta) e {n}: {c:.3f}")


if __name__ == "__main__":
    main()
