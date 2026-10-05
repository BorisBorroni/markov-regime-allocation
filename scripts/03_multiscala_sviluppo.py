"""Variante multi-scala contro catena base e concorrenti semplici, solo sviluppo (fino al 2014-12-31).

Regola fissata in anticipo: la variante passa alla fase di allocazione solo se ha QLIKE
medio inferiore alla catena base e la differenza ha t di Diebold-Mariano <= -2.
Gli iperparametri (ETA, SCONTO) non vengono ottimizzati: la tabella di sensibilita' e' solo informativa.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markov_regime_allocation import dati, multiscala as ms, previsione as pv, regimi  # noqa: E402

SVILUPPO_FINE = "2014-12-31"
H = regimi.ORIZZONTE


def main():
    r = dati.carica_serie(fine=SVILUPPO_FINE)["Mkt"].dropna()
    op = r.index >= dati.INIZIO_OPERATIVO
    reale = pv.varianza_futura(r, H)
    vol = regimi.volatilita_ewma(r)
    stati = regimi.stati_as_of(vol)
    comb, w = ms.previsione_multiscala(r)
    per_scala = ms.previsioni_per_scala(r)
    prev = {
        "base": pv.previsione_catena(r, stati, H),
        "multi": comb,
        "ewma": vol ** 2 / 252,
        "har": pv.previsione_har(r, H),
    }
    for n in per_scala:
        prev[f"scala{n}"] = per_scala[n]
    perdite = {k: pv.qlike(reale, v.clip(lower=pv.FLOOR))[op] for k, v in prev.items()}
    comune = pd.concat(perdite, axis=1).dropna().index
    print(f"QLIKE medio su {len(comune)} giorni di sviluppo (piu' basso = meglio)")
    for k, v in perdite.items():
        print(f"  {k:8s} {v[comune].mean():.4f}")
    print("\nDifferenza multi - concorrente (negativa = multi meglio), t Newey-West con 2h ritardi:")
    for k in ("base", "ewma", "har", "scala5", "scala15", "scala42"):
        d, t = pv.diebold_mariano(perdite["multi"][comune], perdite[k][comune], 2 * H)
        print(f"  contro {k:8s} diff={d:+.4f}  t={t:+.2f}")
    print("\nPeso medio per scala (dev):", w[op].mean().round(3).to_dict())

    print("\nSensibilita' agli iperparametri (QLIKE medio della multi-scala, solo informativa):")
    for eta in (5.0, 10.0, 20.0):
        for sconto in (0.98, 0.99, 0.995):
            pesi = ms.pesi_adattivi(r, per_scala, H, eta, sconto)
            c = (per_scala * pesi).sum(axis=1, min_count=1)
            q = pv.qlike(reale, c.clip(lower=pv.FLOOR))[op].reindex(comune)
            print(f"  eta={eta:4.0f} sconto={sconto:.3f}  {q.mean():.4f}")

    print("\nProbabilita' di stress a", H, "giorni sull'evento comune (stress della scala 15): Brier e log-score medi")
    pm, wm, pn, esito = ms.probabilita_stress_multiscala(r)
    st_rif = regimi.stati_as_of(regimi.volatilita_ewma(r, ms.lambda_da_memoria(ms.SCALA_RIFERIMENTO)))
    candidate = {"multi": pm, "persistenza": (st_rif == 2).astype(float).where(st_rif.notna()),
                 "frequenza": pv.frequenza_stress_espandibile(st_rif)}
    for n in pn:
        candidate[f"catena{n}"] = pn[n]
    idx = esito[op].dropna().index
    for k, q in candidate.items():
        i = idx.intersection(q.dropna().index)
        print(f"  {k:12s} Brier={pv.brier(q[i], esito[i]).mean():.4f}  log={pv.log_score(q[i], esito[i]).mean():.4f}")
    print("Peso medio per scala (stress):", wm[op].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
