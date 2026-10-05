"""Regimi, matrice di transizione e previsione della volatilita': solo periodo di sviluppo.

I dati vengono tagliati al 2014-12-31 prima di qualsiasi calcolo: il test non viene toccato.
Stampa: conteggio degli stati, matrice di transizione, test sull'ordine della catena,
sopravvivenza dei regimi, confronto QLIKE tra catena e previsioni semplici (H1), e
confronto tra probabilita' di stress della catena e due riferimenti semplici.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markov_regime_allocation import dati, previsione as pv, regimi  # noqa: E402

SVILUPPO_FINE = "2014-12-31"
H = regimi.ORIZZONTE


def main():
    serie = dati.carica_serie(fine=SVILUPPO_FINE)
    r = serie["Mkt"].dropna()
    vol = regimi.volatilita_ewma(r)
    stati = regimi.stati_as_of(vol)
    op = stati.index >= dati.INIZIO_OPERATIVO

    print("Stati (0 calma, 1 normale, 2 stress), dal", dati.INIZIO_OPERATIVO, "al", SVILUPPO_FINE)
    print(stati[op].value_counts(normalize=True).sort_index().round(3).to_string())

    P = regimi.matrici_transizione(stati)[-1]
    print("\nMatrice di transizione stimata a fine sviluppo (righe: stato oggi):")
    print(pd.DataFrame(P).round(3).to_string())

    lr, gdl, p = pv.test_ordine_markov(stati[op])
    print(f"\nTest ordine 1 contro ordine 2: LR={lr:.1f}, gradi={gdl}, p-value={p:.3g}")
    print("\nQuota di periodi con durata almeno k giorni (ultimo periodo escluso):")
    print(pv.sopravvivenza_regimi(stati[op]).round(3).to_string())

    # H1: previsione della varianza media dei prossimi H giorni
    reale = pv.varianza_futura(r, H)
    prev = {
        "catena": pv.previsione_catena(r, stati, H),
        "ewma": vol ** 2 / 252,
        "har": pv.previsione_har(r, H),
        "media": pv.previsione_media_espandibile(r),
    }
    perdite = {k: pv.qlike(reale, v.clip(lower=pv.FLOOR))[op] for k, v in prev.items()}
    comune = pd.concat(perdite, axis=1).dropna().index
    print(f"\nH1: QLIKE medio su {len(comune)} giorni (piu' basso = meglio)")
    for k, v in perdite.items():
        print(f"  {k:7s} {v[comune].mean():.4f}")
    print("Differenza catena - concorrente (negativa = catena meglio), t Newey-West con 2h ritardi:")
    for k in ("ewma", "har", "media"):
        d, t = pv.diebold_mariano(perdite["catena"][comune], perdite[k][comune], 2 * H)
        print(f"  contro {k:6s} diff={d:+.4f}  t={t:+.2f}  p={pv.p_value_normale(t):.3f}")

    # Probabilita' di stress: catena contro frequenza e persistenza
    esito = (stati.shift(-H) == 2).astype(float).where(stati.shift(-H).notna())
    pc = regimi.probabilita_stress(stati)
    freq = pv.frequenza_stress_espandibile(stati)
    pers = (stati == 2).astype(float)
    print("\nProbabilita' di stress a", H, "giorni: Brier medio e log-score medio")
    for nome, q in (("catena", pc), ("frequenza", freq), ("persistenza", pers)):
        idx = esito[op].dropna().index.intersection(q.dropna().index)
        print(f"  {nome:12s} Brier={pv.brier(q[idx], esito[idx]).mean():.4f}  log={pv.log_score(q[idx], esito[idx]).mean():.4f}")


if __name__ == "__main__":
    main()
