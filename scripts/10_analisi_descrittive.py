"""Analisi descrittive della parte previsiva, su sviluppo (fino al 2014) e test (dal 2015), separatamente.

Nessun parametro viene scelto qui: le previsioni sono quelle fissate in anticipo e usano solo il passato.
Domande: la multi-scala prevede la varianza meglio della catena base, di EWMA e HAR? La probabilita' di
stress batte persistenza e frequenza? Il primo ordine di Markov basta? I regimi durano come previsto dalla catena?
Risultati salvati in output/descrittive.txt e output/descrittive.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markov_regime_allocation import dati, multiscala as ms, previsione as pv, regimi  # noqa: E402

RADICE = Path(__file__).resolve().parents[1]
H = regimi.ORIZZONTE
SVILUPPO_FINE = pd.Timestamp("2014-12-31")


def segnali(r):
    """Previsioni e perdite calcolate una volta sull'intera serie (ognuna usa solo il passato)."""
    reale = pv.varianza_futura(r, H)
    vol = regimi.volatilita_ewma(r)
    stati = regimi.stati_as_of(vol)
    comb, _ = ms.previsione_multiscala(r)
    prev = {"multi": comb, "base": pv.previsione_catena(r, stati, H),
            "ewma": vol ** 2 / 252, "har": pv.previsione_har(r, H)}
    perdite = pd.concat({k: pv.qlike(reale, v.clip(lower=pv.FLOOR)) for k, v in prev.items()}, axis=1)
    pm, _, _, esito = ms.probabilita_stress_multiscala(r)
    st_rif = regimi.stati_as_of(regimi.volatilita_ewma(r, ms.lambda_da_memoria(ms.SCALA_RIFERIMENTO)))
    prob = {"multi": pm, "persistenza": (st_rif == 2).astype(float).where(st_rif.notna()),
            "frequenza": pv.frequenza_stress_espandibile(st_rif)}
    return perdite, prob, esito, stati


def blocco(nome, sel, sel_stati, perdite, prob, esito, stati, out):
    print(f"\n===== {nome}: {sel.sum()} giorni di borsa =====")
    pe = perdite[sel].dropna()
    print(f"QLIKE medio su {len(pe)} giorni (piu' basso = meglio)")
    res = {"qlike": pe.mean().round(4).to_dict(), "dm": {}}
    for k, v in res["qlike"].items():
        print(f"  {k:6s} {v:.4f}")
    print("Differenza multi - concorrente (negativa = multi meglio), t Newey-West con 2h ritardi:")
    for k in ("base", "ewma", "har"):
        d, t = pv.diebold_mariano(pe["multi"], pe[k], 2 * H)
        res["dm"][k] = {"diff": round(float(d), 4), "t": round(float(t), 2)}
        print(f"  contro {k:5s} diff={d:+.4f}  t={t:+.2f}")
    ev = esito[sel].dropna()
    print("Probabilita' di stress a 5 giorni (evento: stress della scala 15):")
    res["stress"] = {}
    for k, q in prob.items():
        i = ev.index.intersection(q[sel].dropna().index)
        b, ls = pv.brier(q[i], ev[i]).mean(), pv.log_score(q[i], ev[i]).mean()
        res["stress"][k] = {"brier": round(float(b), 4), "log": round(float(ls), 4)}
        print(f"  {k:12s} Brier={b:.4f}  log-score={ls:.4f}")
    s = stati[sel_stati].dropna()
    lr, gdl, p = pv.test_ordine_markov(s)
    res["ordine"] = {"lr": round(float(lr), 1), "gdl": gdl, "p": float(p)}
    print(f"Ordine 1 contro ordine 2 (rapporto di verosimiglianza): LR={lr:.1f}, gdl={gdl}, p={p:.2g}")
    reale_s, atteso = pv.sopravvivenza_regimi(s), pv.sopravvivenza_geometrica(s)
    print("Quota di periodi con durata >= 5/20/60 giorni, osservata contro attesa dalla catena:")
    for j, nm in enumerate(("calma", "normale", "stress")):
        print(f"  {nm:8s} osservata {reale_s.loc[j].round(3).tolist()}  attesa {atteso.loc[j].round(3).tolist()}")
    res["durate"] = {"osservata": reale_s.round(4).values.tolist(), "attesa": atteso.round(4).values.tolist()}
    out[nome] = res


def main():
    r = dati.carica_serie()["Mkt"].dropna()
    perdite, prob, esito, stati = segnali(r)
    op = r.index >= dati.INIZIO_OPERATIVO
    # l'ultimo giorno valutabile dello sviluppo e' h giorni prima della sua fine: il bersaglio guarda avanti di h giorni
    ultimo_sviluppo = r.index[r.index <= SVILUPPO_FINE][-1 - H]
    sviluppo = op & (r.index <= ultimo_sviluppo)
    stati_sviluppo = op & (r.index <= SVILUPPO_FINE)  # gli stati non guardano avanti: tutto lo sviluppo
    test = r.index > SVILUPPO_FINE
    out = {}
    blocco("sviluppo", pd.Series(sviluppo, r.index), pd.Series(stati_sviluppo, r.index), perdite, prob, esito, stati, out)
    blocco("test", pd.Series(test, r.index), pd.Series(test, r.index), perdite, prob, esito, stati, out)
    (RADICE / "output" / "descrittive.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
