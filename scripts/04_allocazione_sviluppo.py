"""Allocazione e benchmark sul solo periodo di sviluppo (fino al 2014-12-31).

Il test non viene toccato. Griglia di 12 configurazioni (q_alto, m, r) per la strategia
multi-scala, per la catena base e per la regola sul solo stato corrente (g); griglia di m
per il vol targeting (d). Selezione: Sharpe netto piu' alto a 5 bp; entro 0,02, il turnover
piu' basso. I primi 504 giorni dal 2005-02-25 sono avviamento (nessuna operativita').
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markov_regime_allocation import motore, multiscala, regimi  # noqa: E402
from markov_regime_allocation.esperimento import GRIGLIA, NUCLEO, esegui, prepara, scegli as scegli_riga  # noqa: E402

SVILUPPO_FINE = "2014-12-31"
M_D = (0.8, 1.0, 1.2)
COSTO = 5.0
OUT = Path(__file__).resolve().parents[1] / "output"


def main():
    mkt, R, rf, w, vol, sigma_rif, dec = prepara(SVILUPPO_FINE)
    inizio = R.index[motore.AVVIAMENTO]
    print(f"Sviluppo: operativita' da {inizio.date()} al {R.index[-1].date()}  sigma_ref={sigma_rif:.4f}")
    vol_mkt = regimi.volatilita_ewma(mkt)

    def p_multi(q):
        return multiscala.probabilita_stress_multiscala(mkt, q_alto=q)[0].reindex(R.index)

    def p_base(q):
        return regimi.probabilita_stress(regimi.stati_as_of(vol_mkt, q_alto=q)).reindex(R.index)

    def p_stato(q):
        st = regimi.stati_as_of(vol_mkt, q_alto=q)
        return (st == 2).astype(float).where(st.notna()).reindex(R.index)

    cache = {}
    for nome, f in (("multi", p_multi), ("base", p_base), ("stato", p_stato)):
        for q in (0.80, 0.90):
            cache[(nome, q)] = f(q)

    righe = []
    uscite = {}
    for nome in ("multi", "base", "stato"):
        for q, m, r in GRIGLIA:
            L = motore.esposizione(cache[(nome, q)], vol, sigma_rif, m, r)
            sim = esegui(R, rf, w, dec, L)
            met = motore.metriche(sim, rf, inizio)
            righe.append({"strategia": nome, "q_alto": q, "m": m, "r": r, **met})
            uscite[(nome, q, m, r)] = (sim, L)
    for m in M_D:
        L = motore.esposizione(pd.Series(0.0, index=R.index), vol, sigma_rif, m, 0.0)
        sim = esegui(R, rf, w, dec, L)
        righe.append({"strategia": "vol_target", "q_alto": np.nan, "m": m, "r": np.nan, **motore.metriche(sim, rf, inizio)})
        uscite[("vol_target", None, m, None)] = (sim, L)
    tabella = pd.DataFrame(righe)
    tabella.to_csv(OUT / "griglia_sviluppo.csv", index=False)

    pd.options.display.float_format = "{:.3f}".format
    print("\nGriglia completa (sviluppo, costi 5 bp):")
    print(tabella.to_string(index=False))
    print("\nEsposizione media massima per strategia (se > 0,98 ovunque la strategia non fa nulla):")
    print(tabella.groupby("strategia")["esposizione_media"].max().round(3).to_string())

    scelte = {n: scegli_riga(tabella, n) for n in ("multi", "base", "stato", "vol_target")}
    print("\nConfigurazioni scelte sullo sviluppo:")
    print(pd.DataFrame(scelte).T.to_string())

    def chiave(n):
        s = scelte[n]
        if n == "vol_target":
            return (n, None, s["m"], None)
        return (n, s["q_alto"], s["m"], s["r"])

    Lm = uscite[chiave("multi")][1]
    for n in ("vol_target", "stato", "base"):
        Ln = uscite[chiave(n)][1]
        c = pd.concat([Lm, Ln], axis=1).loc[inizio:].dropna().corr().iloc[0, 1]
        print(f"Correlazione di L(t) tra multi e {n}: {c:.3f}")

    # benchmark
    bench = {}
    L1 = pd.Series(1.0, index=R.index)
    bench["nucleo_statico"] = esegui(R, rf, w, dec, L1)
    media_L = uscite[chiave("multi")][0]["esposizione"].loc[inizio:].mean()
    bench["esposizione_costante"] = esegui(R, rf, w, dec, pd.Series(media_L, index=R.index))
    mkt_w = pd.DataFrame(np.nan, index=R.index, columns=NUCLEO)
    mkt_w.iloc[motore.AVVIAMENTO] = [0, 1, 0, 0, 0]
    d0 = pd.Series(False, index=R.index)
    d0.iloc[motore.AVVIAMENTO] = True
    bench["mercato_buy_hold"] = motore.simula(R, rf, mkt_w, d0, COSTO)
    mese = R.index.to_period("M")
    dec_m = pd.Series(np.r_[mese[1:] != mese[:-1], False], index=R.index)
    dec_m.iloc[: motore.AVVIAMENTO] = False
    w6040 = pd.DataFrame(0.0, index=R.index, columns=NUCLEO)
    w6040["Mkt"], w6040["Bond7"] = 0.6, 0.4
    bench["60_40"] = motore.simula(R, rf, w6040, dec_m, COSTO)
    # vol targeting a esposizione media uguale (d*): m_d continuo per bisezione, solo sullo sviluppo
    lo, hi = 0.05, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        sim = esegui(R, rf, w, dec, motore.esposizione(pd.Series(0.0, index=R.index), vol, sigma_rif, mid, 0.0))
        if sim["esposizione"].loc[inizio:].mean() > media_L:
            hi = mid
        else:
            lo = mid
    bench["vol_target_esposizione_pari"] = sim
    print(f"\nVol targeting a esposizione pari: m_d={mid:.3f}, esposizione media {sim['esposizione'].loc[inizio:].mean():.3f} (strategia {media_L:.3f})")

    confronto = {n: motore.metriche(b, rf, inizio) for n, b in bench.items()}
    for n in ("multi", "base", "stato", "vol_target"):
        confronto[f"{n} (scelta)"] = motore.metriche(uscite[chiave(n)][0], rf, inizio)
    print("\nConfronto sul periodo di sviluppo (costi 5 bp):")
    print(pd.DataFrame(confronto).T.to_string())
    pd.DataFrame(confronto).T.to_csv(OUT / "confronto_sviluppo.csv")


if __name__ == "__main__":
    main()
