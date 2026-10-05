"""Funzioni comuni agli script di allocazione: preparazione dei dati, esecuzione, selezione sulla griglia."""
import itertools

import pandas as pd

from . import dati, motore

NUCLEO = ["NoDur", "Mkt", "Chips", "Bond7", "Oro"]
GRIGLIA = list(itertools.product((0.80, 0.90), (0.8, 1.0, 1.2), (0.25, 0.50)))
COSTO = 5.0


def prepara(fine):
    s = dati.carica_serie(fine=fine)
    mkt = s["Mkt"].dropna()
    R = s.loc[s.index > dati.INIZIO_OPERATIVO, NUCLEO]
    rf = s["RF"].reindex(R.index)
    w = motore.pesi_nucleo(R)
    vol = motore.volatilita_nucleo(R, w)
    sigma_rif = motore.sigma_riferimento(vol)
    dec = motore.giorni_decisione(R.index)
    dec.iloc[: motore.AVVIAMENTO] = False
    return mkt, R, rf, w, vol, sigma_rif, dec


def esegui(R, rf, w, dec, L, costo=COSTO):
    return motore.simula(R, rf, w.mul(L, axis=0), dec, costo)


def scegli(tabella, nome):
    """Sharpe netto piu' alto; entro 0,02 dal massimo, il turnover piu' basso."""
    t = tabella[tabella["strategia"] == nome]
    buone = t[t["sharpe"] >= t["sharpe"].max() - 0.02]
    return buone.sort_values("turnover_annuo").iloc[0]


def config_da_tabelle(griglia, concorrenti):
    """Configurazioni scelte sullo sviluppo (stesso criterio per tutte) da due tabelle di griglia."""
    def riga(tab, nome):
        s = scegli(tab, nome)
        return {k: (None if pd.isna(v) else float(v)) for k, v in s.items() if k in ("q_alto", "m", "r")}

    return {
        "multi": riga(griglia, "multi"), "base": riga(griglia, "base"), "stato": riga(griglia, "stato"),
        "vol_target": riga(griglia, "vol_target"), "hmm": riga(concorrenti, "hmm"),
        "logistica": riga(concorrenti, "logistica"), "trend_finestra": 200,
    }


def m_per_esposizione(R, rf, w, vol, sigma_rif, dec, esposizione_obiettivo, inizio, fine, costo=COSTO):
    """m del vol targeting (r = 0) tale che l'esposizione media nella finestra [inizio, fine] sia pari all'obiettivo (bisezione)."""
    lo, hi = 0.05, 3.0
    sim = None
    for _ in range(40):
        mid = (lo + hi) / 2
        L = motore.esposizione(pd.Series(0.0, index=R.index), vol, sigma_rif, mid, 0.0)
        sim = esegui(R, rf, w, dec, L, costo)
        if sim["esposizione"].loc[inizio:fine].mean() > esposizione_obiettivo:
            hi = mid
        else:
            lo = mid
    return mid, sim
