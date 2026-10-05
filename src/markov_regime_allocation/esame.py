"""Costruzione condivisa dell'esame (sviluppo o test): segnali, esposizioni, simulazioni di strategie e controlli.

Usata da scripts/06_inferenza.py, dallo script dei grafici e dal notebook, cosi' tutti calcolano le stesse cose.
Le configurazioni valgono per tutta la serie; le metriche si calcolano solo nella finestra di valutazione.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import dati, hmm, logistica, motore, multiscala, regimi, trend
from .esperimento import config_da_tabelle, esegui, m_per_esposizione, prepara

RADICE = Path(__file__).resolve().parents[2]
OUT = RADICE / "output"
SVILUPPO_FINE = "2014-12-31"
TEST_INIZIO = "2015-01-01"
STRATEGIE = ("multi", "base", "stato", "hmm", "logistica")
NOMI_ORDINATI = ["mercato", "60_40", "nucleo_statico", "esposizione_costante", "vol_target", "vol_target_pari",
                 "trend", "hmm", "logistica", "stato", "base", "multi"]


@dataclass
class Contesto:
    modo: str
    cfg: dict
    mkt: pd.Series
    R: pd.DataFrame
    rf: pd.Series
    w: pd.DataFrame
    vol: pd.Series
    sigma_rif: float
    dec: pd.Series
    inizio_val: pd.Timestamp
    fine_val: pd.Timestamp
    finestra: np.ndarray
    p: dict = field(default_factory=dict)
    stati: pd.Series = None


def configurazioni_da_sviluppo():
    return config_da_tabelle(pd.read_csv(OUT / "griglia_sviluppo.csv"), pd.read_csv(OUT / "concorrenti_sviluppo.csv"))


def segnali(mkt, R, cfg, con_hmm=True):
    p = {}
    p["multi"] = multiscala.probabilita_stress_multiscala(mkt, q_alto=cfg["multi"]["q_alto"])[0].reindex(R.index)
    vol0 = regimi.volatilita_ewma(mkt)
    p["base"] = regimi.probabilita_stress(regimi.stati_as_of(vol0, q_alto=cfg["base"]["q_alto"])).reindex(R.index)
    st = regimi.stati_as_of(vol0, q_alto=cfg["stato"]["q_alto"])
    p["stato"] = (st == 2).astype(float).where(st.notna()).reindex(R.index)
    if con_hmm:
        p["hmm"] = hmm.probabilita_stress_hmm(mkt).reindex(R.index)
    p["logistica"] = logistica.probabilita_stress_logistica(mkt, q_alto=cfg["logistica"]["q_alto"]).reindex(R.index)
    return p, regimi.stati_as_of(vol0, q_alto=0.90)


def prepara_esame(modo, cfg, con_hmm=True):
    if modo == "test":
        fine_dati, inizio_val, fine_val = dati.FINE, TEST_INIZIO, dati.FINE
    else:
        fine_dati, inizio_val, fine_val = SVILUPPO_FINE, None, SVILUPPO_FINE
    mkt, R, rf, w, vol, sigma_rif, dec = prepara(fine_dati)
    if inizio_val is None:
        inizio_val = R.index[motore.AVVIAMENTO]
    inizio_val, fine_val = pd.Timestamp(inizio_val), pd.Timestamp(fine_val)
    finestra = ((R.index >= inizio_val) & (R.index <= fine_val))
    assert not R[finestra].isna().any().any() and not rf[finestra].isna().any(), "valori mancanti nei dati della finestra"
    ctx = Contesto(modo, cfg, mkt, R, rf, w, vol, sigma_rif, dec, inizio_val, fine_val, finestra)
    ctx.p, ctx.stati = segnali(mkt, R, cfg, con_hmm)
    return ctx


def esposizione_strategia(ctx, nome):
    c = ctx.cfg[nome]
    return motore.esposizione(ctx.p[nome], ctx.vol, ctx.sigma_rif, c["m"], c["r"])


def esposizione_vol_target(ctx):
    return motore.esposizione(pd.Series(0.0, index=ctx.R.index), ctx.vol, ctx.sigma_rif, ctx.cfg["vol_target"]["m"], 0.0)


def simula_tutti(ctx, costo):
    R, rf, w, dec = ctx.R, ctx.rf, ctx.w, ctx.dec
    s = {n: esegui(R, rf, w, dec, esposizione_strategia(ctx, n), costo) for n in STRATEGIE if n in ctx.p}
    s["vol_target"] = esegui(R, rf, w, dec, esposizione_vol_target(ctx), costo)
    media = s["multi"]["esposizione"][ctx.finestra].mean()
    s["esposizione_costante"] = esegui(R, rf, w, dec, pd.Series(media, index=R.index), costo)
    s["nucleo_statico"] = esegui(R, rf, w, dec, pd.Series(1.0, index=R.index), costo)
    md, sd = m_per_esposizione(R, rf, w, ctx.vol, ctx.sigma_rif, dec, media, ctx.inizio_val, ctx.fine_val, costo)
    s["vol_target_pari"] = sd
    s["_m_d_pari"] = md
    s["trend"] = motore.simula(R, rf, trend.pesi_trend(R, w, ctx.cfg["trend_finestra"]), dec, costo)
    return s


def benchmark_semplici(ctx, costo=5.0):
    R, rf = ctx.R, ctx.rf
    pos0 = int(np.argmax(ctx.finestra))
    w_mkt = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    w_mkt.iloc[pos0] = [0, 1, 0, 0, 0]
    d0 = pd.Series(False, index=R.index)
    d0.iloc[pos0] = True
    mese = R.index.to_period("M")
    dec_m = pd.Series(np.r_[mese[1:] != mese[:-1], False], index=R.index)
    dec_m.iloc[: motore.AVVIAMENTO] = False
    w64 = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    w64["Mkt"], w64["Bond7"] = 0.6, 0.4
    return {"mercato": motore.simula(R, rf, w_mkt, d0, costo), "60_40": motore.simula(R, rf, w64, dec_m, costo)}


def simulazioni_complete(ctx, costo=5.0):
    s = simula_tutti(ctx, costo)
    s.update(benchmark_semplici(ctx, costo))
    return s


def tabella_metriche(ctx, sims):
    return pd.DataFrame({n: motore.metriche(sims[n].loc[ctx.finestra], ctx.rf.loc[ctx.finestra]) for n in NOMI_ORDINATI}).T
