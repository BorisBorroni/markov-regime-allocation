"""Prova di fumo su dati sintetici: il modulo dell'esame e i grafici girano e rispettano le proprieta' di base."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
from markov_regime_allocation import (  # noqa: E402
    esame,
    grafici,
    motore,
    multiscala,
    regimi,
)
from markov_regime_allocation.esperimento import GRIGLIA, NUCLEO  # noqa: E402,F401


@pytest.fixture(scope="module")
def contesto():
    rng = np.random.default_rng(5)
    n = 1700
    giorni = pd.bdate_range("2005-01-03", periods=n)
    vol = 0.006 + 0.008 * (np.sin(np.arange(n) / 90) > 0.6)
    R = pd.DataFrame(rng.normal(0.0003, 1, (n, 5)) * vol[:, None], index=giorni, columns=NUCLEO)
    rf = pd.Series(0.00005, index=giorni)
    mkt = R["Mkt"]
    w = motore.pesi_nucleo(R)
    v = motore.volatilita_nucleo(R, w)
    dec = motore.giorni_decisione(R.index)
    dec.iloc[: motore.AVVIAMENTO] = False
    cfg = {"multi": {"m": 1.0, "q_alto": 0.9, "r": 0.5}, "base": {"m": 1.0, "q_alto": 0.9, "r": 0.5},
           "stato": {"m": 1.0, "q_alto": 0.9, "r": 0.5}, "logistica": {"m": 1.0, "q_alto": 0.9, "r": 0.5},
           "hmm": {"m": 1.0, "q_alto": None, "r": 0.5}, "vol_target": {"m": 1.0, "q_alto": None, "r": None},
           "trend_finestra": 200}
    finestra = np.asarray(R.index >= giorni[1000])
    ctx = esame.Contesto("sviluppo", cfg, mkt, R, rf, w, v, motore.sigma_riferimento(v), dec,
                         giorni[1000], giorni[-1], finestra)
    ctx.p, ctx.stati = esame.segnali(mkt, R, cfg, con_hmm=False)
    return ctx


def test_simulazioni_complete_sono_coerenti(contesto):
    sims = esame.simulazioni_complete(contesto, 5.0)
    assert set(esame.NOMI_ORDINATI) - {"hmm"} <= set(sims)
    for n in esame.NOMI_ORDINATI:
        if n in sims:
            assert not sims[n]["ritorno"][contesto.finestra].isna().any()
            assert (sims[n]["esposizione"] <= 1 + 1e-9).all()
    media = sims["multi"]["esposizione"][contesto.finestra].mean()
    assert np.isclose(sims["esposizione_costante"]["esposizione"][contesto.finestra].iloc[10:].mean(), media, atol=0.02)
    assert sims["nucleo_statico"]["esposizione"][contesto.finestra].iloc[10:].min() > 0.95


def test_grafici_si_costruiscono(contesto):
    sims = esame.simulazioni_complete(contesto, 5.0)
    w = multiscala.probabilita_stress_multiscala(contesto.mkt)[1].reindex(contesto.R.index)
    stati = regimi.stati_as_of(regimi.volatilita_ewma(contesto.mkt)).reindex(contesto.R.index)
    boot = {n: {"sharpe_diff": 0.0, "sharpe_ic": [-0.1, 0.1], "drawdown_diff": 0.0, "drawdown_ic": [-0.01, 0.01]}
            for n in ("vol_target_pari", "base", "stato", "hmm", "logistica", "trend", "nucleo_statico", "mercato", "60_40")}
    figure = [grafici.fig_equity(sims, contesto.finestra), grafici.fig_drawdown(sims, contesto.finestra),
              grafici.fig_esposizione(sims, stati, contesto.finestra), grafici.fig_pesi_scale(w, contesto.finestra),
              grafici.fig_forest(boot),
              grafici.fig_potenza(pd.DataFrame({"delta": [0, 0.1], "potenza_ic": [0.01, 0.4]})),
              grafici.fig_regimi(regimi.matrici_transizione(stati)[-1], contesto.vol, stati, contesto.finestra)]
    assert all(len(f.axes) >= 1 for f in figure)
    grafici.plt.close("all")
