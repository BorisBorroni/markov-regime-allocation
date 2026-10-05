import math

import numpy as np
import pandas as pd

from markov_regime_allocation import inferenza as inf


def test_indici_bootstrap_in_intervallo_e_blocco_medio():
    rng = np.random.default_rng(0)
    n = 500
    idx = next(inf.indici_bootstrap(n, 400, 20, rng))
    assert idx.min() >= 0 and idx.max() < n
    salti = (np.diff(idx, axis=1) != 1) & (np.diff(idx, axis=1) != -(n - 1))
    lunghezza_media = 1 / salti.mean()
    assert 15 < lunghezza_media < 26


def test_bootstrap_con_modulo_diverso_non_sovracampiona_l_inizio():
    rng = np.random.default_rng(9)
    idx = next(inf.indici_bootstrap(3000, 200, 20, rng, chunk=200, modulo=1000))
    assert idx.max() < 1000
    frequenze = np.bincount(idx.ravel(), minlength=1000) / idx.size * 1000
    assert frequenze[:300].mean() < 1.15 and frequenze[700:].mean() > 0.85


def test_differenza_nulla_se_identiche_e_ic_esclude_zero_se_migliore():
    rng = np.random.default_rng(1)
    n = 1500
    b = rng.normal(0.0003, 0.01, n)
    rf = np.zeros(n)
    r0 = inf.differenza_bootstrap(b, b, rf, B=300)
    assert r0["sharpe_diff"] == 0 and r0["sharpe_ic"] == (0.0, 0.0)
    a = b + 0.0008
    r1 = inf.differenza_bootstrap(a, b, rf, B=500)
    assert r1["sharpe_diff"] > 0.8 and r1["sharpe_ic"][0] > 0


def test_ic_include_zero_senza_vantaggio():
    rng = np.random.default_rng(2)
    n = 1500
    b = rng.normal(0.0003, 0.01, n)
    a = b + rng.normal(0, 0.002, n)
    r = inf.differenza_bootstrap(a, b, np.zeros(n), B=500)
    assert r["sharpe_ic"][0] < 0 < r["sharpe_ic"][1]


def test_drawdown_massimo_a_mano():
    assert math.isclose(inf.drawdown_massimo(np.array([0.1, -0.5, 0.2])), -0.5)
    assert math.isclose(inf.drawdown_massimo(np.array([-0.05, 0.01, 0.01, 0.0])), -0.05)


def utilita(r, k):
    return r.mean() - k * (r ** 2).mean()


def test_commissione_zero_se_uguali_e_risolve_l_equazione():
    rng = np.random.default_rng(3)
    b = rng.normal(0.0003, 0.01, 2000)
    assert abs(inf.commissione_performance(b, b, 5)) < 1e-12
    meno_rischio = 0.5 * b + 0.5 * b.mean()  # stessa media, meta' volatilita'
    f1, f10 = (inf.commissione_performance(meno_rischio, b, g) for g in (1, 10))
    assert 0 < f1 < f10  # chi ha piu' avversione al rischio paga di piu'
    a = b + 0.0002
    for gamma in (1, 5, 10):
        fee = inf.commissione_performance(a, b, gamma) / 252
        k = gamma / (2 * (1 + gamma))
        assert abs(utilita(1 + a - fee, k) - utilita(1 + b, k)) < 1e-12
        assert 0 < fee < 0.0003


def test_deflated_sharpe_propriet():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0005, 0.01, 2500)
    d1 = inf.deflated_sharpe(x, 1, 0.0)
    d100 = inf.deflated_sharpe(x, 100, 1e-5)
    assert d1 > d100
    assert inf.deflated_sharpe(x + 0.0005, 100, 1e-5) > d100
    assert inf.sharpe_atteso_massimo(1, 1.0) == 0.0
    assert inf.sharpe_atteso_massimo(1000, 1e-5) > inf.sharpe_atteso_massimo(10, 1e-5)


def test_ols_newey_west_recupera_beta_e_hac_non_minore_in_presenza_di_autocorrelazione():
    rng = np.random.default_rng(5)
    n = 3000
    x = rng.normal(size=n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.6 * e[t - 1] + rng.normal()
    y = 1.0 + 2.0 * x + e
    beta, se0 = inf.ols_newey_west(y, x, 0)
    _, se20 = inf.ols_newey_west(y, x, 20)
    assert np.allclose(beta, [1.0, 2.0], atol=0.15)
    assert se20[0] > se0[0]


def test_alfa_contro_nucleo():
    rng = np.random.default_rng(6)
    n = 2500
    nucleo = rng.normal(0.0003, 0.01, n)
    strat = 0.5 * nucleo + 0.0002 + rng.normal(0, 0.001, n)
    r = inf.alfa_contro_nucleo(strat, nucleo, np.zeros(n), 20)
    assert abs(r["beta"] - 0.5) < 0.05 and abs(r["alfa_annuo"] - 0.0002 * 252) < 0.02


def test_tabella_per_regime_usa_lo_stato_del_giorno_prima():
    idx = pd.bdate_range("2020-01-01", periods=6)
    stati = pd.Series([0, 0, 2, 2, 1, 1], index=idx, dtype=float)
    r = pd.Series([0.0, 0.01, 0.0, -0.02, 0.0, 0.0], index=idx)
    t = inf.tabella_per_regime(r, stati, pd.Series(0.0, index=idx))
    assert math.isclose(t.loc[0, "eccesso_annuo"], 0.005 * 252)  # giorni 1 e 2 (stato 0 il giorno prima): 0.01 e 0.0
    assert math.isclose(t.loc[2, "eccesso_annuo"], -0.01 * 252)  # giorni 3 e 4: 0.0 e -0.02
    assert math.isclose(t["quota_giorni"].sum(), 1.0)
