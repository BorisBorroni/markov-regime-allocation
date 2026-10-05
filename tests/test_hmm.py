import numpy as np
import pandas as pd

from markov_regime_allocation import hmm


def sintetica(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    A = np.array([[0.98, 0.02, 0.0], [0.02, 0.96, 0.02], [0.0, 0.05, 0.95]])
    sd = np.array([0.004, 0.009, 0.025])
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = rng.choice(3, p=A[s[t - 1]])
    return pd.Series(rng.normal(0, sd[s]), index=pd.bdate_range("1996-01-02", periods=n)), s


def test_em_recupera_le_volatilita_e_ordina_per_varianza():
    r, _ = sintetica()
    par = hmm.stima_em(r.to_numpy())
    assert np.all(np.diff(par["varianze"]) > 0)
    assert np.allclose(np.sqrt(par["varianze"]), [0.004, 0.009, 0.025], rtol=0.25)
    assert np.allclose(par["A"].sum(axis=1), 1.0)


def test_filtro_riconosce_lo_stato_di_stress():
    r, s = sintetica()
    par = hmm.stima_em(r.to_numpy())
    alfa = hmm.filtro_avanti(r.to_numpy(), par)
    assert alfa[s == 2, 2].mean() > 0.6
    assert alfa[s == 0, 2].mean() < 0.1


def test_probabilita_valide_e_nessun_uso_del_futuro():
    r, _ = sintetica(2200)
    p1 = hmm.probabilita_stress_hmm(r, primo_riaddestramento="1999-01-01")
    T = 1800
    r2 = r.copy()
    r2.iloc[T + 1:] = r2.iloc[T + 1:] * 8 + 0.02
    p2 = hmm.probabilita_stress_hmm(r2, primo_riaddestramento="1999-01-01")
    ok = p1.dropna()
    assert ((ok >= 0) & (ok <= 1)).all()
    assert np.allclose(p1.iloc[: T + 1].dropna(), p2.iloc[: T + 1].dropna(), atol=1e-9)
