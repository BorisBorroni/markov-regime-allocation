import numpy as np
import pandas as pd

from markov_regime_allocation import multiscala as ms


def serie(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 200) % 2 == 0, 0.006, 0.02)
    return pd.Series(rng.normal(0, vol), index=pd.bdate_range("2000-01-03", periods=n))


def test_lambda_da_memoria():
    assert abs(ms.lambda_da_memoria(15) - 0.875) < 1e-12


def test_pesi_sommano_a_uno_e_sono_positivi():
    r = serie()
    _, w = ms.previsione_multiscala(r)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w >= 0).all().all()


def test_pesi_uguali_se_le_perdite_sono_uguali():
    r = serie()
    f = pd.DataFrame({"a": r.abs() ** 2 + 1e-4, "b": r.abs() ** 2 + 1e-4}, index=r.index)
    w = ms.pesi_adattivi(r, f)
    assert np.allclose(w["a"], 0.5)


def test_peso_alla_previsione_migliore():
    r = serie()
    vero = pd.Series(r.pow(2).rolling(5).mean().shift(-5)).bfill().ffill().clip(lower=1e-8)
    f = pd.DataFrame({"buona": vero, "cattiva": vero * 5}, index=r.index)
    w = ms.pesi_adattivi(r, f)
    assert w["buona"].iloc[-200] > 0.9


def test_nessun_uso_del_futuro():
    r = serie()
    T = 1000
    comb1, w1 = ms.previsione_multiscala(r)
    r2 = r.copy()
    r2.iloc[T + 1:] = r2.iloc[T + 1:] * 7 + 0.05
    comb2, w2 = ms.previsione_multiscala(r2)
    assert np.allclose(comb1.iloc[: T + 1].dropna(), comb2.iloc[: T + 1].dropna())
    assert np.allclose(w1.iloc[: T + 1], w2.iloc[: T + 1])
