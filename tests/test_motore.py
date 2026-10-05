import numpy as np
import pandas as pd

from markov_regime_allocation import motore as mo


def dati(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    R = pd.DataFrame(rng.normal(0.0004, [0.005, 0.01, 0.02], size=(n, 3)), index=idx, columns=list("abc"))
    return R


def test_pesi_nucleo_sommano_a_uno_e_penalizzano_la_volatilita():
    R = dati()
    w = mo.pesi_nucleo(R).dropna()
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w["a"] > w["b"]).all() and (w["b"] > w["c"]).all()


def test_volatilita_nucleo_a_mano():
    R = dati()
    w = mo.pesi_nucleo(R)
    v = mo.volatilita_nucleo(R, w)
    t = 200
    a_mano = (R.iloc[t - 59:t + 1].to_numpy() @ w.iloc[t].to_numpy()).std(ddof=1) * np.sqrt(252)
    assert abs(v.iloc[t] - a_mano) < 1e-12


def test_nessun_uso_del_futuro_nel_nucleo():
    R = dati()
    T = 250
    w1 = mo.pesi_nucleo(R)
    v1 = mo.volatilita_nucleo(R, w1)
    R2 = R.copy()
    R2.iloc[T + 1:] = R2.iloc[T + 1:] * 9 + 0.03
    w2 = mo.pesi_nucleo(R2)
    v2 = mo.volatilita_nucleo(R2, w2)
    assert np.allclose(w1.iloc[: T + 1].dropna(), w2.iloc[: T + 1].dropna())
    assert np.allclose(v1.iloc[: T + 1].dropna(), v2.iloc[: T + 1].dropna())


def test_esposizione_senza_leva_e_decrescente_in_p():
    v = pd.Series([0.05, 0.05, 0.05])
    p = pd.Series([0.0, 0.5, 1.0])
    L = mo.esposizione(p, v, sigma_rif=0.10, m=1.0, r=0.25)
    assert (L <= 1.0).all()
    assert L.is_monotonic_decreasing or (L == 1.0).all()
    v2 = pd.Series([0.40, 0.40, 0.40])
    L2 = mo.esposizione(p, v2, 0.10, 1.0, 0.25)
    assert L2.iloc[0] > L2.iloc[1] > L2.iloc[2]
    assert abs(L2.iloc[2] - 0.10 * 0.25 / 0.40) < 1e-12


def test_giorni_decisione_ultimo_giorno_della_settimana():
    idx = pd.bdate_range("2024-01-01", periods=12)
    d = mo.giorni_decisione(idx)
    assert list(idx[d.to_numpy()]) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")]


def test_esecuzione_ritardata_di_un_giorno():
    R = dati(30)[["a"]]
    rf = pd.Series(0.0001, index=R.index)
    W = pd.DataFrame(np.nan, index=R.index, columns=["a"])
    W.iloc[5] = 1.0
    dec = pd.Series(False, index=R.index)
    dec.iloc[5] = True
    s = mo.simula(R, rf, W, dec, costo_bp=0.0)
    assert np.allclose(s["ritorno"].iloc[:7], 0.0001)       # giorni 0..6: ancora in liquidita'
    assert np.isclose(s["ritorno"].iloc[7], R["a"].iloc[7])  # guadagna dal giorno 7 = d + 2
    assert s["turnover"].iloc[6] == 1.0


def test_costo_e_solo_sulle_gambe_rischiose():
    R = dati(30)[["a"]]
    rf = pd.Series(0.0, index=R.index)
    W = pd.DataFrame(np.nan, index=R.index, columns=["a"])
    W.iloc[5] = 0.5
    dec = pd.Series(False, index=R.index)
    dec.iloc[5] = True
    s = mo.simula(R, rf, W, dec, costo_bp=10.0)
    assert np.isclose(s["ritorno"].iloc[6], -0.5 * 10e-4)


def test_senza_decisioni_resta_in_liquidita():
    R = dati(30)
    rf = pd.Series(0.0002, index=R.index)
    W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    s = mo.simula(R, rf, W, pd.Series(False, index=R.index))
    assert np.allclose(s["ritorno"], 0.0002) and (s["turnover"] == 0).all()


def test_deriva_dei_pesi_senza_ribilanciare():
    R = pd.DataFrame({"a": [0.0, 0.0, 0.10, 0.0], "b": [0.0, 0.0, 0.0, 0.0]}, index=pd.bdate_range("2020-01-06", periods=4))
    rf = pd.Series(0.0, index=R.index)
    W = pd.DataFrame(np.nan, index=R.index, columns=["a", "b"])
    W.iloc[0] = [0.5, 0.5]
    dec = pd.Series([True, False, False, False], index=R.index)
    s = mo.simula(R, rf, W, dec, costo_bp=0.0)
    assert np.isclose(s["ritorno"].iloc[2], 0.05)     # peso 0.5 su a, +10%
    assert np.isclose(s["esposizione"].iloc[3], 1.0)  # nessuna liquidita'
