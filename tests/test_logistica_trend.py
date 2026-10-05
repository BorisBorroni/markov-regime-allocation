import numpy as np
import pandas as pd

from markov_regime_allocation import logistica as lg
from markov_regime_allocation import trend as tr


def rend(n=2500, seed=5):
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 150) % 3 == 2, 0.02, 0.006)
    return pd.Series(rng.normal(0.0003, vol), index=pd.bdate_range("1996-01-02", periods=n))


def test_irls_recupera_i_coefficienti():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 2))
    vero = np.array([-0.5, 1.5, -1.0])
    p = 1 / (1 + np.exp(-(vero[0] + X @ vero[1:])))
    y = (rng.random(4000) < p).astype(float)
    beta = lg.adatta_logistica(X, y, c=1e6)
    assert np.allclose(beta, vero, atol=0.2)


def test_permanenza_a_mano():
    s = pd.Series([np.nan, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0])
    assert list(lg.giorni_permanenza(s).fillna(-1)) == [-1, 1, 2, 1, 2, 3, 1]


def test_logistica_valida_e_senza_futuro():
    r = rend()
    p1 = lg.probabilita_stress_logistica(r, minimo=500)
    ok = p1.dropna()
    assert len(ok) > 500 and ((ok >= 0) & (ok <= 1)).all()
    T = 1900
    r2 = r.copy()
    r2.iloc[T + 1:] = r2.iloc[T + 1:] * 6 + 0.03
    p2 = lg.probabilita_stress_logistica(r2, minimo=500)
    assert np.allclose(p1.iloc[: T + 1].dropna(), p2.iloc[: T + 1].dropna(), atol=1e-9)


def test_logistica_non_usa_etichette_non_ancora_note():
    # l'esito del giorno u (stato a u + 5) deve essere noto: alterare solo gli ultimi h giorni prima
    # della data di stima non cambia le previsioni fino a quella data
    r = rend()
    base = lg.probabilita_stress_logistica(r, minimo=500)
    T = 1500
    r2 = r.copy()
    r2.iloc[T + 1: T + 6] = 0.2
    alt = lg.probabilita_stress_logistica(r2, minimo=500)
    assert np.allclose(base.iloc[: T + 1].dropna(), alt.iloc[: T + 1].dropna(), atol=1e-9)


def test_trend_a_mano_e_senza_futuro():
    idx = pd.bdate_range("2020-01-01", periods=300)
    su = pd.Series(0.001, index=idx)
    giu = pd.Series(-0.001, index=idx)
    R = pd.DataFrame({"su": su, "giu": giu})
    sig = tr.in_tendenza(R, finestra=50)
    assert sig["su"].iloc[-1] == 1.0 and sig["giu"].iloc[-1] == 0.0
    assert sig["su"].iloc[:49].isna().all()
    w = pd.DataFrame(0.5, index=idx, columns=["su", "giu"])
    pw = tr.pesi_trend(R, w, finestra=50)
    assert pw["su"].iloc[-1] == 0.5 and pw["giu"].iloc[-1] == 0.0
    R2 = R.copy()
    R2.iloc[200:] = R2.iloc[200:] * -5
    assert np.allclose(tr.in_tendenza(R2, 50).iloc[:200].dropna(), sig.iloc[:200].dropna())
