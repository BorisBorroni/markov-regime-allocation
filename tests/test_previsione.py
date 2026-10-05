import math

import numpy as np
import pandas as pd

from markov_regime_allocation import previsione as pv
from markov_regime_allocation import regimi


def serie_casuale(n=1500, seed=5):
    rng = np.random.default_rng(seed)
    date = pd.bdate_range("2000-01-03", periods=n)
    scala = np.where((np.arange(n) // 300) % 2 == 0, 0.006, 0.02)
    return pd.Series(rng.normal(0, scala), index=date)


def test_varianza_futura_a_mano():
    r = pd.Series([0.0, 0.1, 0.0, 0.2, 0.0, 0.0], index=pd.bdate_range("2020-01-01", periods=6))
    f = pv.varianza_futura(r, 2)
    attesa = (np.log1p(0.1) ** 2 + 0.0) / 2
    assert np.isclose(f.iloc[0], attesa)
    assert f.iloc[-2:].isna().all()


def test_qlike_zero_se_previsione_esatta_e_positiva_altrimenti():
    x = pd.Series([1.0, 2.0, 3.0])
    assert np.allclose(pv.qlike(x, x), 0.0)
    assert (pv.qlike(x, x * 2) > 0).all()
    assert (pv.qlike(x, x / 2) > 0).all()


def test_diebold_mariano_differenza_costante_nulla_e_positiva():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(1.0, 0.1, 400))
    media, t = pv.diebold_mariano(a, a - 0.5 + rng.normal(0, 0.01, 400), 10)
    assert media > 0.4
    assert t > 10
    media0, t0 = pv.diebold_mariano(a, a + 1e-9 * np.arange(400), 10)
    assert abs(media0) < 1e-6


def test_chi2_valori_noti():
    assert abs(pv.sopravvivenza_chi2(21.0261, 12) - 0.05) < 1e-4
    assert abs(pv.sopravvivenza_chi2(4.0, 2) - math.exp(-2.0)) < 1e-12
    assert pv.sopravvivenza_chi2(0.0, 5) == 1.0


def test_ordine_markov_sequenza_ciclica_non_rifiuta():
    s = pd.Series([0, 1, 2] * 200, dtype=float)
    lr, gradi, p = pv.test_ordine_markov(s)
    assert gradi == 12
    assert abs(lr) < 1e-9
    assert p > 0.99


def test_ordine_markov_rifiuta_se_serve_la_memoria_di_due_giorni():
    s = pd.Series([0, 0, 1, 1] * 300, dtype=float)
    lr, _, p = pv.test_ordine_markov(s)
    assert lr > 100
    assert p < 1e-6


def test_sopravvivenza_regimi_a_mano():
    s = pd.Series([0, 0, 0, 1, 1, 0, 0, 0, 0, 2], dtype=float)
    t = pv.sopravvivenza_regimi(s, durate=(2, 4))
    assert t.loc[0, 2] == 1.0
    assert t.loc[0, 4] == 0.5
    assert t.loc[1, 2] == 1.0


def test_probabilita_di_stress_per_brier_e_log_score():
    p = pd.Series([0.9, 0.1])
    esito = pd.Series([1.0, 0.0])
    assert np.allclose(pv.brier(p, esito), [0.01, 0.01])
    assert (pv.log_score(pd.Series([0.0, 1.0]), esito) > 4).all()


def test_nessuna_informazione_dal_futuro_nelle_previsioni():
    r = serie_casuale()
    T = 1000
    st = regimi.stati_as_of(regimi.volatilita_ewma(r), minimo=100)
    st_tr = regimi.stati_as_of(regimi.volatilita_ewma(r.iloc[:T + 1]), minimo=100)
    for h in (5, 22):
        for nome, completa, troncata in [
            ("catena", pv.previsione_catena(r, st, h), pv.previsione_catena(r.iloc[:T + 1], st_tr, h)),
            ("har", pv.previsione_har(r, h, minimo=100), pv.previsione_har(r.iloc[:T + 1], h, minimo=100)),
        ]:
            assert np.allclose(completa.iloc[:T + 1].to_numpy(), troncata.to_numpy(), equal_nan=True), nome
    assert np.allclose(pv.previsione_media_espandibile(r).iloc[:T + 1], pv.previsione_media_espandibile(r.iloc[:T + 1]))
    f_int = pv.frequenza_stress_espandibile(st)
    f_tr = pv.frequenza_stress_espandibile(st_tr)
    assert np.allclose(f_int.iloc[:T + 1].to_numpy(), f_tr.to_numpy(), equal_nan=True)


def test_har_non_usa_esempi_non_ancora_osservati():
    """Cambiare i rendimenti dopo t - h non cambia la previsione HAR in t se l'esempio e' fuori dalla finestra di stima."""
    r = serie_casuale()
    t, h = 1000, 5
    base = pv.previsione_har(r, h, minimo=100).iloc[t]
    alterata = r.copy()
    alterata.iloc[t + 1:] = 0.2
    assert np.isclose(base, pv.previsione_har(alterata, h, minimo=100).iloc[t])


def test_sopravvivenza_geometrica_a_mano():
    # dallo stato 0 partono 6 transizioni e 4 restano in 0: p00 = 2/3
    s = pd.Series([0, 0, 0, 0, 0, 1, 0, 1, 2, 2, 1], dtype=float)
    t = pv.sopravvivenza_geometrica(s, durate=(1, 3))
    assert t.loc[0, 1] == 1.0
    assert np.isclose(t.loc[0, 3], (2 / 3) ** 2)
