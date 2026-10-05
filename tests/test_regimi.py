import numpy as np
import pandas as pd

from markov_regime_allocation import regimi


def serie_casuale(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    date = pd.bdate_range("2000-01-03", periods=n)
    scala = np.where((np.arange(n) // 300) % 2 == 0, 0.006, 0.02)
    return pd.Series(rng.normal(0, scala), index=date)


def test_volatilita_con_rendimenti_costanti():
    r = pd.Series(0.01, index=pd.bdate_range("2020-01-01", periods=50))
    vol = regimi.volatilita_ewma(r)
    assert np.allclose(vol, np.log1p(0.01) * np.sqrt(252))


def test_stati_validi_e_primi_giorni_vuoti():
    st = regimi.stati_as_of(regimi.volatilita_ewma(serie_casuale()), minimo=100)
    assert st.iloc[:99].isna().all()
    assert set(st.dropna().unique()) <= {0.0, 1.0, 2.0}


def test_matrice_a_mano():
    st = pd.Series([0, 0, 1, 1, 1, 0], dtype=float, index=pd.bdate_range("2020-01-01", periods=6))
    P = regimi.matrici_transizione(st, alpha=1.0)[-1]
    assert np.allclose(P[0], np.array([2, 2, 1]) / 5)
    assert np.allclose(P[1], np.array([2, 3, 1]) / 6)
    assert np.allclose(P[2], 1 / 3)


def test_righe_della_matrice_sommano_a_uno():
    st = regimi.stati_as_of(regimi.volatilita_ewma(serie_casuale()), minimo=100)
    P = regimi.matrici_transizione(st)
    assert np.allclose(P.sum(axis=2), 1.0)


def test_probabilita_a_un_passo_e_riga_della_matrice():
    st = regimi.stati_as_of(regimi.volatilita_ewma(serie_casuale()), minimo=100)
    P = regimi.matrici_transizione(st)
    p = regimi.probabilita_stress(st, h=1)
    t = 800
    assert np.isclose(p.iloc[t], P[t, int(st.iloc[t]), 2])


def test_probabilita_a_h_passi_e_potenza_della_matrice():
    st = regimi.stati_as_of(regimi.volatilita_ewma(serie_casuale()), minimo=100)
    P = regimi.matrici_transizione(st)
    t = 900
    atteso = np.linalg.matrix_power(P[t], 5)[int(st.iloc[t]), 2]
    assert np.isclose(regimi.probabilita_stress(st, h=5).iloc[t], atteso)


def test_nessuna_informazione_dal_futuro():
    """Stati, matrice e probabilita' fino a T non cambiano se i dati dopo T cambiano o spariscono."""
    r = serie_casuale()
    T = 1000
    intero = regimi.volatilita_ewma(r)
    st_int = regimi.stati_as_of(intero, minimo=100)
    p_int = regimi.probabilita_stress(st_int)
    P_int = regimi.matrici_transizione(st_int)

    troncata = r.iloc[:T + 1]
    st_tr = regimi.stati_as_of(regimi.volatilita_ewma(troncata), minimo=100)
    p_tr = regimi.probabilita_stress(st_tr)
    P_tr = regimi.matrici_transizione(st_tr)
    assert st_int.iloc[:T + 1].equals(st_tr)
    assert np.allclose(p_int.iloc[:T + 1].to_numpy(), p_tr.to_numpy(), equal_nan=True)
    assert np.allclose(P_int[:T + 1], P_tr)

    alterata = r.copy()
    alterata.iloc[T + 1:] = 0.15
    st_al = regimi.stati_as_of(regimi.volatilita_ewma(alterata), minimo=100)
    assert st_int.iloc[:T + 1].equals(st_al.iloc[:T + 1])
    assert np.allclose(p_int.iloc[:T + 1].to_numpy(), regimi.probabilita_stress(st_al).iloc[:T + 1].to_numpy(), equal_nan=True)


def test_lo_stato_passato_non_viene_riclassificato():
    """Aggiungendo giorni di volatilita' estrema, gli stati gia' assegnati restano uguali."""
    r = serie_casuale()
    prima = regimi.stati_as_of(regimi.volatilita_ewma(r.iloc[:1000]), minimo=100)
    dopo = regimi.stati_as_of(regimi.volatilita_ewma(r), minimo=100).iloc[:1000]
    assert prima.equals(dopo)
