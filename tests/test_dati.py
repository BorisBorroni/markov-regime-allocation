import numpy as np
import pandas as pd
import pytest

from markov_regime_allocation import dati

FILE_FRENCH = """Intestazione del file.

  Average Value Weighted Returns -- Daily
,AAA,BBB
20200102,   1.00,  -99.99
20200103,  -2.00,   0.50

  Average Equal Weighted Returns -- Daily
,AAA,BBB
20200102,   9.00,   9.00
20200103,   9.00,   9.00

Copyright
"""


def test_leggi_french_prende_solo_la_sezione_pesata(tmp_path):
    f = tmp_path / "ind.csv"
    f.write_text(FILE_FRENCH)
    t = dati.leggi_french(f)
    assert list(t.columns) == ["AAA", "BBB"]
    assert len(t) == 2
    assert t.loc["2020-01-02", "AAA"] == 0.01
    assert np.isnan(t.loc["2020-01-02", "BBB"])
    assert t.loc["2020-01-03", "AAA"] == -0.02


def test_leggi_french_senza_titolo_di_sezione(tmp_path):
    f = tmp_path / "fattori.csv"
    f.write_text("Nota\n\n,Mkt-RF,RF\n20200102,  1.00,  0.01\n20200103,  2.00,  0.01\n\nCopyright\n")
    t = dati.leggi_french(f)
    assert list(t.columns) == ["Mkt-RF", "RF"]
    assert t["Mkt-RF"].tolist() == [0.01, 0.02]


def test_durata_del_titolo_alla_pari_formula_chiusa():
    for y in (0.02, 0.05, 0.08):
        durata, _ = dati.durata_convessita(y, 7)
        v = 1 / (1 + y / 2)
        assert abs(durata - (1 - v ** 14) / y) < 1e-6


def test_convessita_positiva_e_durata_cresce_con_la_scadenza():
    d7, c7 = dati.durata_convessita(0.04, 7)
    d10, _ = dati.durata_convessita(0.04, 10)
    assert c7 > 0
    assert d10 > d7 > 0


def test_obbligazione_con_rendimento_costante_rende_la_cedola():
    date = pd.bdate_range("2020-01-01", periods=6)
    r = dati.rendimento_obbligazione(pd.Series(0.04, index=date))
    assert len(r) == 5
    giorni = date.to_series().diff().dt.days.iloc[1:] / 365
    assert np.allclose(r.to_numpy(), 0.04 * giorni.to_numpy())


def test_obbligazione_guadagna_se_i_tassi_scendono():
    date = pd.bdate_range("2020-01-01", periods=3)
    r = dati.rendimento_obbligazione(pd.Series([0.04, 0.0390, 0.0400], index=date))
    durata, _ = dati.durata_convessita(0.04, 7)
    assert r.iloc[0] > 0.04 / 365
    assert abs(r.iloc[0] - (0.04 / 365 + durata * 0.0010)) < 5e-4
    assert r.iloc[1] < 0.04 / 365


def test_messaggio_chiaro_se_mancano_i_dati(tmp_path, monkeypatch):
    monkeypatch.setattr(dati, "DATI", tmp_path)
    with pytest.raises(FileNotFoundError, match="00_scarica_dati"):
        dati.controlla_file()
