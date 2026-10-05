"""Controllo end-to-end sui dati reali: troncare i dati a una data anteriore non cambia nulla fino a quella data.

Salta se i dati grezzi (non versionati) non sono presenti. Usa solo il periodo di sviluppo.
"""
import numpy as np
import pandas as pd
import pytest

from markov_regime_allocation import dati, hmm, logistica, motore, multiscala, regimi
from markov_regime_allocation.esperimento import prepara

pytestmark = pytest.mark.slow

if not (dati.DATI / "raw" / "gld.us.txt").exists():
    pytest.skip("dati grezzi non presenti", allow_module_level=True)

TAGLIO = "2012-12-31"
FINE = "2014-12-31"


def uguali(a, b, fine):
    a, b = a.loc[:fine], b.loc[:fine]
    return a.index.equals(b.index) and np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True, atol=1e-12)


def test_nucleo_e_segnali_non_dipendono_dal_futuro():
    mkt1, R1, rf1, w1, vol1, sig1, dec1 = prepara(TAGLIO)
    mkt2, R2, rf2, w2, vol2, sig2, dec2 = prepara(FINE)
    assert sig1 == sig2
    assert uguali(w1, w2, TAGLIO) and uguali(vol1, vol2, TAGLIO)
    assert (dec1.loc[:TAGLIO].to_numpy()[:-1] == dec2.loc[:TAGLIO].to_numpy()[:-1]).all()
    assert uguali(multiscala.probabilita_stress_multiscala(mkt1)[0], multiscala.probabilita_stress_multiscala(mkt2)[0], TAGLIO)
    p1 = regimi.probabilita_stress(regimi.stati_as_of(regimi.volatilita_ewma(mkt1)))
    p2 = regimi.probabilita_stress(regimi.stati_as_of(regimi.volatilita_ewma(mkt2)))
    assert uguali(p1, p2, TAGLIO)
    assert uguali(logistica.probabilita_stress_logistica(mkt1), logistica.probabilita_stress_logistica(mkt2), TAGLIO)
    assert uguali(hmm.probabilita_stress_hmm(mkt1), hmm.probabilita_stress_hmm(mkt2), TAGLIO)


def test_simulazione_non_dipende_dal_futuro():
    mkt1, R1, rf1, w1, vol1, sig1, dec1 = prepara(TAGLIO)
    mkt2, R2, rf2, w2, vol2, sig2, dec2 = prepara(FINE)
    out = []
    for mkt, R, rf, w, vol, sig, dec in ((mkt1, R1, rf1, w1, vol1, sig1, dec1), (mkt2, R2, rf2, w2, vol2, sig2, dec2)):
        p = multiscala.probabilita_stress_multiscala(mkt)[0].reindex(R.index)
        L = motore.esposizione(p, vol, sig, 0.8, 0.25)
        out.append(motore.simula(R, rf, w.mul(L, axis=0), dec, 5.0))
    # l'ultimo giorno del troncamento puo' avere un ordine in attesa che dipende dalla decisione successiva: si confronta fino al giorno prima
    fine = pd.Timestamp(TAGLIO) - pd.Timedelta(days=7)
    assert uguali(out[0]["ritorno"], out[1]["ritorno"], fine)
