"""Regimi di volatilita' osservabili e catena di Markov stimata solo sul passato.

Stati: 0 calma, 1 normale, 2 stress. Ogni valore al giorno t usa solo dati fino a t.
"""
import numpy as np
import pandas as pd

LAMBDA = 0.94
Q_BASSO = 0.40
Q_ALTO = 0.90
MINIMO_STORIA = 252
STATI = 3
ORIZZONTE = 5


def volatilita_ewma(rendimenti, lam=LAMBDA):
    """Volatilita' annualizzata: media mobile esponenziale dei quadrati dei rendimenti logaritmici."""
    log_r = np.log1p(rendimenti)
    varianza = (log_r ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(varianza * 252)


def stati_as_of(vol, q_basso=Q_BASSO, q_alto=Q_ALTO, minimo=MINIMO_STORIA):
    """Stato di ogni giorno, con soglie stimate solo sul passato (oggi incluso).

    Le soglie sono i quantili della volatilita' su finestra espandibile. Lo stato di
    un giorno passato non viene mai ricalcolato con le soglie successive.
    I primi `minimo - 1` giorni restano NaN.
    """
    espandente = vol.expanding(min_periods=minimo)
    bassa = espandente.quantile(q_basso)
    alta = espandente.quantile(q_alto)
    stati = pd.Series(1.0, index=vol.index)
    stati[vol <= bassa] = 0.0
    stati[vol > alta] = 2.0
    stati[bassa.isna()] = np.nan
    return stati


def conteggi_transizioni(stati):
    """Conteggi cumulati delle transizioni osservate fino a ogni giorno, array (n, 3, 3).

    La transizione da t-1 a t e' nota alla chiusura di t, quindi entra nel conteggio di t.
    """
    s = stati.to_numpy()
    nuovi = np.zeros((len(s), STATI, STATI))
    valide = ~np.isnan(s[1:]) & ~np.isnan(s[:-1])
    giorno = np.nonzero(valide)[0] + 1
    nuovi[giorno, s[:-1][valide].astype(int), s[1:][valide].astype(int)] = 1.0
    return np.cumsum(nuovi, axis=0)


def matrici_transizione(stati, alpha=1.0):
    """Matrice di transizione stimata a ogni giorno, con lisciamento di Laplace."""
    c = conteggi_transizioni(stati) + alpha
    return c / c.sum(axis=2, keepdims=True)


def probabilita_stress(stati, h=ORIZZONTE, alpha=1.0):
    """Probabilita' di essere in stress tra h giorni, dato lo stato di oggi."""
    potenze = np.linalg.matrix_power(matrici_transizione(stati, alpha), h)
    s = stati.to_numpy()
    p = np.full(len(s), np.nan)
    ok = ~np.isnan(s)
    p[ok] = potenze[ok, s[ok].astype(int), STATI - 1]
    return pd.Series(p, index=stati.index)
