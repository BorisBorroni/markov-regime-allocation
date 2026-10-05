"""Regressione logistica con memoria (penalita' L2, IRLS in numpy) per P(stress tra h giorni).

Variabili: log della volatilita' EWMA, log della volatilita' realizzata a 5, 22 e 66 giorni,
log(1 + giorni di permanenza nello stato corrente). Riaddestramento ogni 252 giorni su finestra
espandibile; un esempio addestra solo se la sua etichetta (stato a u + h) e' nota alla data di stima.
Le variabili sono standardizzate con media e deviazione standard dei soli esempi di addestramento.
"""
import numpy as np
import pandas as pd

from .regimi import ORIZZONTE, Q_ALTO, stati_as_of, volatilita_ewma

C = 1.0
PASSO = 252
MINIMO_ESEMPI = 756


def giorni_permanenza(stati):
    s = stati.to_numpy()
    out = np.full(len(s), np.nan)
    corrente = 0
    for i in range(len(s)):
        if np.isnan(s[i]):
            corrente = 0
            continue
        corrente = corrente + 1 if i > 0 and s[i] == s[i - 1] else 1
        out[i] = corrente
    return pd.Series(out, index=stati.index)


def variabili(rendimenti, stati):
    log_r = np.log1p(rendimenti)
    vol = lambda n: np.sqrt((log_r ** 2).rolling(n).mean() * 252)  # noqa: E731
    X = pd.DataFrame({
        "ewma": np.log(volatilita_ewma(rendimenti)),
        "v5": np.log(vol(5).clip(lower=1e-4)),
        "v22": np.log(vol(22)),
        "v66": np.log(vol(66)),
        "perm": np.log1p(giorni_permanenza(stati)),
    })
    return X


def adatta_logistica(X, y, c=C, iterazioni=50):
    """IRLS con penalita' L2 (intercetta non penalizzata). X gia' standardizzata."""
    n, k = X.shape
    Z = np.c_[np.ones(n), X]
    beta = np.zeros(k + 1)
    pen = np.eye(k + 1) / c
    pen[0, 0] = 0.0
    for _ in range(iterazioni):
        eta = np.clip(Z @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = np.maximum(mu * (1 - mu), 1e-9)
        grad = Z.T @ (y - mu) - pen @ beta
        H = Z.T @ (Z * W[:, None]) + pen
        passo = np.linalg.solve(H, grad)
        beta = beta + passo
        if np.abs(passo).max() < 1e-8:
            break
    return beta


def probabilita_stress_logistica(rendimenti, h=ORIZZONTE, q_alto=Q_ALTO, passo=PASSO, minimo=MINIMO_ESEMPI):
    stati = stati_as_of(volatilita_ewma(rendimenti), q_alto=q_alto)
    X = variabili(rendimenti, stati)
    esito = (stati.shift(-h) == 2).astype(float).where(stati.shift(-h).notna())
    validi = X.notna().all(axis=1).to_numpy()
    Xn = X.to_numpy()
    y = esito.to_numpy()
    n = len(X)
    p = np.full(n, np.nan)
    primo = np.nonzero(validi)[0]
    if len(primo) == 0:
        return pd.Series(p, index=X.index)
    t = primo[0] + minimo + h
    while t < n:
        # esempi con etichetta nota alla data t: u + h <= t
        u = np.arange(primo[0], t - h + 1)
        u = u[validi[u] & ~np.isnan(y[u])]
        media, sd = Xn[u].mean(axis=0), Xn[u].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        beta = adatta_logistica((Xn[u] - media) / sd, y[u])
        fine = min(t + passo, n)
        blocco = np.arange(t, fine)
        blocco = blocco[validi[blocco]]
        z = np.c_[np.ones(len(blocco)), (Xn[blocco] - media) / sd] @ beta
        p[blocco] = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        t = fine
    return pd.Series(p, index=X.index)
