"""Inferenza sul test: bootstrap stazionario, commissione di performance, Deflated Sharpe, alpha, tabella per regime.

Le formule della commissione (Fleming, Kirby e Ostdiek 2001) e della Deflated Sharpe (Bailey e Lopez de Prado 2014)
sono scritte dalla memoria dei lavori originali e NON sono state verificate sul testo.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

EULERO = 0.5772156649015329
_N = NormalDist()


def indici_bootstrap(n, B, blocco_medio, rng, chunk=1000, modulo=None):
    """Genera blocchi di indici di bootstrap stazionario (Politis e Romano): matrici (<=chunk, n).

    Gli indici vanno da 0 a modulo - 1 (di default n - 1); con modulo diverso da n si ricampiona una serie
    piu' corta o piu' lunga di n, con blocchi che ricominciano da capo alla fine della serie.
    """
    modulo = n if modulo is None else modulo
    p = 1.0 / blocco_medio
    fatti = 0
    while fatti < B:
        b = min(chunk, B - fatti)
        nuovo = rng.random((b, n)) < p
        inizi = rng.integers(0, modulo, (b, n))
        idx = np.empty((b, n), dtype=np.int64)
        idx[:, 0] = inizi[:, 0]
        for t in range(1, n):
            idx[:, t] = np.where(nuovo[:, t], inizi[:, t], (idx[:, t - 1] + 1) % modulo)
        yield idx
        fatti += b


def sharpe(eccesso, asse=-1):
    return eccesso.mean(axis=asse) / eccesso.std(axis=asse, ddof=1) * math.sqrt(252)


def drawdown_massimo(rendimenti):
    """Perdita massima dal picco, con valore iniziale 1 (una perdita nei primi giorni conta)."""
    valore = np.cumprod(1 + rendimenti, axis=-1)
    iniziale = np.ones(valore.shape[:-1] + (1,))
    valore = np.concatenate([iniziale, valore], axis=-1)
    picco = np.maximum.accumulate(valore, axis=-1)
    return (valore / picco - 1).min(axis=-1)


def differenza_bootstrap(a, b, rf, B=10000, blocco_medio=20, seed=0, livello=0.95):
    """Differenza (a meno b) di Sharpe e di drawdown massimo, con intervallo di confidenza bootstrap.

    a, b, rf: array di rendimenti giornalieri allineati. La coppia e' ricampionata insieme.
    """
    a, b, rf = (np.asarray(x, dtype=float) for x in (a, b, rf))
    n = len(a)
    rng = np.random.default_rng(seed)
    ds, dd = [], []
    for idx in indici_bootstrap(n, B, blocco_medio, rng):
        ra, rb, rr = a[idx], b[idx], rf[idx]
        ds.append(sharpe(ra - rr) - sharpe(rb - rr))
        dd.append(drawdown_massimo(ra) - drawdown_massimo(rb))
    ds, dd = np.concatenate(ds), np.concatenate(dd)
    coda = (1 - livello) / 2
    return {
        "sharpe_diff": float(sharpe(a - rf) - sharpe(b - rf)),
        "sharpe_ic": (float(np.quantile(ds, coda)), float(np.quantile(ds, 1 - coda))),
        "quota_sharpe_diff_non_positiva": float((ds <= 0).mean()),
        "drawdown_diff": float(drawdown_massimo(a) - drawdown_massimo(b)),
        "drawdown_ic": (float(np.quantile(dd, coda)), float(np.quantile(dd, 1 - coda))),
    }


def commissione_performance(a, b, gamma, giorni=252):
    """Commissione annua (frazione) che un investitore con utilita' quadratica e avversione gamma pagherebbe per passare da b ad a.

    U(R) = R - k R^2 con k = gamma / (2 (1 + gamma)) e R rendimento lordo (1 + r). Si risolve U(a - fee) = U(b) in media,
    con rendimenti giornalieri; positiva se a e' preferibile a b.
    """
    a, b = 1 + np.asarray(a, dtype=float), 1 + np.asarray(b, dtype=float)
    k = gamma / (2 * (1 + gamma))
    ua = a.mean() - k * (a ** 2).mean()
    ub = b.mean() - k * (b ** 2).mean()
    c2, c1, c0 = k, 1 - 2 * k * a.mean(), ub - ua
    disc = c1 * c1 - 4 * c2 * c0
    if disc < 0:
        return float("nan")
    radice = (-c1 + math.sqrt(disc)) / (2 * c2)
    return float(radice * giorni)


def sharpe_atteso_massimo(n_tentativi, varianza_sharpe):
    """Sharpe massimo atteso (non annualizzato) tra n tentativi indipendenti con Sharpe vero nullo."""
    if n_tentativi <= 1:
        return 0.0
    return math.sqrt(varianza_sharpe) * ((1 - EULERO) * _N.inv_cdf(1 - 1 / n_tentativi) + EULERO * _N.inv_cdf(1 - 1 / (n_tentativi * math.e)))


def deflated_sharpe(rendimenti_in_eccesso, n_tentativi, varianza_sharpe_tra_tentativi):
    """Probabilita' che lo Sharpe vero superi la soglia attesa dopo aver provato n_tentativi configurazioni.

    Lo Sharpe e' non annualizzato (giornaliero). `varianza_sharpe_tra_tentativi` e' la varianza degli Sharpe giornalieri dei tentativi.
    """
    x = np.asarray(rendimenti_in_eccesso, dtype=float)
    T = len(x)
    sr = x.mean() / x.std(ddof=1)
    z = (x - x.mean()) / x.std(ddof=1)
    g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
    sr0 = sharpe_atteso_massimo(n_tentativi, varianza_sharpe_tra_tentativi)
    denominatore = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(_N.cdf((sr - sr0) * math.sqrt(T - 1) / denominatore))


def ols_newey_west(y, x, ritardi):
    """Regressione y = alfa + beta x con errori standard Newey-West. Restituisce (coefficienti, errori standard)."""
    y = np.asarray(y, dtype=float)
    X = np.c_[np.ones(len(y)), np.asarray(x, dtype=float)]
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for k in range(1, ritardi + 1):
        G = Xu[k:].T @ Xu[:-k]
        S += (1 - k / (ritardi + 1)) * (G + G.T)
    inv = np.linalg.inv(X.T @ X)
    V = inv @ S @ inv
    return beta, np.sqrt(np.diag(V))


def alfa_contro_nucleo(strategia, nucleo, rf, ritardi, giorni=252):
    """Alfa annualizzato (frazione) della strategia sul nucleo statico, sui rendimenti in eccesso, con t Newey-West."""
    y = np.asarray(strategia) - np.asarray(rf)
    x = np.asarray(nucleo) - np.asarray(rf)
    beta, se = ols_newey_west(y, x, ritardi)
    return {"alfa_annuo": float(beta[0] * giorni), "t_alfa": float(beta[0] / se[0]), "beta": float(beta[1])}


def tabella_per_regime(rendimenti_giornalieri, stati, rf):
    """Rendimento in eccesso medio annualizzato, volatilita' e quota di giorni per stato osservato il giorno prima."""
    ieri = stati.shift(1).reindex(rendimenti_giornalieri.index)
    ecc = rendimenti_giornalieri - rf.reindex(rendimenti_giornalieri.index)
    righe = {}
    for j in (0, 1, 2):
        e = ecc[ieri == j]
        righe[j] = {"quota_giorni": len(e) / ieri.notna().sum(), "eccesso_annuo": e.mean() * 252, "volatilita_annua": e.std(ddof=1) * math.sqrt(252)}
    return pd.DataFrame(righe).T
