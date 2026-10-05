"""HMM gaussiano a 3 stati scritto in numpy, solo filtro in avanti, riaddestramento annuale.

La stima (EM) al giorno di riaddestramento usa solo rendimenti fino a quel giorno; il filtro
in avanti al giorno t usa solo rendimenti fino a t con i parametri dell'ultimo riaddestramento.
Mai Viterbi ne' smoothing sul periodo operativo. Gli stati sono ordinati per varianza.
"""
import numpy as np
import pandas as pd

STATI = 3
FLOOR_VAR = 1e-8


def _emissioni(x, medie, varianze):
    z = (x[:, None] - medie[None, :]) ** 2 / varianze[None, :]
    return np.exp(-0.5 * z) / np.sqrt(2 * np.pi * varianze[None, :])


def _avanti(b, pi, A):
    n = len(b)
    alfa = np.zeros((n, STATI))
    scala = np.zeros(n)
    a = pi * b[0]
    scala[0] = a.sum()
    alfa[0] = a / scala[0]
    for t in range(1, n):
        a = (alfa[t - 1] @ A) * b[t]
        scala[t] = a.sum()
        alfa[t] = a / scala[t]
    return alfa, scala


def _indietro(b, A, scala):
    n = len(b)
    beta = np.ones((n, STATI))
    for t in range(n - 2, -1, -1):
        beta[t] = (A @ (b[t + 1] * beta[t + 1])) / scala[t + 1]
    return beta


def ordina(par):
    ordine = np.argsort(par["varianze"])
    return {"pi": par["pi"][ordine], "A": par["A"][np.ix_(ordine, ordine)],
            "medie": par["medie"][ordine], "varianze": par["varianze"][ordine]}


def parametri_iniziali(x):
    """Inizializzazione deterministica: stati per terzili del valore assoluto del rendimento."""
    soglie = np.quantile(np.abs(x), [1 / 3, 2 / 3])
    classe = np.digitize(np.abs(x), soglie)
    medie = np.array([x[classe == j].mean() for j in range(STATI)])
    varianze = np.array([max(x[classe == j].var(), FLOOR_VAR) for j in range(STATI)])
    A = np.full((STATI, STATI), 0.05 / (STATI - 1))
    np.fill_diagonal(A, 0.95)
    return {"pi": np.full(STATI, 1 / STATI), "A": A, "medie": medie, "varianze": varianze}


def stima_em(x, iniziale=None, iterazioni=60, tolleranza=1e-6):
    par = iniziale if iniziale is not None else parametri_iniziali(x)
    ll_prec = -np.inf
    for _ in range(iterazioni):
        b = np.maximum(_emissioni(x, par["medie"], par["varianze"]), 1e-300)
        alfa, scala = _avanti(b, par["pi"], par["A"])
        beta = _indietro(b, par["A"], scala)
        gamma = alfa * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (alfa[:-1, :, None] * par["A"][None] * (b[1:] * beta[1:])[:, None, :]) / scala[1:, None, None]
        A = xi.sum(axis=0) + 1e-6
        A /= A.sum(axis=1, keepdims=True)
        peso = gamma.sum(axis=0)
        medie = (gamma * x[:, None]).sum(axis=0) / peso
        varianze = np.maximum((gamma * (x[:, None] - medie[None, :]) ** 2).sum(axis=0) / peso, FLOOR_VAR)
        par = {"pi": gamma[0], "A": A, "medie": medie, "varianze": varianze}
        ll = np.log(scala).sum()
        if abs(ll - ll_prec) < tolleranza:
            break
        ll_prec = ll
    return ordina(par)


def filtro_avanti(x, par):
    b = np.maximum(_emissioni(x, par["medie"], par["varianze"]), 1e-300)
    alfa, _ = _avanti(b, par["pi"], par["A"])
    return alfa


def probabilita_stress_hmm(rendimenti, h=5, primo_riaddestramento="1998-01-01", minimo=504):
    """P(stato di varianza piu' alta tra h giorni) dal filtro in avanti.

    Riaddestramento al primo giorno di borsa di ogni anno dal `primo_riaddestramento`, su rendimenti
    fino al giorno prima. Il filtro riparte dall'inizio della serie con i parametri correnti
    (solo rendimenti fino a t).
    """
    x = rendimenti.to_numpy()
    idx = rendimenti.index
    anni = idx.year
    inizio_anno = np.nonzero(np.r_[True, anni[1:] != anni[:-1]])[0]
    p = np.full(len(x), np.nan)
    par = None
    for k, i0 in enumerate(inizio_anno):
        if idx[i0] < pd.Timestamp(primo_riaddestramento) or i0 < minimo:
            continue
        i1 = inizio_anno[k + 1] if k + 1 < len(inizio_anno) else len(x)
        par = stima_em(x[:i0], iniziale=par)
        alfa = filtro_avanti(x[:i1], par)
        Ah = np.linalg.matrix_power(par["A"], h)
        p[i0:i1] = (alfa[i0:i1] @ Ah)[:, -1]
    return pd.Series(p, index=idx)
