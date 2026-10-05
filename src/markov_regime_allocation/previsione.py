"""Previsione della volatilita' e della probabilita' di stress: modelli, perdite e test.

La varianza e' sempre la media giornaliera dei quadrati dei rendimenti logaritmici.
"""
import math

import numpy as np
import pandas as pd

from .regimi import STATI, matrici_transizione

FLOOR = 1e-10


def quadrati(rendimenti):
    return np.log1p(rendimenti) ** 2


def varianza_futura(rendimenti, h):
    """Varianza realizzata media nei giorni da t+1 a t+h (NaN negli ultimi h giorni)."""
    return quadrati(rendimenti).rolling(h).mean().shift(-h)


def varianza_per_stato(rendimenti, stati):
    """Media espandibile dei quadrati dei rendimenti per stato, usando solo il passato: (n, 3)."""
    r2 = quadrati(rendimenti).to_numpy()
    s = stati.to_numpy()
    v = np.full((len(s), STATI), np.nan)
    for j in range(STATI):
        dentro = s == j
        somma = np.cumsum(np.where(dentro, r2, 0.0))
        quanti = np.cumsum(dentro)
        v[:, j] = np.where(quanti > 0, somma / np.maximum(quanti, 1), np.nan)
    return v


def previsione_catena(rendimenti, stati, h, alpha=1.0):
    """Varianza media attesa nei prossimi h giorni secondo la catena di Markov.

    Media su k = 1..h di: sum_j P^k[stato oggi, j] * varianza dello stato j.
    """
    P = matrici_transizione(stati, alpha)
    v = varianza_per_stato(rendimenti, stati)
    s = stati.to_numpy()
    ok = ~np.isnan(s)
    riga = np.zeros(len(s), dtype=int)
    riga[ok] = s[ok].astype(int)
    indice = np.arange(len(s))
    potenza = P.copy()
    totale = np.zeros(len(s))
    for k in range(1, h + 1):
        if k > 1:
            potenza = potenza @ P
        totale += (potenza[indice, riga, :] * v).sum(axis=1)
    previsione = np.where(ok, totale / h, np.nan)
    return pd.Series(previsione, index=stati.index).clip(lower=FLOOR)


def previsione_media_espandibile(rendimenti):
    """Media espandibile dei quadrati dei rendimenti fino a oggi."""
    return quadrati(rendimenti).expanding().mean()


def previsione_har(rendimenti, h, minimo=250):
    """Modello HAR: regressione OLS espandibile della varianza futura su varianza
    giornaliera, media a 5 giorni e media a 22 giorni.

    Alla data t usa solo esempi la cui varianza futura e' gia' interamente osservata
    (esempi fino a t - h).
    """
    r2 = quadrati(rendimenti)
    X = np.column_stack([np.ones(len(r2)), r2, r2.rolling(5).mean(), r2.rolling(22).mean()])
    y = varianza_futura(rendimenti, h).to_numpy()
    valido = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    Xv = np.where(valido[:, None], X, 0.0)
    yv = np.where(valido, y, 0.0)
    xx = np.cumsum(Xv[:, :, None] * Xv[:, None, :], axis=0)
    xy = np.cumsum(Xv * yv[:, None], axis=0)
    quanti = np.cumsum(valido)
    prev = np.full(len(r2), np.nan)
    for t in range(h, len(r2)):
        u = t - h
        if quanti[u] < minimo or np.isnan(X[t]).any():
            continue
        try:
            beta = np.linalg.solve(xx[u], xy[u])
        except np.linalg.LinAlgError:
            continue
        prev[t] = X[t] @ beta
    return pd.Series(prev, index=r2.index).clip(lower=FLOOR)


def qlike(reale, previsto):
    """Perdita QLIKE: reale/previsto - ln(reale/previsto) - 1 (zero se la previsione e' esatta)."""
    rapporto = reale / previsto
    return rapporto - np.log(rapporto) - 1


def diebold_mariano(perdita_a, perdita_b, ritardi):
    """Media della differenza di perdita (a meno b) e statistica t con errori Newey-West."""
    d = (perdita_a - perdita_b).dropna().to_numpy()
    n = len(d)
    media = d.mean()
    u = d - media
    s = (u * u).mean()
    for k in range(1, ritardi + 1):
        s += 2 * (1 - k / (ritardi + 1)) * (u[k:] * u[:-k]).sum() / n
    t = media / math.sqrt(s / n) if s > 0 else float("nan")
    return media, t


def p_value_normale(t):
    """P-value a due code di una normale standard."""
    return math.erfc(abs(t) / math.sqrt(2))


def brier(p, esito):
    return (p - esito) ** 2


def log_score(p, esito, eps=0.01):
    """Perdita logaritmica, con probabilita' limitate a [eps, 1 - eps]."""
    p = p.clip(eps, 1 - eps)
    return -(esito * np.log(p) + (1 - esito) * np.log(1 - p))


def frequenza_stress_espandibile(stati):
    """Frequenza di giorni in stress fino a oggi (previsione incondizionata)."""
    stress = (stati == STATI - 1).astype(float).where(stati.notna())
    return stress.expanding().mean()


def sopravvivenza_chi2(x, gradi):
    """Probabilita' che un chi quadro con `gradi` gradi di liberta' superi x."""
    if x <= 0:
        return 1.0
    a, z = gradi / 2, x / 2
    if z > 700:
        return 0.0
    termine = 1.0
    somma = termine
    for n in range(1, 2000):
        termine *= z / (a + n)
        somma += termine
        if termine < 1e-16 * somma:
            break
    return max(0.0, 1 - somma * math.exp(a * math.log(z) - z - math.lgamma(a + 1)))


def test_ordine_markov(stati):
    """Rapporto di verosimiglianza tra catena di ordine 1 e di ordine 2.

    Restituisce (statistica, gradi di liberta', p-value).
    """
    s = stati.dropna().to_numpy().astype(int)
    n2 = np.zeros((STATI, STATI, STATI))
    for a, b, c in zip(s[:-2], s[1:-1], s[2:], strict=True):
        n2[a, b, c] += 1
    n1 = n2.sum(axis=0)
    riga2 = n2.sum(axis=2, keepdims=True)
    riga1 = n1.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll2 = np.nansum(np.where(n2 > 0, n2 * np.log(n2 / riga2), 0.0))
        ll1 = np.nansum(np.where(n1 > 0, n1 * np.log(n1 / riga1), 0.0))
    lr = 2 * (ll2 - ll1)
    gradi = STATI * (STATI - 1) ** 2
    return lr, gradi, sopravvivenza_chi2(lr, gradi)


def sopravvivenza_regimi(stati, durate=(5, 20, 60)):
    """Quota di periodi consecutivi nello stesso stato con durata almeno k, per stato.

    L'ultimo periodo (non concluso) e' escluso.
    """
    s = stati.dropna().to_numpy().astype(int)
    cambi = np.nonzero(np.diff(s) != 0)[0]
    estremi = np.concatenate(([-1], cambi, [len(s) - 1]))
    lunghezze = np.diff(estremi)[:-1]
    stato_periodo = s[estremi[1:-1]]
    tabella = {}
    for j in range(STATI):
        lung = lunghezze[stato_periodo == j]
        tabella[j] = {k: float((lung >= k).mean()) if len(lung) else float("nan") for k in durate}
    return pd.DataFrame(tabella).T


def sopravvivenza_geometrica(stati, durate=(5, 20, 60)):
    """Quota attesa di periodi con durata almeno k se la catena di ordine 1 fosse vera.

    Con probabilita' di restare nello stato j pari a p_jj, la durata e' geometrica:
    P(durata >= k) = p_jj^(k-1). Le p_jj sono stimate sullo stesso campione.
    """
    s = stati.dropna().to_numpy().astype(int)
    n = np.zeros((STATI, STATI))
    for a, b in zip(s[:-1], s[1:], strict=True):
        n[a, b] += 1
    restare = np.diag(n) / n.sum(axis=1)
    return pd.DataFrame({j: {k: float(restare[j] ** (k - 1)) for k in durate} for j in range(STATI)}).T
