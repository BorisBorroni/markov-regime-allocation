"""Variante multi-scala: tre catene con memoria diversa, combinate con pesi che si adattano.

Ogni catena usa stati calcolati su una volatilita' EWMA con memoria diversa (circa 5 giorni, 3 settimane, 2 mesi di
borsa). Due usi, con la stessa regola di combinazione (pesi proporzionali a exp(-ETA * perdita media scontata)):
 - previsione_multiscala: previsione di varianza, perdita QLIKE (usata nella fase di confronto con EWMA e HAR);
 - probabilita_stress_multiscala: probabilita' di stress, perdita log-score su un evento comune (usata dalla strategia).
La perdita di una previsione fatta al giorno u e' nota solo al giorno u + h, quindi al giorno t si usano solo le
perdite fino a t - h. Gli iperparametri sono fissati in anticipo, non ottimizzati.
"""
import numpy as np
import pandas as pd

from .previsione import FLOOR, log_score, previsione_catena, qlike, varianza_futura
from .regimi import ORIZZONTE, Q_ALTO, probabilita_stress, stati_as_of, volatilita_ewma

# memoria in giorni di borsa -> lambda = 1 - 2 / (N + 1)
MEMORIE = (5, 15, 42)
ETA = 10.0
SCONTO = 0.99


def lambda_da_memoria(giorni):
    return 1 - 2 / (giorni + 1)


def previsioni_per_scala(rendimenti, h=ORIZZONTE, memorie=MEMORIE):
    """Previsione di varianza di ciascuna catena: DataFrame con una colonna per scala."""
    colonne = {}
    for n in memorie:
        stati = stati_as_of(volatilita_ewma(rendimenti, lambda_da_memoria(n)))
        colonne[n] = previsione_catena(rendimenti, stati, h)
    return pd.DataFrame(colonne)


def pesi_adattivi(rendimenti, previsioni, h=ORIZZONTE, eta=ETA, sconto=SCONTO):
    """Pesi (righe che sommano a 1) basati sulle perdite gia' note al giorno t."""
    reale = varianza_futura(rendimenti, h)
    perdite = previsioni.apply(lambda f: qlike(reale, f.clip(lower=FLOOR)))
    media = perdite.ewm(alpha=1 - sconto, adjust=False, ignore_na=True).mean().shift(h)
    media = media.ffill()
    # una scala senza perdita nota riceve la perdita media delle altre
    m = media.to_numpy()
    noti = ~np.isnan(m)
    medie_riga = np.where(noti.any(axis=1), np.where(noti, m, 0.0).sum(axis=1) / np.maximum(noti.sum(axis=1), 1), np.nan)
    media = pd.DataFrame(np.where(noti, m, medie_riga[:, None]), index=media.index, columns=media.columns)
    punteggio = -eta * media
    punteggio = punteggio.sub(punteggio.max(axis=1), axis=0)
    w = np.exp(punteggio)
    w = w.div(w.sum(axis=1), axis=0)
    # senza nessuna perdita nota: pesi uguali
    return w.fillna(1 / previsioni.shape[1])


def previsione_multiscala(rendimenti, h=ORIZZONTE, memorie=MEMORIE, eta=ETA, sconto=SCONTO):
    """Previsione combinata e pesi, entrambi calcolati senza guardare il futuro."""
    f = previsioni_per_scala(rendimenti, h, memorie)
    w = pesi_adattivi(rendimenti, f, h, eta, sconto)
    comb = (f * w).sum(axis=1, min_count=1)
    comb[f.isna().any(axis=1)] = np.nan
    return comb, w


# --- probabilita' di stress multi-scala (opzione A) ---

SCALA_RIFERIMENTO = 15


def stati_per_scala(rendimenti, memorie=MEMORIE, q_alto=Q_ALTO):
    return {n: stati_as_of(volatilita_ewma(rendimenti, lambda_da_memoria(n)), q_alto=q_alto) for n in memorie}


def probabilita_stress_multiscala(rendimenti, h=ORIZZONTE, memorie=MEMORIE, eta=ETA, sconto=SCONTO,
                                  riferimento=SCALA_RIFERIMENTO, q_alto=Q_ALTO):
    """Probabilita' di stress combinata di tre catene, con pesi dati dal log-score su un evento comune.

    Ogni catena n ha la sua probabilita' P_n^h[stato_n, stress_n]. L'evento comune su cui si
    misurano i pesi e' lo stress della scala di riferimento tra h giorni. L'esito del giorno u
    e' noto al giorno u + h: al giorno t si usano solo punteggi fino a t - h.
    """
    stati = stati_per_scala(rendimenti, memorie, q_alto)
    p = pd.DataFrame({n: probabilita_stress(s, h) for n, s in stati.items()})
    s_rif = stati[riferimento]
    esito = (s_rif.shift(-h) == 2).astype(float).where(s_rif.shift(-h).notna())
    perdite = p.apply(lambda q: log_score(q, esito))
    media = perdite.ewm(alpha=1 - sconto, adjust=False, ignore_na=True).mean().shift(h).ffill()
    m = media.to_numpy()
    noti = ~np.isnan(m)
    medie_riga = np.where(noti.any(axis=1), np.where(noti, m, 0.0).sum(axis=1) / np.maximum(noti.sum(axis=1), 1), np.nan)
    media = pd.DataFrame(np.where(noti, m, medie_riga[:, None]), index=media.index, columns=media.columns)
    punteggio = -eta * media
    punteggio = punteggio.sub(punteggio.max(axis=1), axis=0)
    w = np.exp(punteggio)
    w = w.div(w.sum(axis=1), axis=0).fillna(1 / p.shape[1])
    comb = (p * w).sum(axis=1, min_count=1)
    comb[p.isna().any(axis=1)] = np.nan
    return comb, w, p, esito
