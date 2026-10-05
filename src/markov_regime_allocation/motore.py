"""Nucleo a rischio ponderato, regola di esposizione e simulazione con costi.

Tempi: il segnale usa i dati fino alla chiusura del giorno d (ultimo giorno di borsa
della settimana). L'ordine viene eseguito alla chiusura del giorno dopo e il portafoglio
guadagna dal rendimento del giorno ancora successivo. Tra un ribilanciamento e l'altro i
pesi derivano con i prezzi.

Costi: si applicano al turnover delle sole gambe rischiose (la liquidita' non ha costo).
Il costo dell'ordine si sottrae al rendimento di quel giorno e i nuovi pesi sono quelli
obiettivo (approssimazione di secondo ordine).
"""
import numpy as np
import pandas as pd

FINESTRA = 60
AVVIAMENTO = 504


def pesi_nucleo(rendimenti, finestra=FINESTRA):
    """Pesi proporzionali all'inverso della volatilita' a `finestra` giorni, somma uno."""
    inv = 1 / rendimenti.rolling(finestra).std()
    return inv.div(inv.sum(axis=1), axis=0)


def volatilita_nucleo(rendimenti, pesi, finestra=FINESTRA):
    """Deviazione standard annualizzata della serie ottenuta applicando i pesi di oggi agli ultimi `finestra` rendimenti."""
    r = rendimenti.to_numpy()
    w = pesi.to_numpy()
    n = len(r)
    out = np.full(n, np.nan)
    finestre = np.lib.stride_tricks.sliding_window_view(r, finestra, axis=0)  # (n-f+1, assets, f)
    serie = np.einsum("tjf,tj->tf", finestre, w[finestra - 1:])
    out[finestra - 1:] = serie.std(axis=1, ddof=1) * np.sqrt(252)
    out[np.isnan(w).any(axis=1)] = np.nan
    return pd.Series(out, index=rendimenti.index)


def sigma_riferimento(vol_nucleo, avviamento=AVVIAMENTO, finestra=FINESTRA):
    """Mediana della volatilita' del nucleo dal giorno `finestra` al giorno `avviamento`, poi costante."""
    return float(vol_nucleo.iloc[finestra - 1:avviamento].median())


def esposizione(p, vol_nucleo, sigma_rif, m, r):
    """L = min(1, m * sigma_rif * (1 - (1 - r) * p) / sigma_nucleo). Senza leva."""
    return (m * sigma_rif * (1 - (1 - r) * p) / vol_nucleo).clip(upper=1.0)


def giorni_decisione(indice):
    """Ultimo giorno di borsa di ogni settimana di calendario (booleano)."""
    settimana = indice.to_period("W")
    return pd.Series(settimana[1:] != settimana[:-1], index=indice[:-1]).reindex(indice, fill_value=False)


def simula(rendimenti, rf, pesi_obiettivo, decisioni, costo_bp=5.0):
    """Rendimenti giornalieri netti del portafoglio e turnover.

    rendimenti: DataFrame (giorni x strumenti rischiosi); rf: Series; pesi_obiettivo: DataFrame
    con i pesi desiderati calcolati a fine giornata (NaN dove non si opera); decisioni: Series booleana.
    Restituisce DataFrame con colonne ritorno, turnover, esposizione (peso rischioso a inizio giornata).
    """
    R = rendimenti.to_numpy()
    cash = rf.to_numpy()
    W = pesi_obiettivo.to_numpy()
    dec = decisioni.to_numpy()
    n, k = R.shape
    h = np.zeros(k)
    in_attesa = None
    ritorno = np.zeros(n)
    turnover = np.zeros(n)
    esp = np.zeros(n)
    for i in range(n):
        esp[i] = h.sum()
        lordo = h @ R[i] + (1 - h.sum()) * cash[i]
        h = h * (1 + R[i]) / (1 + lordo)
        costo = 0.0
        if in_attesa is not None:
            t = np.abs(in_attesa - h).sum()
            turnover[i] = t
            costo = costo_bp * 1e-4 * t
            h = in_attesa
            in_attesa = None
        ritorno[i] = lordo - costo
        if dec[i] and not np.isnan(W[i]).any():
            in_attesa = W[i].copy()
    return pd.DataFrame({"ritorno": ritorno, "turnover": turnover, "esposizione": esp}, index=rendimenti.index)


def metriche(ritorni, rf, inizio=None):
    """Sharpe annualizzato sui rendimenti in eccesso a RF, rendimento composto annuo, volatilita', drawdown massimo."""
    r = ritorni.copy()
    if inizio is not None:
        r = r[r.index >= inizio]
    eccesso = r["ritorno"] - rf.reindex(r.index)
    anni = len(r) / 252
    valore = (1 + r["ritorno"]).cumprod()
    valore_con_inizio = pd.concat([pd.Series([1.0]), valore.reset_index(drop=True)], ignore_index=True)
    return {
        "sharpe": float(eccesso.mean() / eccesso.std(ddof=1) * np.sqrt(252)),
        "rendimento": float(valore.iloc[-1] ** (1 / anni) - 1),
        "volatilita": float(r["ritorno"].std(ddof=1) * np.sqrt(252)),
        "drawdown": float((valore_con_inizio / valore_con_inizio.cummax() - 1).min()),
        "turnover_annuo": float(r["turnover"].sum() / anni),
        "esposizione_media": float(r["esposizione"].mean()),
    }
