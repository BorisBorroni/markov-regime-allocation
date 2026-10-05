"""Regola di trend: uno strumento e' in portafoglio solo se il prezzo supera la media mobile semplice.

I prezzi sono ricostruiti dai rendimenti totali (indice cumulato). I pesi sono quelli inverso-volatilita'
del nucleo per gli strumenti in portafoglio; la quota degli altri resta in liquidita'.
"""


def indice_prezzi(rendimenti):
    return (1 + rendimenti.fillna(0.0)).cumprod()


def in_tendenza(rendimenti, finestra=200):
    prezzi = indice_prezzi(rendimenti)
    media = prezzi.rolling(finestra).mean()
    return (prezzi > media).astype(float).where(media.notna())


def pesi_trend(rendimenti, pesi_nucleo, finestra=200):
    segnale = in_tendenza(rendimenti, finestra)
    return pesi_nucleo * segnale
