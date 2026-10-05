"""Caricamento e preparazione dei dati.

Tutte le serie sono rendimenti giornalieri semplici in decimale (0.01 = 1%).
I dati grezzi stanno in data/raw (non versionati), il Tesoro in data/treasury.
"""
from pathlib import Path

import pandas as pd

DATI = Path(__file__).resolve().parents[2] / "data"
FINE = "2026-06-30"
INIZIO_OPERATIVO = "2005-02-25"
MANCANTE = (-99.99, -999.0)
FILE_FRENCH = ("F-F_Research_Data_Factors_daily.csv", "10_Industry_Portfolios_Daily.csv",
               "49_Industry_Portfolios_Daily.csv")


def leggi_french(percorso, sezione="Average Value Weighted Returns"):
    """Legge una sezione di un file della biblioteca di French (da % a decimali)."""
    righe = Path(percorso).read_text().splitlines()
    titolo = next((i for i, r in enumerate(righe) if sezione in r), 0)
    inizio = next(i for i in range(titolo, len(righe)) if righe[i].startswith(","))
    colonne = [c.strip() for c in righe[inizio].split(",")][1:]
    date, valori = [], []
    for r in righe[inizio + 1:]:
        parti = r.split(",")
        giorno = parti[0].strip()
        if len(giorno) != 8 or not giorno.isdigit():
            break
        date.append(pd.Timestamp(giorno))
        valori.append([float(x) for x in parti[1:]])
    tabella = pd.DataFrame(valori, index=pd.DatetimeIndex(date), columns=colonne)
    return tabella.where(~tabella.isin(MANCANTE)) / 100


def leggi_prezzi_stooq(percorso):
    """Prezzi di chiusura da un file Stooq."""
    x = pd.read_csv(percorso)
    x.columns = [c.strip("<>").lower() for c in x.columns]
    date = pd.to_datetime(x["date"].astype(str), format="%Y%m%d")
    return pd.Series(x["close"].to_numpy(), index=date)


def prezzo_par(y, cedola, anni):
    """Prezzo (in frazione del nominale) di un titolo a cedola semestrale."""
    v = 1 / (1 + y / 2)
    return cedola / y * (1 - v ** (2 * anni)) + v ** (2 * anni)


def durata_convessita(y, anni, h=1e-4):
    """Durata modificata e convessita' di un titolo alla pari, per differenze finite."""
    p0 = prezzo_par(y, y, anni)
    su, giu = prezzo_par(y + h, y, anni), prezzo_par(y - h, y, anni)
    return -(su - giu) / (2 * h * p0), (su - 2 * p0 + giu) / (h ** 2 * p0)


def rendimento_obbligazione(rendimento, anni=7):
    """Rendimento totale giornaliero stimato di un titolo alla pari a `anni` anni.

    `rendimento` e' il rendimento a scadenza costante in decimale, con indice di date.
    Cedola maturata sui giorni di calendario piu' effetto di durata e convessita'.
    Il rullaggio lungo la curva e' ignorato.
    """
    giorni = rendimento.index.to_series().diff().dt.days / 365
    precedente = rendimento.shift(1)
    variazione = rendimento.diff()
    durata, convessita = durata_convessita(precedente, anni)
    r = precedente * giorni - durata * variazione + 0.5 * convessita * variazione ** 2
    return r.dropna()


def leggi_tesoro(percorso=DATI / "treasury" / "cmt.csv"):
    """Rendimenti a scadenza costante del Tesoro USA (in decimale)."""
    return pd.read_csv(percorso, index_col=0, parse_dates=True) / 100


def controlla_file():
    """Ferma il programma con un messaggio chiaro se mancano i dati grezzi."""
    mancanti = [p for p in (*(DATI / "raw" / "french" / n for n in FILE_FRENCH), DATI / "raw" / "gld.us.txt")
                if not p.exists()]
    if mancanti:
        elenco = "\n  ".join(str(p) for p in mancanti)
        raise FileNotFoundError(
            f"Dati mancanti:\n  {elenco}\nI file di French si scaricano con scripts/00_scarica_dati.py; "
            "il file giornaliero di GLD (formato Stooq) va messo a mano in data/raw/gld.us.txt."
        )


def carica_serie(fine=FINE):
    """Rendimenti giornalieri di tutte le serie sul calendario di borsa di French.

    Colonne: Mkt, NoDur, Chips, HiTec, Bond7, Oro, RF. Il rendimento di Bond7 e Oro
    esiste dal 2005-02-25 (oro) e dal 2000 (Tesoro); le azioni da prima.
    """
    controlla_file()
    fattori = leggi_french(DATI / "raw" / "french" / "F-F_Research_Data_Factors_daily.csv")
    ind10 = leggi_french(DATI / "raw" / "french" / "10_Industry_Portfolios_Daily.csv")
    ind49 = leggi_french(DATI / "raw" / "french" / "49_Industry_Portfolios_Daily.csv")
    calendario = fattori.index[fattori.index <= fine]
    serie = pd.DataFrame(index=calendario)
    serie["RF"] = fattori["RF"]
    serie["Mkt"] = fattori["Mkt-RF"] + fattori["RF"]
    serie["NoDur"] = ind10["NoDur"]
    serie["HiTec"] = ind10["HiTec"]
    serie["Chips"] = ind49["Chips"]
    tesoro = leggi_tesoro()["7Yr"]
    tesoro = tesoro.reindex(tesoro.index.union(calendario)).ffill().reindex(calendario)
    serie["Bond7"] = rendimento_obbligazione(tesoro.dropna())
    oro = leggi_prezzi_stooq(DATI / "raw" / "gld.us.txt")
    oro = oro.reindex(oro.index.union(calendario)).ffill().reindex(calendario)
    serie["Oro"] = oro.pct_change()
    return serie
