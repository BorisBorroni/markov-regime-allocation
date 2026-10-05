"""Scarica i dati della biblioteca di Kenneth R. French e li salva in data/raw/french.

File: fattori giornalieri (mercato e tasso privo di rischio) e portafogli per
industria a 10 e 49 industrie, rendimenti giornalieri a valore ponderato con
dividendi reinvestiti. I dati non vengono versionati: ogni lettore li scarica da
solo. Prima di ogni richiesta lo script controlla il robots.txt del sito e si
ferma se il percorso e' vietato. I file gia' presenti non vengono riscaricati.

I rendimenti dei Treasury stanno gia' in data/treasury/cmt.csv; l'oro (GLD) va
messo a mano in data/raw/gld.us.txt (file giornaliero Stooq).
"""
import io
import sys
import time
import urllib.robotparser
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

DESTINAZIONE = Path(__file__).resolve().parents[1] / "data" / "raw" / "french"
INTESTAZIONE = {"User-Agent": "markov-regime-allocation (progetto universitario, uso personale)"}
ATTESA = 2
INDIRIZZO = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
ZIP = {
    "F-F_Research_Data_Factors_daily_CSV.zip": "F-F_Research_Data_Factors_daily.csv",
    "10_Industry_Portfolios_daily_CSV.zip": "10_Industry_Portfolios_Daily.csv",
    "49_Industry_Portfolios_daily_CSV.zip": "49_Industry_Portfolios_Daily.csv",
}


def permesso(url):
    """True se il robots.txt del sito consente di scaricare l'indirizzo."""
    parti = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{parti.scheme}://{parti.netloc}/robots.txt")
    rp.read()
    return rp.can_fetch(INTESTAZIONE["User-Agent"], url)


def main():
    DESTINAZIONE.mkdir(parents=True, exist_ok=True)
    for archivio, nome in ZIP.items():
        if (DESTINAZIONE / nome).exists():
            print(nome, "gia' presente")
            continue
        url = INDIRIZZO + archivio
        if not permesso(url):
            sys.exit(f"Vietato dal robots.txt: {url}")
        risposta = requests.get(url, headers=INTESTAZIONE, timeout=120)
        risposta.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(risposta.content)) as z:
            z.extractall(DESTINAZIONE)
        print(archivio, "scaricato")
        time.sleep(ATTESA)


if __name__ == "__main__":
    main()
