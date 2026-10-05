"""Scrive config_congelata.json con le configurazioni scelte sullo sviluppo.

Va eseguito una sola volta, prima dell'esame sul test e dopo 04 e 05. Dopo la scrittura le configurazioni non si cambiano.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markov_regime_allocation.esperimento import config_da_tabelle  # noqa: E402

RADICE = Path(__file__).resolve().parents[1]
OUT = RADICE / "output"
DESTINAZIONE = RADICE / "config_congelata.json"


def main():
    if DESTINAZIONE.exists():
        sys.exit("config_congelata.json esiste gia': non si sovrascrive.")
    cfg = config_da_tabelle(pd.read_csv(OUT / "griglia_sviluppo.csv"), pd.read_csv(OUT / "concorrenti_sviluppo.csv"))
    DESTINAZIONE.write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
    print(json.dumps(cfg, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
