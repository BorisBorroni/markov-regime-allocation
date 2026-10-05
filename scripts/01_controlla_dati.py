"""Controlli di qualita' sui dati, senza guardare il periodo di test.

Le statistiche di rendimento sono calcolate solo fino al 2014-12-31.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markov_regime_allocation import dati  # noqa: E402

SVILUPPO_FINE = "2014-12-31"


def main():
    s = dati.carica_serie()
    print("Calendario:", s.index.min().date(), "->", s.index.max().date(), len(s), "giorni")
    print("\nValori mancanti per serie (su tutto il calendario) e prima data valida:")
    for c in s:
        print(f"  {c:6s} mancanti={s[c].isna().sum():6d}  prima data={s[c].first_valid_index().date()}")
    op = s[s.index >= dati.INIZIO_OPERATIVO]
    print("\nDal 2005-02-25, mancanti per serie:", op.isna().sum().to_dict())
    print("Giorni di calendario di French senza prezzo GLD proprio:")
    oro = dati.leggi_prezzi_stooq(dati.DATI / "raw" / "gld.us.txt")
    cal = op.index
    print("  ", [str(d.date()) for d in cal.difference(oro.index)][:10], "| date GLD fuori dal calendario:", [str(d.date()) for d in oro.index[oro.index >= dati.INIZIO_OPERATIVO].difference(cal)][:10])
    tes = dati.leggi_tesoro()
    print("  giorni di borsa senza rendimento Tesoro dal 2005:", len(cal.difference(tes.index)))
    print("\nRendimenti giornalieri oltre 15% in valore assoluto (tutto il periodo):")
    for c in ["Mkt", "NoDur", "Chips", "HiTec", "Bond7", "Oro"]:
        grandi = s[c][s[c].abs() > 0.15]
        print(f"  {c:6s}", [(str(d.date()), round(v, 3)) for d, v in grandi.items()])
    dev = op[op.index <= SVILUPPO_FINE]
    print("\nStatistiche solo sviluppo (dal 2005-02-25 al 2014-12-31):")
    tab = pd.DataFrame({"media_ann%": dev.mean() * 252 * 100, "vol_ann%": dev.std() * np.sqrt(252) * 100, "min_giorno%": dev.min() * 100, "max_giorno%": dev.max() * 100})
    print(tab.round(2).to_string())
    print("\nCorrelazioni giornaliere (sviluppo):")
    print(dev[["Mkt", "NoDur", "Chips", "HiTec", "Bond7", "Oro"]].corr().round(2).to_string())


if __name__ == "__main__":
    main()
