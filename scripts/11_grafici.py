"""Genera i grafici in img/ per il periodo di test, con gli stessi calcoli dell'esame (modulo `esame`).

Non sceglie nulla: usa le configurazioni congelate (config_congelata.json) e ricalcola le serie. Come controllo,
verifica che gli Sharpe ricalcolati coincidano con quelli salvati dall'esame (output/metriche_test.csv).
Serve output/potenza.csv (prodotto da 07_potenza.py) e output/bootstrap_test.json (da 06_inferenza.py test).
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markov_regime_allocation import grafici, multiscala, regimi  # noqa: E402
from markov_regime_allocation.esame import NOMI_ORDINATI, prepara_esame, simulazioni_complete, tabella_metriche  # noqa: E402

RADICE = Path(__file__).resolve().parents[1]
OUT, IMG = RADICE / "output", RADICE / "img"


def main():
    IMG.mkdir(exist_ok=True)
    cfg = json.loads((RADICE / "config_congelata.json").read_text())
    ctx = prepara_esame("test", cfg)
    sims = simulazioni_complete(ctx, 5.0)
    met = tabella_metriche(ctx, sims)
    salvate = pd.read_csv(OUT / "metriche_test.csv", index_col=0)
    assert np.allclose(met["sharpe"], salvate.loc[met.index, "sharpe"], atol=1e-9), "Sharpe ricalcolati diversi dall'esame"
    print("Controllo: gli Sharpe del test ricalcolati coincidono con output/metriche_test.csv")

    w = multiscala.probabilita_stress_multiscala(ctx.mkt, q_alto=cfg["multi"]["q_alto"])[1].reindex(ctx.R.index)
    stati_mkt = regimi.stati_as_of(regimi.volatilita_ewma(ctx.mkt), q_alto=0.90)
    matrice = regimi.matrici_transizione(stati_mkt)[-1]
    stati = stati_mkt.reindex(ctx.R.index)
    boot = json.loads((OUT / "bootstrap_test.json").read_text())
    met_sv = pd.read_csv(OUT / "metriche_sviluppo.csv", index_col=0)
    nomi_sv = ["mercato", "60_40", "nucleo_statico", "vol_target_pari", "trend", "hmm", "logistica", "base", "multi"]
    figure = {
        "equity_test": grafici.fig_equity(sims, ctx.finestra),
        "drawdown_test": grafici.fig_drawdown(sims, ctx.finestra),
        "esposizione_test": grafici.fig_esposizione(sims, stati, ctx.finestra),
        "pesi_scale_test": grafici.fig_pesi_scale(w, ctx.finestra),
        "differenze_bootstrap": grafici.fig_forest(boot),
        "regimi": grafici.fig_regimi(matrice, ctx.vol, stati, ctx.finestra),
        "sharpe_sviluppo_test": grafici.fig_sviluppo_test(met_sv, met, nomi_sv),
        "drawdown_massimo_test": grafici.fig_drawdown_massimo(met, [n for n in NOMI_ORDINATI if n != "vol_target"]),
    }
    if (OUT / "potenza.csv").exists():
        figure["potenza"] = grafici.fig_potenza(pd.read_csv(OUT / "potenza.csv"))
    for nome, fig in figure.items():
        fig.savefig(IMG / f"{nome}.png")
        print("scritto", f"img/{nome}.png")


if __name__ == "__main__":
    main()
