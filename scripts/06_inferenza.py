"""Inferenza completa: confronto della strategia con i controlli, bootstrap, costi, sottoperiodi, commissione, DSR, alfa, regimi, placebo.

Modi:
  sviluppo  prova generale sul periodo di sviluppo (fino al 2014-12-31), con configurazioni scelte sullo sviluppo.
  test      esame sul periodo di test (dal 2015-01-01): richiede config_congelata.json e UNA sola esecuzione.
Le configurazioni (m, r, q_alto) valgono per tutta la serie; le metriche si calcolano solo nella finestra di valutazione.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markov_regime_allocation import inferenza, motore  # noqa: E402
from markov_regime_allocation.esame import (  # noqa: E402
    configurazioni_da_sviluppo, prepara_esame, simula_tutti, simulazioni_complete,
    tabella_metriche,
)
from markov_regime_allocation.esperimento import esegui  # noqa: E402

RADICE = Path(__file__).resolve().parents[1]
OUT = RADICE / "output"
CONFIG = RADICE / "config_congelata.json"
# configurazioni valutate per la selezione: 57 celle di griglia (multi 12, base 12, stato 12, vol targeting 3, HMM 6,
# logistica 12) e 3 finestre di trend; 192 e' un valore prudenziale (circa il triplo)
N_TENTATIVI = 60
N_TENTATIVI_PRUDENZIALE = 192
ESITO = OUT / "esito_test.txt"
FINE_ANALISI = "FINE ANALISI"
B = 10000
N_PLACEBO = 1000


class Tee:
    """Scrive su schermo e su file: il risultato del test resta registrato."""

    def __init__(self, percorso):
        self.file = open(percorso, "w")
        self.schermo = sys.stdout

    def write(self, testo):
        self.schermo.write(testo)
        self.file.write(testo)
        self.file.flush()

    def flush(self):
        self.schermo.flush()
        self.file.flush()


def main(modo):
    if modo == "test":
        if not CONFIG.exists():
            sys.exit("config_congelata.json non trovato: congelare prima le configurazioni.")
        if ESITO.exists() and FINE_ANALISI in ESITO.read_text():
            sys.exit("Il test e' gia' stato eseguito (output/esito_test.txt): non si ripete.")
        cfg = json.loads(CONFIG.read_text())
        if cfg != configurazioni_da_sviluppo():
            sys.exit("config_congelata.json non coincide con le scelte delle tabelle di sviluppo.")
        sys.stdout = Tee(ESITO)
    else:
        cfg = configurazioni_da_sviluppo()
    ctx = prepara_esame(modo, cfg)
    R, rf, w, vol, sigma_rif, dec = ctx.R, ctx.rf, ctx.w, ctx.vol, ctx.sigma_rif, ctx.dec
    p, stati, finestra = ctx.p, ctx.stati, ctx.finestra
    inizio_val, fine_val = ctx.inizio_val, ctx.fine_val
    print(f"Modo {modo}: dati fino al {R.index[-1].date()}, finestra di valutazione {inizio_val.date()} -> {fine_val.date()}")
    print("Configurazioni:", json.dumps(cfg))
    sims = simulazioni_complete(ctx, 5.0)

    def win(sim):
        return sim["ritorno"][finestra].to_numpy()

    rfw = rf[finestra].to_numpy()
    pd.options.display.float_format = "{:.3f}".format
    tab = tabella_metriche(ctx, sims)
    print("\nMetriche nella finestra (5 bp):")
    print(tab.to_string())
    tab.to_csv(OUT / f"metriche_{modo}.csv")
    print(f"Vol targeting a esposizione pari: m_d={sims['_m_d_pari']:.3f}")

    print(f"\nBootstrap stazionario ({B} ricampionamenti, blocco medio 20): differenza multi meno controllo")
    print("(dDD = drawdown multi meno drawdown controllo: positivo = multi meno profonda)")
    risultati = {}
    for nome in ("vol_target_pari", "esposizione_costante", "vol_target", "stato", "base", "hmm", "logistica", "trend", "nucleo_statico", "mercato", "60_40"):
        r = inferenza.differenza_bootstrap(win(sims["multi"]), win(sims[nome]), rfw, B=B, seed=11)
        risultati[nome] = r
        print(f"  contro {nome:22s} dSharpe={r['sharpe_diff']:+.3f} IC95=[{r['sharpe_ic'][0]:+.3f}, {r['sharpe_ic'][1]:+.3f}]  dDD={r['drawdown_diff']:+.3f} IC95=[{r['drawdown_ic'][0]:+.3f}, {r['drawdown_ic'][1]:+.3f}]")

    centro = inizio_val + (fine_val - inizio_val) / 2
    meta = pd.Timestamp(year=centro.year, month=1, day=1) if modo == "sviluppo" else pd.Timestamp("2020-01-01")
    print(f"\nSottoperiodi (divisione al {meta.date()}), differenza di Sharpe multi meno controllo:")
    for nome in ("vol_target_pari", "esposizione_costante"):
        out = []
        for a, b in ((inizio_val, meta - pd.Timedelta(days=1)), (meta, fine_val)):
            m = (R.index >= a) & (R.index <= b)
            x, y, z = sims["multi"]["ritorno"][m].to_numpy(), sims[nome]["ritorno"][m].to_numpy(), rf[m].to_numpy()
            out.append(inferenza.sharpe(x - z) - inferenza.sharpe(y - z))
        print(f"  contro {nome:22s} prima parte {out[0]:+.3f}  seconda parte {out[1]:+.3f}")

    print("\nSensibilita' ai costi, differenza di Sharpe multi meno vol targeting a esposizione pari:")
    for c in (0.0, 10.0, 25.0):
        s = simula_tutti(ctx, c)
        x, y = s["multi"]["ritorno"][finestra].to_numpy(), s["vol_target_pari"]["ritorno"][finestra].to_numpy()
        print(f"  {c:4.0f} bp: {inferenza.sharpe(x - rfw) - inferenza.sharpe(y - rfw):+.3f}")

    print("\nCommissione di performance annua (multi contro vol targeting a esposizione pari):")
    for g in (1, 5, 10):
        f = inferenza.commissione_performance(win(sims["multi"]), win(sims["vol_target_pari"]), g)
        print(f"  gamma={g:2d}: {f * 1e4:+.1f} punti base")

    print("\nAlfa della multi sul nucleo statico (rendimenti in eccesso, Newey-West):")
    for lag in (20, 60):
        a = inferenza.alfa_contro_nucleo(win(sims["multi"]), win(sims["nucleo_statico"]), rfw, lag)
        print(f"  ritardi {lag}: alfa annuo {a['alfa_annuo'] * 100:+.2f}%, t={a['t_alfa']:+.2f}, beta={a['beta']:.2f}")

    grid_d = pd.read_csv(OUT / "griglia_sviluppo.csv")
    conc_d = pd.read_csv(OUT / "concorrenti_sviluppo.csv")
    sh_giorn = pd.concat([grid_d["sharpe"], conc_d["sharpe"]]).to_numpy() / np.sqrt(252)
    ecc_multi = win(sims["multi"]) - rfw
    var_sh = float(np.var(sh_giorn, ddof=1))
    for n_t in (N_TENTATIVI, N_TENTATIVI_PRUDENZIALE):
        print(f"\nDeflated Sharpe della multi con {n_t} tentativi (varianza degli Sharpe dalle griglie di sviluppo): {inferenza.deflated_sharpe(ecc_multi, n_t, var_sh):.3f}")

    print("\nTabella per regime (stato del giorno prima), multi e nucleo statico:")
    tr = inferenza.tabella_per_regime(sims["multi"]["ritorno"][finestra], stati, rf)
    tn = inferenza.tabella_per_regime(sims["nucleo_statico"]["ritorno"][finestra], stati, rf)
    print(pd.concat({"multi": tr, "nucleo": tn}, axis=1).to_string())

    rng = np.random.default_rng(7)
    valido = np.nonzero(p["multi"].notna().to_numpy())[0]
    base_p = p["multi"].to_numpy()[valido]
    sharpe_reale = inferenza.sharpe(ecc_multi)
    c = cfg["multi"]
    batte = 0
    for _ in range(N_PLACEBO):
        s = int(rng.integers(252, len(valido) - 252))
        q = p["multi"].copy()
        q.iloc[valido] = np.roll(base_p, s)
        sim = esegui(R, rf, w, dec, motore.esposizione(q, vol, sigma_rif, c["m"], c["r"]))
        if inferenza.sharpe(sim["ritorno"][finestra].to_numpy() - rfw) >= sharpe_reale:
            batte += 1
    print(f"\nPlacebo con p traslata ciclicamente ({N_PLACEBO} traslazioni): quota con Sharpe pari o superiore = {batte / N_PLACEBO:.3f}")
    (OUT / f"bootstrap_{modo}.json").write_text(json.dumps(risultati, indent=1))
    print(f"\n{FINE_ANALISI}")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "sviluppo"
    if modo not in ("sviluppo", "test"):
        sys.exit("uso: 06_inferenza.py [sviluppo|test]")
    main(modo)
