"""Grafici del progetto. Ogni funzione restituisce una figura matplotlib e non scrive su disco.

Colori per entita', sempre gli stessi in tutti i grafici (tavolozza categorica in ordine fisso):
blu = strategia Markov multi-scala, arancio = vol targeting, acqua = mercato, giallo = nucleo statico, magenta = 60/40.
Il testo usa sempre i colori dell'inchiostro, mai quello della serie: l'identita' e' data da legenda e etichette dirette.
"""
import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.edgecolor": "#c9c8c2", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 150,
})
import matplotlib.pyplot as plt  # noqa: E402

INCHIOSTRO, SECONDARIO = "#0b0b0b", "#52514e"
COLORI = {"multi": "#2a78d6", "vol_target_pari": "#eb6834", "mercato": "#1baf7a",
          "nucleo_statico": "#eda100", "60_40": "#e87ba4"}
ETICHETTE = {"multi": "Markov multi-scala", "vol_target_pari": "Vol targeting (stessa esposizione)",
             "mercato": "Mercato", "nucleo_statico": "Nucleo statico", "60_40": "60/40",
             "vol_target": "Vol targeting", "esposizione_costante": "Esposizione costante",
             "trend": "Trend SMA 200", "hmm": "HMM", "logistica": "Logistica", "stato": "Stato di oggi",
             "base": "Catena base"}
BREVI = {**ETICHETTE, "vol_target_pari": "Vol targeting", "esposizione_costante": "Esposiz. costante"}
FONTE = "Dati: Kenneth R. French Data Library (rendimenti giornalieri), costi 5 bp. Periodo di test."


def _etichette_dirette(ax, finali, colori, minimo=0.045):
    """Etichette a destra delle linee, spaziate per non sovrapporsi (distanza minima in frazione d'asse)."""
    y0, y1 = ax.get_ylim()
    trasforma = np.log if ax.get_yscale() == "log" else (lambda v: v)
    f = {k: (trasforma(v) - trasforma(y0)) / (trasforma(y1) - trasforma(y0)) for k, v in finali.items()}
    ordine = sorted(f, key=f.get)
    pos = {}
    for k in ordine:
        pos[k] = max(f[k], pos[ordine[ordine.index(k) - 1]] + minimo) if k != ordine[0] else f[k]
    for k in ordine:
        ax.annotate(BREVI[k], xy=(1.0, f[k]), xytext=(1.012, pos[k]), xycoords="axes fraction",
                    textcoords="axes fraction", va="center", fontsize=9, color=INCHIOSTRO,
                    arrowprops={"arrowstyle": "-", "color": colori[k], "lw": 0.8, "shrinkA": 0, "shrinkB": 0})


def _firma(fig, testo=FONTE):
    fig.text(0.01, 0.005, testo, fontsize=7.5, color=SECONDARIO, ha="left", va="bottom")


def _ombra_stress(ax, stati, finestra):
    s = (stati[finestra] == 2).astype(int)
    inizio = None
    giorni = s.index
    for i, v in enumerate(s.to_numpy()):
        if v and inizio is None:
            inizio = giorni[i]
        if (not v or i == len(s) - 1) and inizio is not None:
            ax.axvspan(inizio, giorni[i], color="#9a998f", alpha=0.18, lw=0)
            inizio = None


def serie_finestra(sims, nome, finestra):
    return sims[nome]["ritorno"][finestra]


def valore(sims, nome, finestra):
    return (1 + serie_finestra(sims, nome, finestra)).cumprod()


def fig_equity(sims, finestra, nomi=("mercato", "60_40", "nucleo_statico", "vol_target_pari", "multi")):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    finali = {}
    for n in nomi:
        v = valore(sims, n, finestra)
        ax.plot(v.index, v, color=COLORI[n], lw=2.0 if n == "multi" else 1.4, label=ETICHETTE[n])
        finali[n] = v.iloc[-1]
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 4])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel("Valore di 1 euro investito (scala log)")
    ax.set_title("Crescita del capitale nel test (al netto dei costi)")
    ax.margins(x=0.01)
    _etichette_dirette(ax, finali, COLORI)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.subplots_adjust(right=0.84, bottom=0.12)
    _firma(fig)
    return fig


def fig_drawdown(sims, finestra, nomi=("mercato", "nucleo_statico", "vol_target_pari", "multi")):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    finali = {}
    for n in nomi:
        v = valore(sims, n, finestra)
        dd = v / v.cummax() - 1
        ax.plot(dd.index, dd * 100, color=COLORI[n], lw=2.0 if n == "multi" else 1.3, label=ETICHETTE[n])
        finali[n] = dd.min() * 100
    ax.set_ylabel("Distanza dal massimo precedente (%)")
    ax.set_title("Drawdown nel test")
    ax.margins(x=0.01)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)
    fig.subplots_adjust(bottom=0.13)
    _firma(fig)
    return fig


def fig_esposizione(sims, stati, finestra):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    _ombra_stress(ax, stati, finestra)
    for n in ("vol_target_pari", "multi"):
        e = sims[n]["esposizione"][finestra]
        ax.plot(e.index, e, color=COLORI[n], lw=1.6 if n == "multi" else 1.1, label=ETICHETTE[n])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Quota investita nel nucleo rischioso")
    ax.set_title("Quanto rischio prende ogni regola (grigio = giorni in stress)")
    ax.margins(x=0.01)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)
    fig.subplots_adjust(bottom=0.13)
    _firma(fig)
    return fig


def fig_pesi_scale(w, finestra):
    fig, ax = plt.subplots(figsize=(10, 3.9))
    x = w[finestra].rolling(20, min_periods=1).mean()
    colori = ["#a8c9f0", "#5b9be0", "#1d5aa6"]
    ax.stackplot(x.index, x.T.to_numpy(), colors=colori, edgecolor="#fcfcfb", linewidth=1.2,
                 labels=[f"memoria {n} giorni" for n in x.columns])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Peso della catena (media mobile 20 giorni)")
    ax.set_title("Quale memoria pesa di piu' nella catena multi-scala")
    ax.margins(x=0)
    ax.legend(loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(bottom=0.24)
    _firma(fig, "Pesi calcolati con il log-score fino a 5 giorni prima. Periodo di test.")
    return fig


def fig_forest(bootstrap, nomi=("vol_target_pari", "base", "stato", "hmm", "logistica", "trend",
                                "nucleo_statico", "mercato", "60_40")):
    fig, assi = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, (chiave, titolo) in zip(assi, (("sharpe", "Differenza di Sharpe"),
                                           ("drawdown", "Differenza di drawdown massimo (punti %)")),
                                 strict=True):
        y = np.arange(len(nomi))[::-1]
        for yi, n in zip(y, nomi, strict=True):
            d = bootstrap[n][f"{chiave}_diff"]
            lo, hi = bootstrap[n][f"{chiave}_ic"]
            s = 100 if chiave == "drawdown" else 1
            ax.plot([lo * s, hi * s], [yi, yi], color=COLORI["multi"], lw=1.6, solid_capstyle="butt")
            ax.plot(d * s, yi, "o", color=COLORI["multi"], ms=6, mec="#fcfcfb", mew=1.5)
        ax.axvline(0, color=SECONDARIO, lw=1.0)
        ax.set_title(titolo)
        ax.set_yticks(y, [BREVI[n] for n in nomi])
        ax.grid(axis="y", visible=False)
    assi[0].set_xlabel("positivo = meglio la multi-scala")
    assi[1].set_xlabel("positivo = perdita massima piu' contenuta")
    fig.suptitle("Markov multi-scala meno ogni confronto: stima e intervallo al 95%", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.17, wspace=0.08, left=0.17)
    _firma(fig, "Bootstrap stazionario, 10.000 ricampionamenti, blocco medio 20 giorni. Periodo di test.")
    return fig


def fig_potenza(tab):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(tab["delta"], tab["potenza_ic"], "-o", color=COLORI["multi"], lw=1.8, ms=6, mec="#fcfcfb", mew=1.5)
    ax.axhline(0.8, color=SECONDARIO, lw=0.9, ls=(0, (4, 3)))
    ax.text(tab["delta"].max(), 0.81, "80%", ha="right", va="bottom", fontsize=8.5, color=SECONDARIO)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Vantaggio di Sharpe vero della strategia sul vol targeting")
    ax.set_ylabel("Quota di volte in cui il test lo riconosce")
    ax.set_title("Che vantaggio poteva riconoscere il test?")
    fig.subplots_adjust(bottom=0.2)
    _firma(fig, "Simulazione sui dati di sviluppo, 300 ripetizioni per punto, lunghezza pari al test.")
    return fig


def fig_regimi(matrice, vol, stati, finestra):
    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.9], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax.grid(False)
    nomi = ["calma", "normale", "stress"]
    ax.imshow(matrice, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f1f6fc", "#2a78d6"]),
              vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrice[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if matrice[i, j] > 0.55 else INCHIOSTRO)
    ax.set_xticks(range(3), nomi)
    ax.set_yticks(range(3), nomi)
    ax.set_xlabel("stato domani")
    ax.set_ylabel("stato oggi")
    ax.spines[:].set_visible(False)
    ax.set_title("Matrice di transizione")
    ax2 = fig.add_subplot(gs[1])
    _ombra_stress(ax2, stati, finestra)
    v = vol[finestra] * 100
    ax2.plot(v.index, v, color=COLORI["multi"], lw=1.1)
    ax2.set_ylabel("Volatilita' EWMA annualizzata (%)")
    ax2.set_title("Volatilita' del mercato (grigio = stress)")
    ax2.margins(x=0.01)
    fig.subplots_adjust(bottom=0.16, top=0.88)
    _firma(fig, "Matrice stimata su tutti i dati fino a fine test (descrittiva). Stati classificati solo con il passato.")
    return fig


def fig_sviluppo_test(met_sviluppo, met_test, nomi):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(nomi))[::-1]
    h = 0.36
    ax.barh(y + h / 2, [met_sviluppo.loc[n, "sharpe"] for n in nomi], h - 0.04, color=COLORI["multi"],
            label="Sviluppo (2007-2014)")
    ax.barh(y - h / 2, [met_test.loc[n, "sharpe"] for n in nomi], h - 0.04, color=COLORI["vol_target_pari"],
            label="Test (2015-2026)")
    ax.set_yticks(y, [BREVI[n] for n in nomi])
    ax.set_xlabel("Sharpe annualizzato, al netto dei costi")
    ax.set_title("Sharpe in sviluppo e nel test")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.subplots_adjust(bottom=0.14, left=0.2)
    _firma(fig, "Gli Sharpe dei due periodi non sono confrontabili in valore assoluto: dipendono dal mercato.")
    return fig


def fig_drawdown_massimo(met_test, nomi):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(nomi))[::-1]
    ax.barh(y, [met_test.loc[n, "drawdown"] * 100 for n in nomi], 0.6,
            color=[COLORI["multi"] if n == "multi" else "#b9b8b0" for n in nomi])
    ax.set_yticks(y, [BREVI[n] for n in nomi])
    ax.set_xlabel("Drawdown massimo nel test (%)")
    ax.set_title("Perdita massima dal picco")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(bottom=0.15, left=0.2)
    _firma(fig)
    return fig
