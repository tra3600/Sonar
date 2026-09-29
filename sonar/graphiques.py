"""Tracés (matplotlib) : chirp, spectrogramme, enveloppe, ROC, importances."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from .metriques import courbe_roc, auc_roc


def tracer_signal(temps, s, titre="Signal", ax=None):
    ax = ax or plt.gca()
    ax.plot(temps, s, lw=0.7)
    ax.set(xlabel="Temps (s)", ylabel="Amplitude", title=titre)


def tracer_spectrogramme(f, t, S, titre="Spectrogramme", ax=None):
    ax = ax or plt.gca()
    ax.pcolormesh(t, f, S, shading="auto")
    ax.set(xlabel="Temps (s)", ylabel="Fréquence (Hz)", title=titre)


def tracer_roc(y, score, ax=None):
    ax = ax or plt.gca()
    fpr, tpr = courbe_roc(y, score)
    ax.plot(fpr, tpr, label=f"AUC = {auc_roc(y, score):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set(xlabel="Faux positifs", ylabel="Vrais positifs", title="Courbe ROC")
    ax.legend()


def tracer_importances(imp, top=15, ax=None):
    ax = ax or plt.gca()
    idx = np.argsort(imp)[::-1][:top]
    ax.bar(range(len(idx)), imp[idx])
    ax.set_xticks(range(len(idx)), [str(i) for i in idx], fontsize=7)
    ax.set(xlabel="Colonne", ylabel="Importance", title="Importance des variables")


def rapport(fichier, temps, e, s_roche, s_metal, Tf, roc=None, imp=None):
    """Figure de synthèse enregistrée dans `fichier`."""
    from .signaux import spectrogramme
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    tracer_signal(temps, e, "Chirp émis", axs[0, 0])
    tracer_signal(temps, s_metal, "Écho métal", axs[0, 1])
    tracer_signal(temps, s_roche, "Écho roche", axs[0, 2])
    tracer_spectrogramme(*spectrogramme(s_metal, Tf), "Spectrogramme métal", axs[1, 0])
    tracer_spectrogramme(*spectrogramme(s_roche, Tf), "Spectrogramme roche", axs[1, 1])
    if roc is not None:
        tracer_roc(*roc, ax=axs[1, 2])
    elif imp is not None:
        tracer_importances(imp, ax=axs[1, 2])
    fig.tight_layout()
    fig.savefig(fichier, dpi=110)
    plt.close(fig)
