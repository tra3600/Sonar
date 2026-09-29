"""Lecture, génération et découpage des données (Q8-Q9)."""
import numpy as np
from .signaux import temps_vecteur, chirp, simuler_echo
from .caracteristiques import extraire_caracteristiques

ROCHE, METAL = 0.0, 1.0


def lire_donnees(nom_fichier):
    """Q8 : lit un CSV (60 réels + 'R'/'M') -> tableau (nx, 61), classe 0.0/1.0."""
    donnees = []
    with open(nom_fichier, "r") as fichier:
        for ligne in fichier:
            ligne = ligne.strip()
            if not ligne:
                continue
            a = ligne.split(",")
            b = [float(x) for x in a[:-1]]
            b.append(ROCHE if a[-1].strip() == "R" else METAL)
            donnees.append(b)
    return np.array(donnees)


def sauver_donnees(donnees, nom_fichier):
    """Écrit un tableau (nx, m+1) au format du fichier d'origine."""
    with open(nom_fichier, "w") as f:
        for ligne in donnees:
            f.write(",".join(f"{x:.4f}" for x in ligne[:-1]) + "," + ("M" if ligne[-1] else "R") + "\n")


def generer_jeu(nx=208, m=60, Tf=1.0, N=4000, snr_db=10.0, graine=0):
    """Campagne d'essais simulée : nx expériences équilibrées roche/métal.

    Un CHIRP est émis, réfléchi (distance ~10 m, incidence variable), bruité,
    puis résumé par une enveloppe normalisée de m valeurs.
    """
    rng = np.random.default_rng(graine)
    temps = temps_vecteur(Tf, N)
    e = chirp(temps, f0=200, Deltafe=700, T=0.25, E0=1.0)
    classes = np.array([ROCHE, METAL] * (nx // 2) + [ROCHE] * (nx % 2))
    rng.shuffle(classes)
    X = np.empty((nx, m + 1))
    for k, c in enumerate(classes):
        s = simuler_echo(int(c), temps, e, distance=rng.uniform(8, 12), snr_db=snr_db, rng=rng)
        X[k, :m] = extraire_caracteristiques(s, Tf, m=m)
        X[k, m] = c
    return X


def decouper(donnees, n_train, rng=None):
    """Mélange puis sépare en (train, test)."""
    rng = np.random.default_rng(rng)
    idx = rng.permutation(len(donnees))
    donnees = np.asarray(donnees)[idx]
    return donnees[:n_train], donnees[n_train:]


def taille_memoire(nx=208, m=60):
    """Q9 : (m+1) * nx flottants de 8 octets (=99 840 octets pour 208 x 61)."""
    return nx * (m + 1) * 8
