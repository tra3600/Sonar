"""Extraction de l'enveloppe du spectrogramme (Q5-Q7).

Les m fenêtres w(t, f, eta) sont placées le long de la diagonale du plan
temps-fréquence (le chirp balaye linéairement les fréquences) : chaque fenêtre
a pour taille DT = Tmax/m et Df = fmax/m.
"""
import numpy as np
from .signaux import spectrogramme


def w(t, f, eta, DT, T, Df):
    """Q5 : 1 si (t, f) est dans la fenêtre d'indice eta, 0 sinon.

    Centre : t_eta = DT/2 + eta*DT, f_eta = Df/2 + eta*Df (T est conservé pour
    fidélité à la signature de l'énoncé ; la fenêtre ne dépend que de DT et Df).
    """
    t_eta = DT / 2 + eta * DT
    f_eta = Df / 2 + eta * Df
    return 1 if (abs(t - t_eta) <= DT / 2 and abs(f - f_eta) <= Df / 2) else 0


def enveloppe(eta, S, p, dt, df, DT=None, Df=None, T=None):
    """Q6 : P_eta = sum_ij p_ij S_ij w(t_i, f_j, eta) df dt (version boucles).

    S est indexé [i (temps), j (fréquence)] comme dans l'énoncé.
    """
    ni, nj = S.shape
    DT = DT if DT is not None else ni * dt
    Df = Df if Df is not None else nj * df
    DT_cell, Df_cell = DT, Df
    total = 0.0
    for i in range(ni):
        for j in range(nj):
            total += p[i, j] * S[i, j] * w(i * dt, j * df, eta, DT_cell, T, Df_cell) * df * dt
    return total


def enveloppe_rapide(S, m=60, p=None, dt=1.0, df=1.0):
    """Version vectorisée : m valeurs P_0..P_{m-1} (fenêtres diagonales).

    S : tableau [temps, fréquence]. Chaque fenêtre couvre ~ni/m instants et
    ~nj/m fréquences ; on utilise des bornes entières pour couvrir tout le plan.
    """
    S = np.asarray(S, dtype=float)
    ni, nj = S.shape
    p = np.ones_like(S) if p is None else p
    bi = np.linspace(0, ni, m + 1).astype(int)
    bj = np.linspace(0, nj, m + 1).astype(int)
    P = np.empty(m)
    for e in range(m):
        i0, i1 = bi[e], max(bi[e + 1], bi[e] + 1)
        j0, j1 = bj[e], max(bj[e + 1], bj[e] + 1)
        P[e] = np.sum(p[i0:i1, j0:j1] * S[i0:i1, j0:j1]) * df * dt
    return P


def normalisation(P):
    """Q7 : transformation affine ramenant min à 0 et max à 1."""
    P = np.asarray(P, dtype=float)
    mn, mx = P.min(), P.max()
    if mx == mn:
        return np.zeros_like(P)
    return (P - mn) / (mx - mn)


def extraire_caracteristiques(s, Tf, m=60, n=128):
    """Signal de retour -> vecteur de m valeurs normalisées dans [0, 1]."""
    _, t, S = spectrogramme(s, Tf, n=n)
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    return normalisation(enveloppe_rapide(S.T, m=m, dt=dt))
