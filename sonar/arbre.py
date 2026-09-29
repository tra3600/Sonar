"""Arbres de décision CART (indice de Gini) — Q10 à Q18.

Représentation identique à l'énoncé : un nœud est [ind, val, gauche, droite],
où gauche / droite est soit un nœud, soit une feuille 0 (roche) / 1 (métal).
On descend à gauche si donnee[ind] < val, sinon à droite.
"""
import numpy as np


def indices_aleatoires(m, p_var, rng=None):
    """Q12 : p_var numéros de colonnes distincts tirés dans [0, m[."""
    rng = np.random.default_rng(rng)
    return [int(i) for i in rng.choice(m, size=min(p_var, m), replace=False)]


def test_separation(ind, val, donnees):
    """Q13 : (gauche, droite) selon donnees[:, ind] < val."""
    donnees = np.asarray(donnees)
    masque = donnees[:, ind] < val
    return [donnees[masque], donnees[~masque]]


def gini_groupe(groupe):
    """Gini d'un groupe : 1 - sum_i p_i^2 (0 pour un groupe pur ou vide)."""
    n = len(groupe)
    if n == 0:
        return 0.0
    p1 = np.mean(np.asarray(groupe)[:, -1] == 1)
    return 1.0 - p1 ** 2 - (1 - p1) ** 2


def Gini_groupes(groupes):
    """Q14 : Gini total pondéré par la taille relative des deux groupes."""
    total = sum(len(g) for g in groupes)
    if total == 0:
        return 0.0
    return sum(len(g) / total * gini_groupe(g) for g in groupes)


def feuille(data):
    """Q15 : classe majoritaire (0 roche / 1 métal ; égalité -> 1)."""
    data = np.asarray(data)
    return 1 if np.sum(data[:, -1] == 1) >= len(data) / 2 else 0


def meilleure_separation(donnees, colonnes):
    """Meilleur (ind, val, gini) parmi `colonnes`, en O(n log n) par colonne."""
    y = donnees[:, -1]
    n = len(y)
    meilleur = (None, None, np.inf)
    for ind in colonnes:
        ordre = np.argsort(donnees[:, ind], kind="stable")
        xs, ys = donnees[ordre, ind], y[ordre]
        c1 = np.cumsum(ys)                    # nb de 1 dans les i premiers
        i = np.arange(1, n)                   # taille du groupe gauche
        valides = xs[1:] != xs[:-1]
        if not valides.any():
            continue
        i, l1 = i[valides], c1[:-1][valides]
        r1 = c1[-1] - l1
        nr = n - i
        pl, pr = l1 / i, r1 / nr
        g = (i * (1 - pl ** 2 - (1 - pl) ** 2) + nr * (1 - pr ** 2 - (1 - pr) ** 2)) / n
        k = int(np.argmin(g))
        if g[k] < meilleur[2]:
            meilleur = (ind, xs[1:][valides][k], float(g[k]))
    return meilleur


def separe(donnees, p_var, rng=None):
    """Sépare les données sur la meilleure des p_var colonnes tirées au hasard."""
    donnees = np.asarray(donnees)
    m = donnees.shape[1] - 1
    ind, val, gini = meilleure_separation(donnees, indices_aleatoires(m, p_var, rng))
    if ind is None:
        return [None, None, donnees[:0], donnees], None
    g, d = test_separation(ind, val, donnees)
    return [ind, val, g, d], gini


def construit(arbre, sep_max, taille_min, p_var, ind_rec, rng=None, importances=None, n_total=None):
    """Q16 : construction récursive ; remplace gauche/droite par nœuds ou feuilles."""
    g, d = arbre[2], arbre[3]
    if len(g) == 0 or len(d) == 0:                       # pas de séparation utile
        arbre[2] = arbre[3] = feuille(d if len(g) == 0 else g)
        return
    if importances is not None and n_total:               # gain d'impureté pondéré
        tous = np.vstack([g, d])
        gain = (gini_groupe(tous) - Gini_groupes([g, d])) * len(tous) / n_total
        importances[arbre[0]] += gain
    if ind_rec >= sep_max:                                # profondeur maximale
        arbre[2], arbre[3] = feuille(g), feuille(d)
        return
    for pos, groupe in ((2, g), (3, d)):
        if len(groupe) < taille_min or gini_groupe(groupe) == 0:
            arbre[pos] = feuille(groupe)
        else:
            arbre[pos], _ = separe(groupe, p_var, rng)
            construit(arbre[pos], sep_max, taille_min, p_var, ind_rec + 1, rng, importances, n_total)


def construire_arbre(data_train, sep_max=10, taille_min=2, p_var=None, rng=None, importances=None):
    data_train = np.asarray(data_train, dtype=float)
    p_var = p_var or data_train.shape[1] - 1
    rng = np.random.default_rng(rng)
    arbre, _ = separe(data_train, p_var, rng)
    construit(arbre, sep_max, taille_min, p_var, 1, rng, importances, len(data_train))
    return arbre


def prediction(arbre, donnee):
    """Q18 : parcours récursif de l'arbre jusqu'à une feuille."""
    branche = arbre[2] if donnee[arbre[0]] < arbre[1] else arbre[3]
    if isinstance(branche, list):
        return prediction(branche, donnee)
    return branche


def profondeur(arbre):
    if not isinstance(arbre, list):
        return 0
    return 1 + max(profondeur(arbre[2]), profondeur(arbre[3]))


class ArbreCART:
    """Interface fit / predict autour des fonctions de l'énoncé."""

    def __init__(self, sep_max=10, taille_min=2, p_var=None, graine=None):
        self.sep_max, self.taille_min, self.p_var, self.graine = sep_max, taille_min, p_var, graine

    def fit(self, donnees):
        donnees = np.asarray(donnees, dtype=float)
        self.importances_ = np.zeros(donnees.shape[1] - 1)
        self.arbre_ = construire_arbre(donnees, self.sep_max, self.taille_min, self.p_var,
                                       self.graine, self.importances_)
        return self

    def predict(self, X):
        return np.array([prediction(self.arbre_, x) for x in np.asarray(X)], dtype=float)
