"""Forêts aléatoires (Q19-Q20) : bootstrap + sous-ensemble de variables + vote."""
import json
import numpy as np
from .arbre import construire_arbre, prediction, profondeur


def construire_foret(data_train, sep_max, taille_min, n_arbres, p_var, rng=None, bootstrap=True):
    """Liste d'arbres ; renvoie (arbres, indices hors-sac, importances cumulées)."""
    data_train = np.asarray(data_train, dtype=float)
    rng = np.random.default_rng(rng)
    n, m = len(data_train), data_train.shape[1] - 1
    arbres, oob, imp = [], [], np.zeros(m)
    for _ in range(n_arbres):
        idx = rng.integers(0, n, n) if bootstrap else np.arange(n)
        arbres.append(construire_arbre(data_train[idx], sep_max, taille_min, p_var, rng, imp))
        oob.append(np.setdiff1d(np.arange(n), idx))
    return arbres, oob, imp / max(n_arbres, 1)


def random_forest(data_train, data_test, sep_max=10, taille_min=2, n_arbres=20, p_var=None, rng=None):
    """Q19 : classe prédite (vote majoritaire) pour chaque ligne de data_test."""
    data_train = np.asarray(data_train, dtype=float)
    p_var = p_var or max(1, int(round(np.sqrt(data_train.shape[1] - 1))))
    arbres, _, _ = construire_foret(data_train, sep_max, taille_min, n_arbres, p_var, rng)
    return [1.0 if np.mean([prediction(a, x) for a in arbres]) >= 0.5 else 0.0
            for x in np.asarray(data_test)]


class ForetAleatoire:
    """Classifieur : fit / predict / predict_proba / score, OOB, importances, JSON."""

    def __init__(self, n_arbres=100, sep_max=12, taille_min=2, p_var=None, graine=None, bootstrap=True):
        self.n_arbres, self.sep_max, self.taille_min = n_arbres, sep_max, taille_min
        self.p_var, self.graine, self.bootstrap = p_var, graine, bootstrap

    def fit(self, donnees):
        donnees = np.asarray(donnees, dtype=float)
        m = donnees.shape[1] - 1
        p_var = self.p_var or max(1, int(round(np.sqrt(m))))   # racine du nb de variables
        self.arbres_, oob, self.importances_ = construire_foret(
            donnees, self.sep_max, self.taille_min, self.n_arbres, p_var, self.graine, self.bootstrap)
        s = self.importances_.sum()
        if s > 0:
            self.importances_ = self.importances_ / s
        self.oob_score_ = self._oob(donnees, oob) if self.bootstrap else None
        return self

    def _oob(self, donnees, oob):
        votes, cnt = np.zeros(len(donnees)), np.zeros(len(donnees))
        for a, idx in zip(self.arbres_, oob):
            for i in idx:
                votes[i] += prediction(a, donnees[i, :-1])
                cnt[i] += 1
        ok = cnt > 0
        if not ok.any():
            return None
        pred = (votes[ok] / cnt[ok] >= 0.5).astype(float)
        return float(np.mean(pred == donnees[ok, -1]))

    def predict_proba(self, X):
        """Probabilité de la classe métal = fraction d'arbres votant 1."""
        X = np.asarray(X, dtype=float)
        return np.array([np.mean([prediction(a, x) for a in self.arbres_]) for x in X])

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(float)

    def score(self, donnees):
        donnees = np.asarray(donnees, dtype=float)
        return float(np.mean(self.predict(donnees[:, :-1]) == donnees[:, -1]))

    def profondeur_moyenne(self):
        return float(np.mean([profondeur(a) for a in self.arbres_]))

    def sauver(self, chemin):
        def conv(a):
            return [int(a[0]), float(a[1]), conv(a[2]), conv(a[3])] if isinstance(a, list) else int(a)
        with open(chemin, "w") as f:
            json.dump({"arbres": [conv(a) for a in self.arbres_],
                       "importances": list(map(float, self.importances_))}, f)

    @classmethod
    def charger(cls, chemin):
        with open(chemin) as f:
            d = json.load(f)
        o = cls(n_arbres=len(d["arbres"]))
        o.arbres_, o.importances_, o.oob_score_ = d["arbres"], np.array(d["importances"]), None
        return o
