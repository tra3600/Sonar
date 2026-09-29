import numpy as np
import pytest
from sonar import *
from sonar.arbre import gini_groupe, Gini_groupes, feuille, indices_aleatoires, test_separation as separation
from sonar.signaux import spectrogramme as spectro


def test_temps_et_chirp():
    t = temps_vecteur(1.2, 1000)
    assert len(t) == 1000 and t[-1] < 1.2 and t[0] == 0
    e = chirp(t, 200, 100, 0.5, 1)
    assert np.all(e[t > 0.5] == 0) and np.max(np.abs(e)) <= 1


def test_stft_tailles():
    f, tt, S = spectro(np.random.randn(1000), 1.2, n=128)
    assert len(f) == 65 and S.shape == (65, len(tt))


def test_normalisation():
    P = normalisation([3, 5, 9])
    assert P.min() == 0 and P.max() == 1
    assert np.all(normalisation([2, 2]) == 0)


def test_w_et_enveloppe_coherentes():
    assert w(0.5, 0.5, 0, 1, 10, 1) == 1 and w(5, 5, 0, 1, 10, 1) == 0
    S = np.random.rand(6, 6)
    P = enveloppe_rapide(S, m=3)
    assert P.shape == (3,) and np.isclose(P.sum(), S.sum() - 0, atol=S.sum())


def test_gini_et_feuille():
    pur = np.array([[0.1, 1.0], [0.2, 1.0]])
    mix = np.array([[0.1, 1.0], [0.2, 0.0]])
    assert gini_groupe(pur) == 0 and np.isclose(gini_groupe(mix), 0.5)
    assert np.isclose(Gini_groupes([pur, mix]), 0.25)
    assert feuille(pur) == 1 and feuille(np.array([[0, 0.0]])) == 0


def test_indices_aleatoires_distincts():
    i = indices_aleatoires(60, 7, 1)
    assert len(i) == 7 == len(set(i)) and all(0 <= k < 60 for k in i)


def test_separation_stricte():
    d = np.array([[0.2, 0.0], [0.5, 1.0], [0.9, 1.0]])
    g, dr = separation(0, 0.5, d)
    assert len(g) == 1 and len(dr) == 2


def test_arbre_exemple_enonce():
    # arbre 5(b) : [2, 0.39, [6, 0.19, 1, 0], 1]
    arbre = [2, 0.39, [6, 0.19, 1, 0], 1]
    a = [0, 0, 0.1, 0, 0, 0, 0.5, 0, 0, 0]
    assert prediction(arbre, a) == 0
    assert prediction(arbre, [0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0]) == 1


def test_arbre_apprend_regle_simple():
    rng = np.random.default_rng(0)
    X = rng.random((200, 5)); y = (X[:, 3] > 0.5).astype(float)
    D = np.c_[X, y]
    arbre = ArbreCART(sep_max=3, p_var=5, graine=0).fit(D)
    assert exactitude(y, arbre.predict(X)) > 0.98
    assert np.argmax(arbre.importances_) == 3


@pytest.fixture(scope="module")
def jeu():
    return generer_jeu(nx=120, graine=1)


def test_jeu_forme(jeu):
    assert jeu.shape == (120, 61) and jeu[:, :-1].min() >= 0 and jeu[:, :-1].max() <= 1
    assert set(np.unique(jeu[:, -1])) == {0.0, 1.0}


def test_foret_bat_le_hasard(jeu):
    train, test = decouper(jeu, 80, 0)
    f = ForetAleatoire(30, graine=0).fit(train)
    assert f.score(test) > 0.8 and f.oob_score_ > 0.75
    assert auc_roc(test[:, -1], f.predict_proba(test[:, :-1])) > 0.85
    assert np.isclose(f.importances_.sum(), 1)


def test_random_forest_enonce(jeu):
    train, test = decouper(jeu, 80, 0)
    pred = random_forest(train, test, 8, 2, 15, 7, rng=0)
    assert len(pred) == len(test) and exactitude(test[:, -1], pred) > 0.75


def test_json_aller_retour(jeu, tmp_path):
    f = ForetAleatoire(10, graine=0).fit(jeu)
    f.sauver(tmp_path / "m.json")
    g = ForetAleatoire.charger(tmp_path / "m.json")
    assert np.array_equal(f.predict(jeu[:, :-1]), g.predict(jeu[:, :-1]))


def test_csv_aller_retour(jeu, tmp_path):
    from sonar.donnees import sauver_donnees
    sauver_donnees(jeu, tmp_path / "d.csv")
    r = lire_donnees(tmp_path / "d.csv")
    assert r.shape == jeu.shape and np.array_equal(r[:, -1], jeu[:, -1])


def test_metriques():
    y = np.array([0, 0, 1, 1.0])
    assert auc_roc(y, [0.1, 0.2, 0.8, 0.9]) == 1 and auc_roc(y, [0.5] * 4) == 0.5
    assert matrice_confusion(y, [0, 1, 1, 1]).tolist() == [[1, 1], [0, 2]]
    acc, auc = validation_croisee(generer_jeu(nx=60, graine=2), lambda: ForetAleatoire(10, graine=0), k=3)
    assert len(acc) == 3 and acc.mean() > 0.7
