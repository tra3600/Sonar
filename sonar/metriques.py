"""Métriques d'évaluation et validation croisée."""
import numpy as np


def exactitude(y, yp):
    return float(np.mean(np.asarray(y) == np.asarray(yp)))


def matrice_confusion(y, yp):
    """[[VN, FP], [FN, VP]] avec métal = classe positive."""
    y, yp = np.asarray(y), np.asarray(yp)
    return np.array([[np.sum((y == 0) & (yp == 0)), np.sum((y == 0) & (yp == 1))],
                     [np.sum((y == 1) & (yp == 0)), np.sum((y == 1) & (yp == 1))]])


def precision_rappel_f1(y, yp):
    (vn, fp), (fn, vp) = matrice_confusion(y, yp)
    p = vp / (vp + fp) if vp + fp else 0.0
    r = vp / (vp + fn) if vp + fn else 0.0
    return float(p), float(r), float(2 * p * r / (p + r)) if p + r else 0.0


def auc_roc(y, score):
    """Aire sous la courbe ROC (statistique de Mann-Whitney, ex aequo à 1/2)."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def courbe_roc(y, score):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    seuils = np.unique(np.r_[score, 0, 1])[::-1]
    tpr = [np.mean(score[y == 1] >= s) for s in seuils]
    fpr = [np.mean(score[y == 0] >= s) for s in seuils]
    return np.array(fpr), np.array(tpr)


def validation_croisee(donnees, fabrique, k=5, graine=0):
    """k-fold stratifié : renvoie (exactitudes par pli, AUC par pli)."""
    donnees = np.asarray(donnees)
    rng = np.random.default_rng(graine)
    plis = [[] for _ in range(k)]
    for c in (0.0, 1.0):
        idx = rng.permutation(np.where(donnees[:, -1] == c)[0])
        for j, i in enumerate(idx):
            plis[j % k].append(i)
    accs, aucs = [], []
    for j in range(k):
        test = np.array(sorted(plis[j]))
        train = np.setdiff1d(np.arange(len(donnees)), test)
        mod = fabrique().fit(donnees[train])
        proba = mod.predict_proba(donnees[test, :-1]) if hasattr(mod, "predict_proba") else mod.predict(donnees[test, :-1])
        accs.append(exactitude(donnees[test, -1], (proba >= 0.5).astype(float)))
        aucs.append(auc_roc(donnees[test, -1], proba))
    return np.array(accs), np.array(aucs)
