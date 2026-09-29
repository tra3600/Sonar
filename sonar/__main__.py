"""Interface en ligne de commande : python -m sonar <commande>.

  demo     jeu simulé -> forêt -> métriques + figure
  bench    arbre seul vs forêt (validation croisée)
  entrainer  entraîne sur un CSV (ou jeu simulé) et sauve le modèle JSON
  predire    classe une mesure (ligne CSV de 60 valeurs) avec un modèle sauvé
  generer    écrit un jeu de données simulé au format CSV
"""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
from . import (lire_donnees, generer_jeu, decouper, ForetAleatoire, ArbreCART, exactitude,
               matrice_confusion, precision_rappel_f1, auc_roc, validation_croisee)
from .donnees import sauver_donnees
from .signaux import temps_vecteur, chirp, simuler_echo


def _jeu(a):
    if getattr(a, "csv", None):
        return lire_donnees(a.csv)
    return generer_jeu(nx=a.nx, snr_db=a.snr, graine=a.graine)


def cmd_demo(a):
    X = _jeu(a)
    train, test = decouper(X, int(0.7 * len(X)), a.graine)
    t0 = time.time()
    foret = ForetAleatoire(a.arbres, a.sep_max, a.taille_min, graine=a.graine).fit(train)
    dt = time.time() - t0
    proba = foret.predict_proba(test[:, :-1])
    y, yp = test[:, -1], (proba >= 0.5).astype(float)
    p, r, f1 = precision_rappel_f1(y, yp)
    print(f"{len(train)} apprentissage / {len(test)} test, {a.arbres} arbres ({dt:.2f} s)")
    print(f"exactitude={exactitude(y, yp):.3f}  OOB={foret.oob_score_:.3f}  AUC={auc_roc(y, proba):.3f}")
    print(f"précision={p:.3f} rappel(métal)={r:.3f} F1={f1:.3f}")
    print("matrice de confusion [[VN FP],[FN VP]] =", matrice_confusion(y, yp).tolist())
    print("colonnes les plus importantes :", np.argsort(foret.importances_)[::-1][:5].tolist())
    if a.figure:
        from .graphiques import rapport
        temps = temps_vecteur(1.0, 4000)
        e = chirp(temps, 200, 700, 0.25, 1.0)
        rng = np.random.default_rng(a.graine)
        rapport(a.figure, temps, e, simuler_echo("R", temps, e, 10, a.snr, rng),
                simuler_echo("M", temps, e, 10, a.snr, rng), 1.0, roc=(y, proba))
        print("figure :", a.figure)


def cmd_bench(a):
    X = _jeu(a)
    for nom, fab in (("arbre seul", lambda: ArbreCART(a.sep_max, a.taille_min, graine=a.graine)),
                     (f"forêt {a.arbres} arbres", lambda: ForetAleatoire(a.arbres, a.sep_max, a.taille_min, graine=a.graine))):
        t0 = time.time()
        acc, auc = validation_croisee(X, fab, k=a.k, graine=a.graine)
        print(f"{nom:18s} exactitude {acc.mean():.3f} ± {acc.std():.3f}   AUC {np.nanmean(auc):.3f}   {time.time()-t0:.1f} s")


def cmd_entrainer(a):
    X = _jeu(a)
    foret = ForetAleatoire(a.arbres, a.sep_max, a.taille_min, graine=a.graine).fit(X)
    foret.sauver(a.sortie)
    print(f"modèle sauvé dans {a.sortie} (OOB={foret.oob_score_:.3f})")


def cmd_predire(a):
    foret = ForetAleatoire.charger(a.modele)
    x = np.array([float(v) for v in a.mesure.split(",")])
    p = foret.predict_proba(x[None, :])[0]
    print(f"{'MÉTAL (danger potentiel)' if p >= 0.5 else 'ROCHE'}  (P(métal) = {p:.2f})")


def cmd_generer(a):
    sauver_donnees(generer_jeu(nx=a.nx, snr_db=a.snr, graine=a.graine), a.sortie)
    print(f"{a.nx} expériences écrites dans {a.sortie}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sonar", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    def commun(q):
        q.add_argument("--csv", help="fichier de données réel (60 réels + R/M)")
        q.add_argument("--nx", type=int, default=208)
        q.add_argument("--snr", type=float, default=10.0, help="rapport signal/bruit simulé (dB)")
        q.add_argument("--graine", type=int, default=0)
        q.add_argument("--arbres", type=int, default=100)
        q.add_argument("--sep-max", dest="sep_max", type=int, default=12)
        q.add_argument("--taille-min", dest="taille_min", type=int, default=2)
    for nom, fn in (("demo", cmd_demo), ("bench", cmd_bench), ("entrainer", cmd_entrainer)):
        q = sp.add_parser(nom); commun(q); q.set_defaults(f=fn)
        if nom == "demo": q.add_argument("--figure", help="PNG de synthèse")
        if nom == "bench": q.add_argument("-k", type=int, default=5)
        if nom == "entrainer": q.add_argument("-o", "--sortie", default="modele.json")
    q = sp.add_parser("predire"); q.add_argument("modele"); q.add_argument("mesure", help="60 valeurs séparées par des virgules"); q.set_defaults(f=cmd_predire)
    q = sp.add_parser("generer"); commun(q); q.add_argument("-o", "--sortie", default="sonar.csv"); q.set_defaults(f=cmd_generer)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
