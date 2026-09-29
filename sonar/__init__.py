"""Sonar : détection d'obstacles (roche / métal) par un sonar de sous-marin.

Chaîne complète : chirp -> écho -> spectrogramme (STFT) -> enveloppe -> arbres CART
-> forêt aléatoire.
"""
from .signaux import temps_vecteur, chirp, simuler_echo, spectrogramme
from .caracteristiques import w, enveloppe, enveloppe_rapide, normalisation, extraire_caracteristiques
from .donnees import lire_donnees, generer_jeu, decouper, ROCHE, METAL
from .arbre import ArbreCART, construire_arbre, prediction
from .foret import ForetAleatoire, random_forest, construire_foret
from .metriques import (exactitude, matrice_confusion, precision_rappel_f1, auc_roc,
                        validation_croisee)

__version__ = "2.0.0"
