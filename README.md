# Sonar

Détection d'obstacle par un sonar de sous-marin : l'objet en face est-il une **roche** ou un **objet métallique** (danger potentiel) ?
Implémentation complète du sujet `informatique_PSI5IN.pdf` (PSI 2020), prolongée en un outil utilisable.

## Chaîne de traitement

| Étape | Module | Sujet |
|---|---|---|
| Vecteur temps, CHIRP, simulation d'écho bruité (retard 2d/c, absorption, résonances) | `sonar/signaux.py` | Q1-Q2 |
| Spectrogramme (STFT, Hamming, 50 % de recouvrement) | `sonar/signaux.py` | Q3-Q4 |
| Enveloppe du spectrogramme (fenêtres `w`, intégration pondérée, normalisation) | `sonar/caracteristiques.py` | Q5-Q7 |
| Lecture / écriture CSV, jeu simulé, découpage | `sonar/donnees.py` | Q8-Q9 |
| Arbres CART (Gini, `separe`, `construit`, `prediction`), représentation `[ind, val, gauche, droite]` | `sonar/arbre.py` | Q10-Q18 |
| Forêt aléatoire : bootstrap, √m variables, vote, erreur OOB, importances, sauvegarde JSON | `sonar/foret.py` | Q19-Q20 |
| Exactitude, confusion, précision/rappel/F1, AUC ROC, validation croisée stratifiée | `sonar/metriques.py` | – |
| Figures | `sonar/graphiques.py` | – |

Améliorations par rapport aux fonctions de l'énoncé : recherche du meilleur seuil en O(n log n) par colonne
(au lieu de O(n²)), gains de Gini cumulés pour l'importance des variables, score hors-sac, `predict_proba`.

## Utilisation

```bash
pip install -r requirements.txt
python -m sonar demo --figure rapport.png     # jeu simulé -> forêt -> métriques + figure
python -m sonar bench                         # arbre seul vs forêt (validation croisée 5 plis)
python -m sonar generer -o sonar.csv          # écrit un jeu simulé (60 valeurs + R/M)
python -m sonar entrainer --csv sonar.csv -o modele.json
python -m sonar predire modele.json "0.02,0.04,...(60 valeurs)"
pytest                                        # tests
```

Toutes les commandes acceptent `--csv fichier` pour utiliser le vrai jeu de données Sonar (208 × 60, classes `R`/`M`)
au lieu du jeu simulé, ainsi que `--arbres`, `--sep-max`, `--taille-min`, `--snr`, `--graine`.

En Python :

```python
from sonar import generer_jeu, decouper, ForetAleatoire
train, test = decouper(generer_jeu(), 145, 0)
foret = ForetAleatoire(100).fit(train)
print(foret.score(test), foret.oob_score_)
```

## Résultats (jeu simulé, SNR 10 dB, 208 expériences)

Arbre seul ≈ 75-80 % d'exactitude ; forêt de 100 arbres ≈ 92 % (AUC ≈ 0,98) : la forêt réduit le
surapprentissage et la variance de l'arbre unique, comme l'établit la conclusion du sujet.

## Notes

- Le jeu simulé est un substitut : les performances sur données réelles peuvent différer.
- Dans `chirp`, la formule `sin(2π fe(t) t)` donne une fréquence instantanée `f0 + 2Δf·t/T` : elle doit rester sous la fréquence de Nyquist.
- Les fenêtres de l'enveloppe suivent la diagonale du plan temps-fréquence, car le chirp balaie linéairement les fréquences.
