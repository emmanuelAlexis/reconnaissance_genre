# Classificateur de Genre Homme/Femme avec MobileNetV2 et PyQt5

Ce projet propose une application graphique PyQt5 permettant de prédire le genre (Homme/Femme) à partir d'une image, basée sur un modèle de deep learning entraîné avec MobileNetV2.
Il inclut également un script d'entraînement du modèle ainsi qu'un script de prédiction en ligne de commande.

---

## Contenu du projet

- `main.py` : Application PyQt5 GUI pour sélectionner une image ou prendre une photo via webcam et afficher la prédiction de genre avec synthèse vocale.
- `train_model.py` : Script d'entraînement du modèle de classification de genre avec MobileNetV2 sur le dataset UTKFace.
- `predict_gender.py` : Script en ligne de commande pour prédire le genre d'une image donnée avec le modèle pré-entraîné.
- `require.txt` : Fichier listant les dépendances Python nécessaires.

---

## Prérequis

- Python 3.8 ou supérieur
- GPU recommandé pour l'entraînement (optionnel)
- Dataset UTKFace placé dans le dossier `data/UTKFace` (images nommées au format `age_gender_race_date.jpg` optionnel pour le test mais obligatoire pour l'entrainement du model)
- Téléchargement officiel du dataset UTKFace (1,3go zip file) :  
  [https://susanqq.github.io/UTKFace/](https://susanqq.github.io/UTKFace/)

---

## Installation des dépendances

```bash
pip install -r require.txt
```

---

## Entraînement du modèle

Lancez le script `train_model.py` pour entraîner le modèle sur le dataset UTKFace :

```bash
python train_model.py
```

- Le modèle sera sauvegardé sous le nom `gender_model.h5`.
- Le modèle utilise MobileNetV2 avec des poids pré-entraînés sur ImageNet, un fine-tuning est appliqué.
- Les images sont redimensionnées à 96x96 pixels et normalisées.

---

## Utilisation de l'application GUI (main.py)

Pour lancer l'application graphique :

```bash
python main.py
```

Fonctionnalités :

- Sélectionner une image via un dialogue fichier.
- Capturer une photo directement depuis la webcam.
- Affichage de l'image sélectionnée.
- Affichage des pourcentages Homme/Femme prédit.
- Synthèse vocale en français annonçant la prédiction.

---

## Prédiction en ligne de commande (predict_gender.py)

Pour prédire le genre d'une image en ligne de commande :

Exemple :

```bash
python predict_gender.py  //puis tu entre le nom du fichier existant dans dataToTest ex: emmanuel ou homme ou femme_2 ...
```

Affiche les pourcentages Homme/Femme dans la console.

---

## Organisation des fichiers

```
project/
│
└── data/
    └── UTKFace/          # Dataset d'images (non inclus)
    └── dataToTest/          # Donnee d'entree pour tester avec predict_gender.py (non inclus)
├── main.py               # Application PyQt5 GUI
├── train_model.py        # Entraînement du modèle MobileNetV2
├── predict_gender.py     # Prédiction via ligne de commande
├── require.txt           # Dépendances Python
├── gender_model.h5       # Modèle sauvegardé après entraînement

```

---

## Remarques importantes

- Assurez-vous que le fichier `gender_model.h5` existe avant d'exécuter `main.py` ou `predict_gender.py`.
- Le script d'entraînement peut prendre plusieurs minutes selon la configuration matérielle.
- Le dataset UTKFace est disponible publiquement et doit être téléchargé séparément (non inclus dans ce projet).

**Amusez-vous bien avec votre classificateur de genre ! BY ADOLPHE Alexis Emmanuel👨👩**
