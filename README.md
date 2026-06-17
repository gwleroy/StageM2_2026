# StageM2_2026

> **Projet de stage de 2 Année Master Mathématiques Appliquées ** : Calcul Scientifique et Modélisation (Année 2026).

Ce dépôt contient et assure le suivi du code développé durant mon stage de fin d'études.

---

## Objectif du Projet

Le but principal de ce projet est de construire un **modèle réduit** (Reduced Order Model - ROM) pour simuler et étudier des instabilités de type **KHRTI** (Kelvin-Helmholtz / Rayleigh-Taylor Instabilities). 

Pour y parvenir, la démarche se structure en deux étapes :
1. **Compression Spatiale ($\beta$-VAE)** : Nous utilisons des $\beta$-VAE ($\beta$-Variational Autoencoders) pour apprendre un espace latent. L'objectif est de capturer efficacement les caractéristiques physiques et géométriques des instabilités à un instant donné.
2. **Prédiction Temporelle (Perspectives)** : Dans un second temps, l'évolution temporelle de ces instabilités au sein de l'espace latent sera modélisée. L'intégration de modèles séquentiels avancés, tels que les **LSTM** (Long Short-Term Memory) ou les **Transformers**, est envisagée afin de prédire la trajectoire temporelle de la dynamique dans cet espace réduit.

---

## Technologies & Bibliothèques

* **Langage principal** : Python 3
* **Apprentissage Profond** : PyTorch
* **Calcul Scientifique / Physique** : NumPy, SciPy
* **Visualisation** : Matplotlib

---

## Structure du Repository

Voici l'organisation principale des dossiers et fichiers du projet :

```text
├── data/               # Données brutes
├── src/                # Code source principal
│   ├── models/         # Architecture du β-VAE (et futurs LSTM/Transformers)
│   ├── dataset.py      # Chargement et prétraitement des données (PyTorch Dataset)
│   ├── train.py        # Scripts d'entraînement des modèles
├── outputs/            # Poids des modèles sauvegardés (*.pt) et graphiques générés
├── README.md           # Documentation du projet
└── requirements.txt    # Liste des dépendances Python
