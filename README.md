# StageM2_2026

> **Projet de stage de 2 Année Master Mathématiques Appliquées ** : Calcul Scientifique et Modélisation (Année 2026).

Ce dépôt contient et assure le suivi du code développé durant mon stage de fin d'études.

---

## Objectif du Projet

Le but principal de ce projet est de construire un **modèle réduit** (Reduced Order Model - ROM) pour simuler et étudier des instabilités de type **KHRTI** (Kelvin-Helmholtz / Rayleigh-Taylor Instabilities). 

Pour y parvenir, nous utilisons une approche basée sur l'apprentissage profond, et plus particulièrement les **$\beta$-VAE** ($\beta$-Variational Autoencoders). L'objectif de l'architecture $\beta$-VAE est d'apprendre un espace latent structuré et d'isoler des facteurs de variation indépendants (disentanglement) afin de capturer la dynamique physique des instabilités.

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

