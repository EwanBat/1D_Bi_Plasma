# Simulation 1D bi-fluide + Maxwell (C++ / Eigen)

## 📖 Description

Ce projet implémente une **simulation spectrale en 1D** d’un plasma collisionless modélisé par les **équations bi-fluide (ions + électrons) couplées à Maxwell**.  
L’objectif est de reproduire les **ondes longitudinales (Langmuir, ion-acoustiques)** et les **ondes électromagnétiques transverses** en régime linéaire.

Le code est écrit en **C++17**, utilise la bibliothèque **[Eigen](https://eigen.tuxfamily.org/)** pour l’algèbre linéaire (vecteurs, matrices, valeurs propres), et est compilé avec un **Makefile** portable.

---

## ⚡ Caractéristiques principales

- Représentation **spectrale (FFT)** en espace (1D périodique).  
- **Linéarisation** du système bi-fluide + Maxwell.  
- Découpage en blocs indépendants :
  - Bloc longitudinal (plasma électrostatique).  
  - Bloc transverse (ondes électromagnétiques).  
- Évolution temporelle par :
  - Diagonalisation des matrices pour chaque mode \(k\), ou  
  - Application d’un schéma d’intégration (ex. RK4).  
- Analyse de la **relation de dispersion numérique** par calcul des valeurs propres.

---

## 📦 Dépendances

- **C++17** ou supérieur  
- **Eigen 3** (bibliothèque header-only)  
- **FFTW3** *(optionnel, si FFT rapide est utilisée au lieu d’un FFT maison)*  

Sous Debian/Ubuntu :

```bash
sudo apt-get install libeigen3-dev libfftw3-dev
