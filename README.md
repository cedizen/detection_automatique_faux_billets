![Python](https://img.shields.io/badge/Python-blue?logo=python) ![Jupyter](https://img.shields.io/badge/Jupyter-orange?logo=jupyter) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

# Détection automatique de faux billets

Ce projet porte sur la mise en place d'un modèle capable de distinguer automatiquement de vrais billets de faux billets à partir de leurs caractéristiques géométriques.

L'objectif était à la fois de comprendre les données disponibles, de tester plusieurs approches de modélisation et d'aboutir à une méthode de classification exploitable.

## À propos du projet

Le jeu de données contient différentes mesures prises sur des billets.  
À partir de ces variables, le but est de déterminer si un billet est authentique ou non.

Le travail ne s'est pas limité à entraîner un modèle directement. Il a d'abord fallu explorer les données, repérer d'éventuels problèmes, mieux comprendre les relations entre variables, puis comparer plusieurs approches avant de retenir les plus pertinentes.

## Ce qu'on trouve dans ce repo

- un notebook principal avec l'exploration, le nettoyage et la modélisation ;
- éventuellement un ou plusieurs fichiers de données ;
- les éléments nécessaires pour suivre la démarche du projet ;
- ce `README.md`.

## Ce qui a été fait

Le projet comprend notamment :

- une exploration des variables ;
- un traitement des valeurs manquantes ou des points à surveiller dans les données ;
- une analyse descriptive ;
- une réduction de dimension / visualisation des données ;
- des essais de classification supervisée et non supervisée ;
- une modélisation pour prédire si un billet est vrai ou faux.

Selon la version du projet, on peut notamment retrouver :
- une ACP pour mieux visualiser la structure des données ;
- un clustering ;
- une régression logistique pour la classification.

## Outils utilisés

Le projet a été réalisé en Python, avec des bibliothèques classiques d'analyse de données et de machine learning, notamment :

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- Jupyter Notebook

## Lancer le projet

1. Cloner le dépôt :

```bash
git clone https://github.com/cedizen/detection_automatique_faux_billets.git
cd detection_automatique_faux_billets
```

2. Créer un environnement virtuel si besoin :

```bash
python -m venv .venv
source .venv/bin/activate
```

Sous Windows :

```bash
.\.venv\Scripts\activate
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

Si le fichier `requirements.txt` n'est pas encore présent dans le repo, il faudra installer manuellement les bibliothèques utilisées.

4. Lancer Jupyter Notebook :

```bash
jupyter notebook
```

5. Ouvrir le notebook principal et exécuter les cellules dans l'ordre.

## Objectif du projet

L'idée était de construire une démarche complète autour d'un problème de classification :

- comprendre les données ;
- tester plusieurs méthodes ;
- évaluer leur capacité à séparer les vrais billets des faux ;
- retenir une approche cohérente avec le problème posé.

C'est donc un projet qui mélange analyse exploratoire, statistiques et machine learning.

## Ce que ce projet montre

Au-delà du sujet lui-même, ce projet permet surtout de montrer :

- une démarche d'analyse structurée ;
- un travail de préparation de données ;
- l'utilisation de méthodes de classification ;
- la capacité à comparer plusieurs approches ;
- la volonté d'expliquer les résultats plutôt que de seulement produire un score.

## Pistes d'amélioration

Quelques prolongements possibles :

- comparer plus finement les performances des modèles ;
- ajouter une matrice de confusion et d'autres métriques ;
- industrialiser la prédiction dans une petite application ;
- tester d'autres algorithmes de classification ;
- mieux formaliser la phase d'évaluation du modèle final.

## Auteur

Projet réalisé par Cédric Berthezene.

GitHub : [cedizen](https://github.com/cedizen)
