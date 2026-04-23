![Python](https://img.shields.io/badge/Python-blue?logo=python) ![Jupyter](https://img.shields.io/badge/Jupyter-orange?logo=jupyter) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

# Projet: Lutte contre la contrefaçon des billets de banque

Ce projet utilise des techniques de machine learning pour détecter les faux billets de banque. Le notebook `billets.ipynb` explore un jeu de données de 1 500 billets, décrits par leurs caractéristiques physiques et leur nature (vrai/faux).

## Objectifs du Projet

L'objectif principal est de développer et d'évaluer plusieurs modèles de classification afin d'identifier efficacement les billets contrefaits. Le projet couvre les étapes suivantes :

- **Exploration des données** : Comprendre la structure et les caractéristiques du jeu de données.
- **Préparation des données** : Gérer les valeurs manquantes et prétraiter les variables.
- **Entraînement et évaluation des modèles** : Mettre en œuvre et comparer des modèles supervisés et non supervisés.
- **Sélection du modèle** : Choisir le modèle le plus performant pour la détection des faux billets.

## Jeu de Données

Le jeu de données `billets.csv` contient 1 500 entrées avec les caractéristiques suivantes :

- 'is_genuine' (booléen) : Indique si le billet est authentique (True) ou faux (False). C'est la variable cible pour la classification supervisée.
- 'diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length' (float) : Caractéristiques physiques du billet.

## Méthodologie

### 1. Traitement des valeurs manquantes

Le champ `margin_low` contenait 37 valeurs manquantes. Après une analyse des corrélations et de la distribution des données, un modèle de **Régression Linéaire** a été utilisé pour prédire ces valeurs, car il a montré la meilleure performance parmi les modèles de régression testés (`LinearRegression`, `RidgeCV`, `DecisionTreeRegressor`, `RandomForestRegressor`). Le jeu de données a ensuite été complété avec ces prédictions.

### 2. Classification Supervisée

Plusieurs modèles de classification supervisée ont été entraînés et évalués en utilisant une stratégie de validation croisée (`GridSearchCV`) et la métrique F1-score, étant donné le déséquilibre de classes (`66.67%` de vrais billets).

Les modèles suivants ont été testés :
- **Régression Logistique**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **Decision Tree Classifier**

Le modèle le plus performant est sélectionné en fonction de son F1-score sur l'ensemble de test.

### 3. Clustering Non Supervisé (K-Means)

Pour explorer la structure intrinsèque des données sans labels, un modèle K-Means a été appliqué. Pour faciliter la visualisation et potentiellement améliorer les performances du clustering, une **Analyse en Composantes Principales (ACP)** a été utilisée pour réduire la dimensionnalité des données à 2 composantes principales.

Le nombre optimal de clusters a été déterminé à l'aide du **Coefficient de Silhouette**.

## Résultats Clés

### Modèle Supervisé

Le **KNeighborsClassifier** s'est avéré être le meilleur modèle supervisé, atteignant un F1-score de test de **98.75%** avec seulement **2 faux positifs** et **3 faux négatifs** sur l'ensemble de test. Ce modèle est considéré comme le plus pertinent pour la détection des faux billets en raison de sa haute précision et de sa capacité à généraliser sur de nouvelles données.

### Clustering K-Means

L'analyse par K-Means a identifié **2 clusters** principaux :
- **Cluster 0** : Contient environ **98.80% de vrais billets**.
- **Cluster 1** : Contient environ **2.98% de vrais billets**, ce qui indique une forte concentration de faux billets dans ce cluster.

La visualisation des clusters via les composantes principales (ACP1 et ACP2) a montré que les clusters se différencient principalement le long de l'ACP1, qui est corrélée aux billets longs et fins (Cluster 1) versus des billets plus courts et larges (Cluster 0).

## Comment Utiliser le Projet

1.  **Cloner le dépôt** :
    '''bash
    git clone <URL_DE_VOTRE_DEPOT>
    cd <NOM_DU_DEPOT>
    '''
2.  **Installer les dépendances** :
    '''bash
    pip install -r requirements.txt
    '''
3.  **Lancer Jupyter/Colab** :
    '''bash
    jupyter notebook billets.ipynb
    '''
    ou téléchargez le notebook sur Google Colab.
4. **Importer le fichier CSV dans le notebook**
5.  **Exécuter les cellules** : Exécutez toutes les cellules du notebook séquentiellement pour reproduire l'analyse et l'entraînement des modèles.
6.  **Exécuter le script 'main.py'** en :
   ''' bash python main.py --file input.csv'''

## Fichiers Générés

- 'billets_df_cleaned.csv' : Version du jeu de données avec les valeurs manquantes de `margin_low` imputées par le modèle de régression linéaire, et une colonne `id` ajoutée.
- 'model.pkl' : Le modèle supervisé le plus performant (KNeighborsClassifier) exporté au format `pickle`, prêt à être utilisé pour de nouvelles prédictions.
- 'predictions.csv': dataset input avec un nouveau champs 'prédictions' ajouté

## Auteur

cedizen
