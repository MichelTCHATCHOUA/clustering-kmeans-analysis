# 🤖 Clustering par l'Algorithme des K-Moyennes

Ce projet présente une étude complète de l'algorithme de **clustering non supervisé K-means** appliqué à des données synthétiques et à la base de données de chiffres manuscrits **Digits**. L'objectif est d'évaluer les performances de l'algorithme, de déterminer le nombre optimal de clusters et d'utiliser les centroïdes comme prototypes pour une tâche de classification.

---

## 🎯 Objectif

L'étude vise à maîtriser et analyser l'algorithme K-means à travers plusieurs axes :

- Appliquer l'algorithme K-means sur des données synthétiques et mesurer son **coût (inertie intra-cluster)**.
- Déterminer le **nombre optimal de clusters K** via la méthode du coude et le critère de **Calinski-Harabasz**.
- Évaluer la **stabilité** de l'algorithme à travers plusieurs initialisations.
- Appliquer K-means à la **reconnaissance de chiffres manuscrits** (base Digits) et mesurer la **pureté des clusters**.
- Utiliser les centroïdes K-means comme base d'apprentissage pour un classifieur **Plus-Proche-Voisin (PPV)**.
- Implémenter **manuellement** l'algorithme K-means et comparer les résultats avec `sklearn`.

---

## 📊 Données

### Données synthétiques
- [**base1.txt**](données/base1.txt) : 300 points répartis en **3 classes réelles**, représentés en 2D — clusters bien séparés.
- [**base3.txt**](données/base3.txt) : 600 points répartis en **4 classes réelles**, représentés en 2D — structure plus complexe avec zones de recouvrement.

### Base Digits (sklearn)
- **Source** : `sklearn.datasets.load_digits`
- **Échantillon** : 1797 images de chiffres manuscrits (0 à 9)
- **Descripteurs** : 64 pixels par image (8×8)
- **Classes** : 10 chiffres (0 à 9)
- **Découpage** : 70% apprentissage / 30% test via `train_test_split`

---

## 🛠️ Outils utilisés

- **Python 3**
- **Bibliothèques** :
  - `numpy`, `pandas` — manipulation des données
  - `matplotlib`, `seaborn` — visualisation
  - `scikit-learn` — KMeans, KNeighborsClassifier, métriques, Digits
- **Environnement** : Jupyter Notebook

---

## ⚙️ Prérequis et Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/MichelTCHATCHOUA/clustering-kmeans-analysis
cd clustering-kmeans-analysis
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 📂 Structure du projet
```text
TP2-Clustering-KMeans/
│
├── README.md                  # Ce fichier
├── requirements.txt           # Dépendances Python
├── TP2_TCHATCHOUA.pdf         # Rapport complet d'analyse
├── TP2_TCHATCHOUA.ipynb       # Notebook avec tous les calculs et graphiques
├── base1.txt                  # Données synthétiques - 3 classes
├── base3.txt                  # Données synthétiques - 4 classes
└── images/                    # Visualisations générées
    ├── base1&base3_clusters.png
    ├── elbow_base1.png
    ├── elbow_base3.png
    ├── calinski_base1.png
    ├── calinski_base3.png
    ├── recognition_rate.png
    └── kmeans_manual_vs_sklearn.png
```

---

## 🔬 Méthodologie

L'analyse est structurée en quatre parties :

### 1. Évaluation sur données synthétiques (base1 & base3)
- Application de K-means pour différentes valeurs de K.
- Observation visuelle de la distribution autour des centroïdes.
- Calcul et comparaison du coût (inertie) selon K.

### 2. Détermination du K optimal
- **5 initialisations** par valeur de K pour mesurer la stabilité (coût moyen et variance).
- **Méthode du coude** : identification du point d'inflexion de la courbe d'inertie.
- **Critère de Calinski-Harabasz** : maximisation du rapport variance inter/intra-cluster.
- Vérification de la **convergence** des centroïdes entre initialisations.

### 3. Application à la reconnaissance de chiffres (Digits)
- **Clustering** : K=10 clusters, calcul de la **pureté** par cluster.
- **Classification PPV** : les centroïdes servent de prototypes, évaluation du taux de reconnaissance pour k' = 1, 2, 3, 4 centroïdes par classe.

### 4. Implémentation manuelle
- Codage from scratch de l'algorithme K-means en Python.
- Comparaison des résultats avec l'implémentation `sklearn`.

---

## 💡 Résultats Clés

### Données synthétiques

| Base | K optimal | Coût moyen | Variance du coût | Score Calinski-Harabasz |
|------|-----------|------------|------------------|--------------------------|
| Base1 | **K = 3** | 18.50 | 0.000000 | ~1000 |
| Base3 | **K = 6** | 28.4 | ~0.000 | ~1415 |

- **Base1** : K = 3 correspond parfaitement aux 3 classes réelles, avec une stabilité excellente (variance nulle).
- **Base3** : bien que 4 classes réelles soient présentes, K = 6 est recommandé pour capturer les subdivisions internes.

### Clustering sur Digits (K = 10)

| Métrique | Valeur |
|----------|--------|
| Pureté Moyenne E[p] | **0.7760** |
| Variance de la Pureté Var[p] | **0.000180** |

➡️ La faible variance confirme la **robustesse** du clustering sur plusieurs initialisations.

### Classification PPV avec centroïdes K-means

| k' (centroïdes/classe) | Taux moyen de reconnaissance | Variance |
|------------------------|------------------------------|----------|
| 1 | ~0.90 | 0.000000 |
| 2 | ~0.92 | 0.000042 |
| 3 | ~0.95 | 0.000002 |
| 4 | ~0.97 | 0.000038 |

➡️ L'augmentation de k' améliore le taux de reconnaissance en modélisant mieux la variabilité intra-classe.

### Implémentation manuelle vs sklearn

| Implémentation | Inertie Finale | Stabilité |
|----------------|----------------|-----------|
| Manuelle | **18.4980** | ✅ OK |
| sklearn | **18.4980** | ✅ OK |

---

## 📊 Visualisations

### Distribution des clusters — Base1
Les données de `base1.txt` présentent **3 clusters bien séparés**. la répartition des points montre une structure relativement claire composée de plusieurs groupes distincts, bien séparés les uns des autres.

![Clusters Base1&Base3](images/Base1_distribution.png)

Cette configuration suggère que l’algorithme des k-moyennes devrait parvenir à regrouper efficacement les données avec un nombre de clusters modéré (autour de 3 à 5).

---

### Distribution des clusters — Base3
Les données de `base3.txt` sont plus complexes, avec des zones de recouvrement. Cela rend le choix du nombre de clusters k plus délicat.  

![Clusters Base3](images/Base3_distribution.png)

Il se souligne donc l’intérêt d’utiliser des critères objectifs tels que la méthode du coude ou le score de Calinski-Harabasz pour déterminer la valeur optimale de k.

---

### Méthode du Coude & Score Calinski-Harabasz
Les deux critères convergent vers les mêmes valeurs optimales :
- **Base1** : coude visible à K = 3, score CH maximal à K = 3.
- **Base3** : coude moins marqué, score CH orientant vers K = 6.

<table>
  <tr>
    <td align="center">
      <img src="images/Coude_base1.png" width="400"><br>
      <em>Méthode du coude — Base1</em>
    </td>
    <td align="center">
      <img src="images/Critère_CH_base1.png" width="400"><br>
      <em>Score Calinski-Harabasz — Base1</em>
    </td>
  </tr>
</table>

<br>

<table>
  <tr>
    <td align="center">
      <img src="images/Coude_base3.png" width="400"><br>
      <em>Méthode du coude — Base3</em>
    </td>
    <td align="center">
      <img src="images/Critère_CH_base3.png" width="400"><br>
      <em>Score Calinski-Harabasz — Base3</em>
    </td>
  </tr>
</table>

---

### Taux de reconnaissance en fonction de k'
La courbe montre une progression du **taux de reconnaissance** à mesure que le nombre de voisins *k'* augmente, avec une **faible variance** confirmant la stabilité des résultats sur plusieurs initialisations.

![Taux de reconnaissance](images/Reconnaissance.png)

On observe que l’augmentation de *k'* améliore progressivement la qualité de la classification, jusqu’à atteindre une zone de **stabilisation** où les gains deviennent marginaux, ce qui indique un choix de *k'* robuste.

---

### Comparaison manuelle vs sklearn (K = 3)
Les deux implémentations produisent des **visualisations identiques** et une **inertie finale parfaitement concordante** (**18.4980**), validant la **correction de l’implémentation manuelle** de l’algorithme K-means.

![K-means manuel vs sklearn](images/LastFig.png)

Cette concordance confirme que l’algorithme développé reproduit fidèlement le comportement d’une implémentation de référence, garantissant la fiabilité des résultats obtenus.

---

## 📝 Algorithme K-means — Rappel

L'algorithme itère entre deux étapes jusqu'à convergence :

**Affectation** : chaque point est assigné au cluster dont le centroïde est le plus proche.

$$
S_i^{(t)} = \lbrace \mathbf{x}_j : \|\mathbf{x}_j - \mathbf{m}_i^{(t)}\| \leq \|\mathbf{x}_j - \mathbf{m}_{i^\ast}^{(t)}\| \quad \forall \, i^\ast \rbrace
$$

**Mise à jour** : recalcul du barycentre de chaque cluster.

$$\mathbf{m}_i^{(t+1)} = \frac{1}{|S_i^{(t)}|} \sum_{\mathbf{x}_j \in S_i^{(t)}} \mathbf{x}_j$$

**Fonction de coût minimisée** :

$$\sum_{i=1}^{k} \sum_{\mathbf{x}_j \in S_i} \|\mathbf{x}_j - \mathbf{m}_i\|^2$$

---

## 💻 Code Source et Fichiers
*   **Notebook** : Le code complet pour le traitement des données, les calculs de variance et la génération des graphiques est disponible dans [codeSource.ipynb](TP2_TCHATCHOUA.ipynb).
*   **Rapport** : Pour une interprétation détaillée des courbes de densité et des matrices de covariance, consultez le [Rapport PDF](TP2_TCHATCHOUA.pdf).
*   
---

## 👤 Auteur

**Michel Peslier**
