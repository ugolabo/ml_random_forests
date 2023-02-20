# *Machine Learning* avec des modèles de *Random Forests*; prédire l'obésité

**Objectif:** maitriser les fondements de la science des données comme phase exploratoire et analytique d'un projet de *Machine Learning*.

[Projet](#projet-avec-des-mod%C3%A8les-de-random-forests-et-plus) avec des modèles de *Random Forests* (et plus) à la toute fin.

1. Science des données: faire une analyse de statistiques descriptives (tendance centrale, dispersion, extrêmes, intervalles, visualisation), préparer les données (rectifier la disposition, nettoyer ou remplacer les valeurs manquantes), modéliser les données avec des filtres, des tris, des pivots, etc. et pour faire des statistiques inférentielles (lois, distributions, extrapolation d'une population à partir d'un échantillon)

<img src="img/numpy_stack.jpg" alt="" width="300">

2. *Machine Learning*: définir les observations (lignes) et les *features* (colonnes), préparer les jeux de données d'entrainement pour alimenter des algorithmes supervisés de régression et de classification et des algorithmes non supervisés, normaliser et standardiser les données, tester les résultats avec des jeux de données de tests, visualiser les données et les résultats

<img src="img/ml_algorithms.jpg" alt="" width="600">

<img src="img/scikit_learn.jpg" alt="" width="600">

5. *Deep Learning*: aborder des cas simples avec TensorFlow et Keras

<img src="img/tensorflow_keras.jpg" alt="" width="500">

## Projet avec des modèles de *Random Forests* (et plus)

Le projet utilise un modèle de classification, les *Random Forests*, pour déterminer si un individu est ou sera obèse à partir des métriques de son hygiène de vie. Le modèle est précédé d'une analyse exploratoire des *features* (corrélations, histogrammes, etc.).

|    |    |
|:---|:---|
| <img src="img/correlation.jpg" alt="" width="400"> | <img src="img/histogrammes.jpg" alt="" width="400"> |

L'analyse se poursuit sur la cible: l'obésité. Dans un modèle de classification, le cible peut être multinomiale ou binomiale. L'idée des de dériver des hypothèses explicatives.

|    |    |
|:---|:---|
| <img src="img/obesite.jpg" alt="" width="400"> | <img src="img/obesite2.jpg" alt="" width="400"> |
| <img src="img/obesite3.jpg" alt="" width="400"> | <img src="img/obesite4.jpg" alt="" width="400"> |

Le modèle est entrainé avec des *features* principalement catégoriques; comme des catégories de fréquences de consommation d'aliments hypercaloriques, de consommation de légumes avec les repas, de collations entre les repas, etc. Avec les *Random Forests*, il faut trouver le modèle qui maximise la justesse, mais qui optimise le nombre d'arbres de décision pour limiter le temps de calcul (les *Random Forests* sont des modèles d'ensembles comptant plusieurs arbres décision).

|    |    |
|:---|:---|
| <img src="img/apprentissage.jpg" alt="" width="400"> | <img src="img/arbre.jpg" alt="" width="400"> |

Finalement, le modèle est évalué et en plus d'être comparé à d'autres modèles de classification.

|    |    |
|:---|:---|
| <img src="img/confusion.jpg" alt="" width="400"> | <img src="img/apprentissage2.jpg" alt="" width="400"> |

Le modèle peut être utilisé pour faire des prévision avec de nouvelles observations.
