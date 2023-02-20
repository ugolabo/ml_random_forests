# Random Forests, Forêts aléatoires

Construire un modèle qui permet de prédire si un individu est ou sera obèse à partir des métriques de son hygiène de vie.

C'est une classification binaire: 0, non obèse ou 1, obèse.

Il faut d'abord entrainer le modèle avec des données d'individus dont on connait le poids, l'Indice de Masse Corporelle (qui permet de le qualifié d'obèse ou non) et d'autres métriques de leur hygiène de vie.

## Fichiers

- 1 notebook:
- 2 codes sources:
    - `rf_classification.py` comporte une classe pour faire une prédiction
    - `main.py` utilise le fichier précédent et sa classe pour faire une prédiction
- 1 fichier Excel: `questionnaire.xlsx` permet d'entrer des données et alimenter `rf_classification.py` 
- 1 dossier data: pour les données d'origine (CSV) et les données nettoyées  (PKL) pour l'entrainement du modèle
- 1 dossier img: pour les images

## Sources

Jeu de données disponible sur diverses plateformes:

- Kaggle:  https://www.kaggle.com/code/mpwolke/obesity-levels-life-style/notebook
- Github: https://github.com/zeglam/Obesity_Levels_Analysis

Les données accompagnent un article scientifique repris dans divers journaux:

- National Center for Biotechnology Information,  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6710633/
	- Science Direct,  https://www.sciencedirect.com/science/article/pii/S2352340919306985
	- etc.

Le projet est inspiré de d'analyses, de visualisations sur des enjeux de santé publique à partir de facteurs socio-économique: https://www.gapminder.org/data/

## Données d'origine et données utilisées

Il n'y a pas de valeurs manquantes, abérrantes ou autres anomalies à nettoyer.

Par contre, le jeu de données compte plusieurs colonnes de données non-numériques. Ces colonnes contiennent des types Pandas `object` (similaires au type Python `str`). Ce sont des données qualitatives ou catégoriques. Les autres colonnes du jeu de données sont numériques.
  
D'abord, il s'agit de transformer les types `object` en types `category` et de standardiser les catégories. Par exemple, des features binomiaux de catégories binaires (`femme`-`homme` ou `non`-`oui`), des features multinomiaux de catégories ordinales (`jamais`, `parfois`, `souvent`) et des features faits de tranches de données numériques continues (`moins 1 litre`, `1–2 litres`, `plus 2 litres`).

Les features faits de tranches de données numériques continues doivent être ordonnées (`moins 1 litre` < `1–2 litres` < `plus 2 litres`). Les catégories non-numériques, qui ne sont pas des tranches de données numériques continues, ont tendance à suivre un ordre alphanumérique. Les catégories ordinales (`jamais` < `parfois` < `souvent`) doivent être ordonnées (comme une séquence numérique) pour rester logiques.  Les catégories binaires  (`femme`-`homme` ou `non`-`oui`)  ne sont pas ordonnées.

Les données catégoriques permettent d'explorer les données, mais elles ne peuvent pas alimenter un algorithme de ML.

Consulter:  https://machinelearningknowledge.ai/categorical-data-encoding-with-sklearn-labelencoder-and-onehotencoder/

Il faut donc créer un équivalent numérique (discret ou entier) dans une autre colonne. Par exemple, `femme`-`homme` devient `0`-`1`, `jamais`, `parfois`, `souvent` deviennent `0`, `1`, `2` et  `moins 1 litre`, `1–2 litres`, `plus 2 litres` deviennent `1`, `2`, `3`.

Consulter:   https://stackoverflow.com/questions/38088652/pandas-convert-categories-to-numbers

Parfois, il n'y a pas d'ordre dans le feature, mais plusieurs catégories non ordinales ou des données qualificatives (`Automobile`, `Motorbike`, `Bike`, etc). Il faut encoder chaque catégorie dans de nouvelle colonnes de données binaires numériques ( `0`-`1`). 5 qualitatifs donnent 5 nouvelles colonnes. À chaque observation, une seule des 5 colonnes montre `1` dans la colonne du transport désigné et les autres montrent `0`.

Consulter: https://datagy.io/sklearn-random-forests

Le type `category` est un compromis entre le type `object` et un type numérique, car il permet de faire des calculs tout en affichant des chaines de caractères.

Consulter:

- https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html#differences-to-r-s-factor
- https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html

Les calculs statistiques (`mean()`, `std()`, etc., `describe()`), les tris (`sort_values()`), les opérateurs de comparaison (`==`, `>`, etc.) et les masques d'extraction (`loc[masque]`) qui les utilisent, les méthodes de modélisation comme `groupby()` fonctionnent sur les données catégoriques.

Certains graphiques fonctionnent très bien avec les données catégoriques. Le module Seaborn est particulièrement bien pourvu pour visualiser des données catégoriques.

Le fichier CSV d'origine est traité et enrichi. Le résultat est sauvegardé dans un format Pickle (binaire) qui permet de préserver le `DataFrame` et ses types.

Consulter: 

- https://datascienceparichay.com/article/save-pandas-dataframe-to-a-pickle-file/
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html
