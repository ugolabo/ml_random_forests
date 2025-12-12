# Machine Learning avec Random Forests

Le projet est une suite d'étapes (MLOps) pour entrainer un modèle de Machine Learning avec Python. Le modèle prédit la propension à l'obésité.

<img src="img/arbre.jpg" alt="" width="300px">

## Origine du projet

L'obésité peut être mesurée avec l'Indice de Masse Corporelle (IMC). L'IMC est une valeur continue.

On classe l'IMC en catégories : poids insuffisant, poids normal, surpoids de niveau I, surpoids de niveau II, obésité de type I, obésité type de II, obésité type de III. Les classements varient, mais ce sont des classements multinomiaux ; à plusieurs catégories.

On peut aussi regrouper toutes les classes obésité pour obtenir un classement binomial (à deux catégories) : non obèse ou obèse.

On entraine les modèles à établir un lien entre une catégorie d'IMC et des facteurs de vie (alimentation, activité, habitudes, etc.) chez des individus. Les facteurs de vie sont les features, les variables explicatives, les variables indépendantes ou x. Chaque individu est une observation ou ligne de données. La variable cible, la variable dépendante ou y est la catégorie d'IMC. Les facteurs de vie (les features x) déterminent la catégorie d'IMC (la variable cible y).

Avec de nouvelles données (de nouveaux individus (de nouvelles observations) et leurs facteurs de vie (les features x)), le modèle prédit la catégorie d'IMC des individus**** (la variable y). Le modèle multinomial prédit une des 7 catégories. Le modèle binomial prédit une des 2 catégories.

## Mise en place et structure

Ce sont les étapes (MLOps) qui permettent d'arriver à la **sauvegarde en format Pickle (format binaire) de données nettoyées, d'un scaler et de deux modèles**. Cette sauvegarde permet de pousuivre avec les étapes manquantes du MLOps : le déploiement du modèle. Le projet se termine dans le dépôt: **ml_random_forests_streamlit**.

Consulter aussi le dépôt à propos des apps interactives : **apps_streamlit**.

Tous les fichiers nommés dans le étapes qui suivrent se retrouvent dans dossier 'projet_ml/' de ce dépôt. Consulter aussi le README dans le dossier.

### Nettoyage des données

1. Exécuter le code source 'nettoyage_donnee.py'.
    - Le code source agit comme un pipeline ETL (Extract-Transform-Load).
    - Les données d'origine en CSV dans 'data/' sont importées au début, nettoyées, puis **sauvegardées** en Pickle à la fin. 

### Science des données

<img src="img/numpy_stack.jpg" alt="" width="300">

1. Exécuter les notebooks 'nb_v3.ipynb' et 'nb_v4.ipynb'.
    - Les v1 à v2 étaient exploratoires. Ils n'ont pas été retenus. Une partie de v2 a donnée l'étape précédente, mais en code source. L'autre partie de v2 a donnée le notebook v3.
    - v3 est plus long que v4, car il est expérimental. Il teste différents modèles de classification.
    - v4 reprend une partie de v3 et va à l'essentiel. C'est le notebook qui **sauvegarde** le scaler et les modèles en Pickle.
1. Faire une analyse de statistiques descriptives : tendance centrale, dispersion, distribution, valeurs extrêmes, corrélations, visualisation.
1. Manipuler les données : filtres ou extractions conditionnelles, tris, visualisation, etc.

|    |    |
|:---|:---|
| <img src="img/correlation.jpg" alt="" width="400"> | <img src="img/histogrammes.jpg" alt="" width="400"> |
| <img src="img/obesite.jpg" alt="" width="400"> | <img src="img/obesite2.jpg" alt="" width="400"> |
| <img src="img/obesite3.jpg" alt="" width="400"> | <img src="img/obesite4.jpg" alt="" width="400"> |

### Machine Learning

1. Définir les features x et la variable cible y (colonnes) et les observations (lignes).
1. Préparer les jeux de données d'entrainement, transformer les features. Cette étape débouche sur la **sauvegarde d'un scaler** en Pickle.
    - Cette transformation nivelle les grandes différences de variance. Ce qui évite qu'un feature avec une large variance absolue (-1M à 1M) ne marginalise pas un autre feature avec une petite variance absolue (-10 à 10) dans l'entrainement d'un modèle.
    - Le StandardScaler transforme les features en forçant chaque moyenne à 0. La variance de chaque feature se limite à la fourchette -1 et 1 (autour de la moyenne). Si un feature est très variable, il reste très variable, mais entre -1 et 1. Si un feature est plus stable, sa variance démontre cette stabilité dans la fourchette -1 et 1. La variance des features est alors comparable.
    - MinMaxScaler transforme les features en forçant la variance dans une fourchette de 0 et 1.
    - Il existe une panoplie de scalers. Chaque scaler s'applique en fonction d'une configuration de données. Par exemple, en présence de données non normales (Loi Normale) ou non paramétriques, de données avec beaucoup d'outliers qui biaisent la moyenne, là où la médiane et l'IQR sont moins influencés par les outliers, le RobustScaler est plus approprié.
1. Explorer les possibilités de modèles avant de converger : choix d'un modèle de classification. 

<img src="img/ml_algorithms.jpg" alt="" width="600">

<img src="img/scikit_learn.jpg" alt="" width="600">

1. Sélectionner un modèle de classification, expérimenter.
    - Un modèle de régression logistique est simple et rapide à entrainer. Ce modèle se montre sensible aux données et aux transformations des données d'entrainement ; il souffre de problèmes de biais élevé.
    - Un modèle d'arbre de décision simple est sensible aux petites variations dans les données d'entrainement. Un léger changement peut entrainer un arbre et une prédiction complètement différents. L'arbre échoue aussi à généraliser avec de nouvelles données, car il a cette tendance au sur-apprentissage (overfitting).
    - Il existe d'autres modèles de classification.
    - Les modèles d'ensembles sont parmi les meilleures options.
        - Les modèles de boosting (XGBoost, LightGBM) évitent le sur-apprentissage et réduisent les biais avec des techniques de régularisation et de sous-échantillonnage, par exemple.
        - Les modèles de stacking corrigent les problèmes en combinant les forces de plusieurs types de modèles (par exemple, un arbre de décision, une régression logistique, un autre modèle de classification), évitant qu'un seul modèle ne dicte la prédiction finale et offrant un meilleur pouvoir prédictif global.
        - Les modèles de bagging ressemblent aux modèles de stacking. C'est un ensemble d'arbres de décision : les Random Forests.
1. Choisir les Random Forests, car ils sont simples à maitriser, simples à interpréter, rapides à entrainer et peu intensifs en ressources.
1. Entrainer différents modèles (binomial et multinomial), tester les résultats avec des jeux de données de tests, visualiser les données et les résultats.

Avec les Random Forests, il faut trouver un modèle qui maximise la justesse (accuracy), mais qui optimise le nombre d'arbres de décision. Par exemple, le score de justesse du modèle binomial avec les données de test converge à 94.3% passé un ensemble de 23 estimateurs ou arbres de décision (le premier arbre de décision est illustré). 125 à 150 estimateurs sont les nombres d'arbres optimaux pour obtenir un score de 94.3% tout en limitant le temps d'entrainement.

|    |    |
|:---|:---|
| <img src="img/arbre.jpg" alt="" width="400"> | <img src="img/apprentissage.jpg" alt="" width="400"> |
| <img src="img/confusion.jpg" alt="" width="400"> | <img src="img/apprentissage2.jpg" alt="" width="400"> |

Finalement, une configuration de chaque modèle (binomial et multinomial) est retenue. Ces modèles sont entrainés, évalués et **sauvegardés** pour être mis en production (pour faire des prévisions avec de nouvelles observations).

Consulter le README dans le sous-répertoire du projet pour plus de détails.
