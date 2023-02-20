import os
import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import OneHotEncoder


# Facultatif
# Déterminer le répertoire de travail
os.chdir(r'C:\Users\Usager\Documents\Ahuntsic\__420-318-AH INTELLIGENCE ARTIFICIELLE III\Projet1')


# Importer les données du fichier CSV
data: pd.DataFrame \
    = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv',
                  sep=",",
                  decimal=',')


# Nettoyer chaque colonne...

# Genre

# Explorer
data['Gender'].unique()
data['Gender'].dtypes

# Convertir object en category
data['Gender']: pd.DataFrame = pd.Categorical(data['Gender'], ordered=False)

# Explorer
data['Gender'].unique()
data['Gender'].dtypes
data['Gender'].cat.categories

# Créer une colonne numérique
# ['Female', 'Male'] équivaut à [0, 1]
data['Gender_n']: pd.DataFrame = data['Gender'].cat.codes

# Explorer
data['Gender_n'].unique()
data['Gender_n'].dtypes

# Tester
data[['Gender', 'Gender_n']].loc[data['Gender'] == 'Male'].head(3)
data[['Gender', 'Gender_n']].loc[data['Gender'] == 'Female'].head(3)


# Âge

# Redimensionner la colonne numérique
data['Age']: pd.DataFrame = data['Age'].astype(np.float16)

# Explorer
data['Age'].unique()
data['Age'].dtypes


# Taille, en mètres

# Redimensionner la colonne numérique
data['Height']: pd.DataFrame = data['Height'].astype(np.float16)

# Explorer
data['Height'].unique()
data['Height'].dtypes


# Masse, en kilogrammes

# Redimensionner la colonne numérique
data['Weight']: pd.DataFrame = data['Weight'].astype(np.float16)

# Explorer
data['Weight'].unique()
data['Weight'].dtypes


# Famille souffrant ou ayant souffert d'obésité

# Changer le nom de la colonne pour FHWO
data['FHWO'] = data['family_history_with_overweight']

# Supprimer l'ancienne colonne
data.drop(labels='family_history_with_overweight', axis=1, inplace=True)

# Explorer
data['FHWO'].unique()
data['FHWO'].dtypes

# Convertir object en category
data['FHWO']: pd.DataFrame = pd.Categorical(data['FHWO'], ordered=False)

# Explorer
data['FHWO'].unique()
data['FHWO'].dtypes
data['FHWO'].cat.categories

# Créer une colonne numérique
# ['no', 'yes'] équivaut à [0, 1]
data['FHWO_n']: pd.DataFrame = data['FHWO'].cat.codes

# Explorer
data['FHWO_n'].unique()
data['FHWO_n'].dtypes

# Tester
data[['FHWO', 'FHWO_n']].loc[data['FHWO'] == 'no'].head(3)
data[['FHWO', 'FHWO_n']].loc[data['FHWO'] == 'yes'].head(3)


# Haute fréquence de consommation d'aliments hypercaloriques

# Explorer
data['FAVC'].unique()
data['FAVC'].dtypes

# Convertir object en category
data['FAVC']: pd.DataFrame = pd.Categorical(data['FAVC'], ordered=False)

# Explorer
data['FAVC'].unique()
data['FAVC'].dtypes
data['FAVC'].cat.categories

# Créer une colonne numérique
# ['no', 'yes'] équivaut à [0, 1]
data['FAVC_n']: pd.DataFrame = data['FAVC'].cat.codes

# Explorer
data['FAVC_n'].unique()
data['FAVC_n'].dtypes

# Tester
data[['FAVC', 'FAVC_n']].loc[data['FAVC'] == 'no'].head(3)
data[['FAVC', 'FAVC_n']].loc[data['FAVC'] == 'yes'].head(3)


# Consommation de légumes avec les repas

# Explorer
data['FCVC'].unique()
data['FCVC'].dtypes

# Arrondir,
# convertir en entier pour retrouver 0, 1, 2
# et redimensionner la colonne
data['FCVC_n'] = data['FCVC'].round(0).astype(np.int8) - 1

# Explorer
data['FCVC_n'].unique()
data['FCVC_n'].dtypes

# Convertir en category
# ['never', 'sometimes', 'always'] équivaut à [0, 1, 2]
data['FCVC']: pd.DataFrame = data['FCVC'].round(0).astype(np.int8) - 1
data['FCVC']: pd.DataFrame \
    = pd.Categorical(data['FCVC'], categories=[0, 1, 2], ordered=True)

# Explorer
data['FCVC'].unique()
data['FCVC'].dtypes
data['FCVC'].cat.categories

# Renommer les catégories
data['FCVC']: pd.DataFrame \
    = data['FCVC'].cat.rename_categories({0: 'never',
                                          1: 'sometimes',
                                          2: 'always'})

# Explorer
data['FCVC'].dtypes
data['FCVC'].unique()
data['FCVC'].cat.categories

# Tester
data[['FCVC', 'FCVC_n']].loc[data['FCVC'] == 'never'].head(3)
data[['FCVC', 'FCVC_n']].loc[data['FCVC'] == 'sometimes'].head(3)
data[['FCVC', 'FCVC_n']].loc[data['FCVC'] == 'always'].head(3)


# Nombre quotidien de repas

# Explorer
data['NCP'].unique()
data['NCP'].dtypes

# Arrondir,
# convertir en entier pour retrouver 1, 2, 3, 4
# et redimensionner la colonne
data['NCP_n'] = data['NCP'].round(0).astype(np.int8)

# Explorer
data['NCP_n'].unique()
data['NCP_n'].dtypes

# Convertir en category
# ['1', '2', '3', '4+'] équivaut à [1, 2, 3, 4]
data['NCP']: pd.DataFrame = data['NCP'].round(0).astype(np.int8)
data['NCP']: pd.DataFrame \
    = pd.Categorical(data['NCP'], categories=[1, 2, 3, 4], ordered=True)

# Explorer
data['NCP'].unique()
data['NCP'].dtypes
data['NCP'].cat.categories

# Renommer les catégories
data['NCP']: pd.DataFrame \
    = data['NCP'].cat.rename_categories({1: '1',
                                         2: '2',
                                         3: '3',
                                         4: '4+'})

# Explorer
data['NCP'].dtypes
data['NCP'].unique()
data['NCP'].cat.categories

# Tester
data[['NCP', 'NCP_n']].loc[data['NCP'] == '1'].head(3)
data[['NCP', 'NCP_n']].loc[data['NCP'] == '2'].head(3)
data[['NCP', 'NCP_n']].loc[data['NCP'] == '3'].head(3)
data[['NCP', 'NCP_n']].loc[data['NCP'] == '4+'].head(3)


# Collations entre les repas

# Explorer
data['CAEC'].unique()
data['CAEC'].dtypes

# Convertir object en category
data['CAEC']: pd.DataFrame \
    = pd.Categorical(data['CAEC'], ['no',
                                    'Sometimes',
                                    'Frequently',
                                    'Always'],
                     ordered=True)

# Explorer
data['CAEC'].unique()
data['CAEC'].dtypes
data['CAEC'].cat.categories

# Créer une colonne numérique
# ['no', 'Sometimes', 'Frequently', 'Always'] équivaut à [0, 1, 2, 3]
data['CAEC_n']: pd.DataFrame = data['CAEC'].cat.codes

# Explorer
data['CAEC_n'].unique()
data['CAEC_n'].dtypes

# Tester
data[['CAEC', 'CAEC_n']].loc[data['CAEC'] == 'no'].head(3)
data[['CAEC', 'CAEC_n']].loc[data['CAEC'] == 'Sometimes'].head(3)
data[['CAEC', 'CAEC_n']].loc[data['CAEC'] == 'Frequently'].head(3)
data[['CAEC', 'CAEC_n']].loc[data['CAEC'] == 'Always'].head(3)


# Tabagisme

# Explorer
data['SMOKE'].unique()
data['SMOKE'].dtypes

# Convertir object en category
data['SMOKE']: pd.DataFrame = pd.Categorical(data['SMOKE'], ordered=False)

# Explorer
data['SMOKE'].unique()
data['SMOKE'].dtypes
data['SMOKE'].cat.categories

# Créer une colonne numérique
# ['no', 'yes'] équivaut à [0, 1]
data['SMOKE_n']: pd.DataFrame = data['SMOKE'].cat.codes

# Explorer
data['SMOKE_n'].unique()
data['SMOKE_n'].dtypes

# Tester
data[['SMOKE', 'SMOKE_n']].loc[data['SMOKE'] == 'no'].head(3)
data[['SMOKE', 'SMOKE_n']].loc[data['SMOKE'] == 'yes'].head(3)


# Consommation quotidienne d'eau

# Explorer
data['CH2O'].unique()
data['CH2O'].dtypes

# Arrondir,
# convertir en entier pour retrouver 1, 2, 3, 4
# et redimensionner la colonne
data['CH2O_n'] = data['CH2O'].round(0).astype(np.int8)

# Explorer
data['CH2O_n'].unique()
data['CH2O_n'].dtypes

# Convertir en category
# ['less_than_a_liter', '1–2_liters', 'more_than_2_liters'] équivaut à [1, 2, 3]
data['CH2O']: pd.DataFrame = data['CH2O'].round(0).astype(np.int8)
data['CH2O']: pd.DataFrame \
    = pd.Categorical(data['CH2O'], categories=[1, 2, 3], ordered=True)

# Explorer
data['CH2O'].unique()
data['CH2O'].dtypes
data['CH2O'].cat.categories

# Renommer les catégories
data['CH2O']: pd.DataFrame \
    = data['CH2O'].cat.rename_categories({1: 'less_than_a_liter',
                                          2: '1–2_liters',
                                          3: 'more_than_2_liters'})

# Explorer
data['CH2O'].dtypes
data['CH2O'].unique()
data['CH2O'].cat.categories

# Tester
data[['CH2O', 'CH2O_n']].loc[data['CH2O'] == 'less_than_a_liter'].head(3)
data[['CH2O', 'CH2O_n']].loc[data['CH2O'] == '1–2_liters'].head(3)
data[['CH2O', 'CH2O_n']].loc[data['CH2O'] == 'more_than_2_liters'].head(3)


# Surveillance de sa consommation colorique

# Explorer
data['SCC'].unique()
data['SCC'].dtypes

# Convertir object en category
data['SCC']: pd.DataFrame = pd.Categorical(data['SCC'], ordered=False)

# Explorer
data['SCC'].unique()
data['SCC'].dtypes
data['SCC'].cat.categories

# Créer une colonne numérique
# ['no', 'yes'] équivaut à [0, 1]
data['SCC_n']: pd.DataFrame = data['SCC'].cat.codes

# Explorer
data['SCC_n'].unique()
data['SCC_n'].dtypes

# Tester
data[['SCC', 'SCC_n']].loc[data['SCC'] == 'no'].head(3)
data[['SCC', 'SCC_n']].loc[data['SCC'] == 'yes'].head(3)


# Fréquence d'activités physiques

# Explorer
data['FAF'].unique()
data['FAF'].dtypes

# Arrondir,
# convertir en entier pour retrouver 0, 1, 2, 3
# et redimensionner la colonne
data['FAF_n']: pd.DataFrame = data['FAF'].round(0).astype(np.int8)

# Explorer
data['FAF_n'].unique()
data['FAF_n'].dtypes

# Convertir en category
# ['none', '1_to_2_days', '2_to_4_days', '4_to_5_days'] équivaut à [0, 1, 2, 3]
data['FAF']: pd.DataFrame = data['FAF'].round(0).astype(np.int8)
data['FAF']: pd.DataFrame \
    = pd.Categorical(data['FAF'], categories=[0, 1, 2, 3], ordered=True)

# Explorer
data['FAF'].unique()
data['FAF'].dtypes
data['FAF'].cat.categories

# Renommer les catégories
data['FAF']: pd.DataFrame\
    = data['FAF'].cat.rename_categories({0: 'none',
                                         1: '1_to_2_days',
                                         2: '2_to_4_days',
                                         3: '4_to_5_days'})

# Explorer
data['FAF'].dtypes
data['FAF'].unique()
data['FAF'].cat.categories

# Tester
data[['FAF', 'FAF_n']].loc[data['FAF'] == 'none'].head(3)
data[['FAF', 'FAF_n']].loc[data['FAF'] == '1_to_2_days'].head(3)
data[['FAF', 'FAF_n']].loc[data['FAF'] == '2_to_4_days'].head(3)
data[['FAF', 'FAF_n']].loc[data['FAF'] == '4_to_5_days'].head(3)


# Temps quotidien d'utilisation d'appareils
# (mobile, jeux vidéo, TV, ordinateur, etc.)

# Explorer
data['TUE'].unique()
data['TUE'].dtypes

# Arrondir,
# convertir en entier pour retrouver 1, 2, 3
# et redimensionner la colonne
data['TUE_n'] = data['TUE'].round(0).astype(np.int8) + 1

# Explorer
data['TUE_n'].unique()
data['TUE_n'].dtypes

# Convertir en category
# ['0–2_hours', '3–5_hours', 'more_than_5_hours'] équivaut à [1, 2, 3]
data['TUE']: pd.DataFrame = data['TUE'].round(0).astype(np.int8) + 1
data['TUE']: pd.DataFrame \
    = pd.Categorical(data['TUE'], categories=[1, 2, 3], ordered=True)

# Explorer
data['TUE'].unique()
data['TUE'].dtypes
data['TUE'].cat.categories

# Renommer les catégories
data['TUE']: pd.DataFrame \
    = data['TUE'].cat.rename_categories({1: '0–2_hours',
                                         2: '3–5_hours',
                                         3: 'more_than_5_hours'})

# Explorer
data['TUE'].dtypes
data['TUE'].unique()
data['TUE'].cat.categories

# Tester
data[['TUE', 'TUE_n']].loc[data['TUE'] == '0–2_hours'].head(3)
data[['TUE', 'TUE_n']].loc[data['TUE'] == '3–5_hours'].head(3)
data[['TUE', 'TUE_n']].loc[data['TUE'] == 'more_than_5_hours'].head(3)


# Consommation d'alcool

# Explorer
data['CALC'].unique()
data['CALC'].dtypes

# Convertir object en category
data['CALC']: pd.DataFrame \
    = pd.Categorical(data['CALC'], ['no',
                                    'Sometimes',
                                    'Frequently',
                                    'Always'],
                     ordered=True)

# Explorer
data['CALC'].unique()
data['CALC'].dtypes
data['CALC'].cat.categories

# Créer une colonne numérique
# ['no', 'Sometimes', 'Frequently', 'Always'] équivaut à [0, 1, 2, 3]
data['CALC_n']: pd.DataFrame = data['CALC'].cat.codes

# Explorer
data['CALC_n'].unique()
data['CALC_n'].dtypes

data[['CALC', 'CALC_n']].loc[data['CALC'] == 'no'].head(3)
data[['CALC', 'CALC_n']].loc[data['CALC'] == 'Sometimes'].head(3)
data[['CALC', 'CALC_n']].loc[data['CALC'] == 'Frequently'].head(3)
data[['CALC', 'CALC_n']].loc[data['CALC'] == 'Always'].head(3)


# Transport le plus utilisé

# Explorer
data['MTRANS'].unique()
data['MTRANS'].dtypes

# Encoder chaque catégorie du feature
one_hot = OneHotEncoder()
encode = one_hot.fit_transform(data['MTRANS'].values.reshape(-1, 1))
data[one_hot.categories_[0]] = encode.toarray()

# Valider
data.info()

# Renommer les colonnes
data.rename(columns={'Automobile': 'Automobile_n',
                     'Motorbike': 'Motorbike_n',
                     'Bike': 'Bike_n',
                     'Public_Transportation': 'Public_Transportation_n',
                     'Walking': 'Walking_n'}, inplace=True)

# Valider
data.info()

# Convertir les colonnes en entier
data[['Automobile_n',
      'Motorbike_n',
      'Bike_n',
      'Public_Transportation_n',
      'Walking_n']] = \
    data[['Automobile_n',
          'Motorbike_n',
          'Bike_n',
          'Public_Transportation_n',
          'Walking_n']].astype(np.int8)
    
# Valider
data[['MTRANS',
      'Automobile_n',
      'Motorbike_n',
      'Bike_n',
      'Public_Transportation_n',
      'Walking_n']].head(3)

# Explorer (1 seul feature)
data['Automobile_n'].unique()
data['Automobile_n'].dtypes


# Niveau d'obésité

# Explorer
data['NObeyesdad'].unique()
data['NObeyesdad'].dtypes

# Convertir object en category
data['NObeyesdad']: pd.DataFrame \
    = pd.Categorical(data['NObeyesdad'], ['Insufficient_Weight',
                                          'Normal_Weight',
                                          'Overweight_Level_I',
                                          'Overweight_Level_II',
                                          'Obesity_Type_I',
                                          'Obesity_Type_II',
                                          'Obesity_Type_III'],
                     ordered=True)

# Explorer
data['NObeyesdad'].unique()
data['NObeyesdad'].dtypes
data['NObeyesdad'].cat.categories

# Créer une colonne numérique
# ['Insufficient_Weight', 'Normal_Weight',
# 'Overweight_Level_I', 'Overweight_Level_II',
# 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
# équivaut à [1, 2, 3, 4, 5, 6, 7]
data['NObeyesdad_n']: pd.DataFrame = data['NObeyesdad'].cat.codes + 1

# Explorer
data['NObeyesdad_n'].unique()
data['NObeyesdad_n'].dtypes


# Créer une autre colonne
# grâce à une fonction...
def simplifier_NObeyesdad(x: str) -> str:
    if (x in ['Insufficient_Weight', 'Normal_Weight',
              'Overweight_Level_I', 'Overweight_Level_II']):
        return 'not_obese'
    else:
        return 'obese'

# ...appliquée à la colonne
# et envoyer le résultat dans une nouvelle colonne
data['NObeyesdad2']: pd.DataFrame \
    = data['NObeyesdad'].apply(simplifier_NObeyesdad)

# Convertir object en category
data['NObeyesdad2']: pd.DataFrame \
    = pd.Categorical(data['NObeyesdad2'], ['not_obese', 'obese'], ordered=False)

# Explorer
data['NObeyesdad2'].unique()
data['NObeyesdad2'].dtypes
data['NObeyesdad2'].cat.categories

# Créer une colonne numérique
# ['not_obese', 'obese'] équivaut à [0, 1]
data['NObeyesdad_n2']: pd.DataFrame = data['NObeyesdad2'].cat.codes

# Explorer
data['NObeyesdad_n2'].unique()
data['NObeyesdad_n2'].dtypes

# Tester
data[['NObeyesdad',
      'NObeyesdad_n',
      'NObeyesdad2',
      'NObeyesdad_n2']].loc[data['NObeyesdad'] == 'Insufficient_Weight'].head(3)
data[['NObeyesdad',
      'NObeyesdad_n',
      'NObeyesdad2',
      'NObeyesdad_n2']].loc[data['NObeyesdad'] == 'Normal_Weight'].head(3)
data[['NObeyesdad',
      'NObeyesdad_n',
      'NObeyesdad2',
      'NObeyesdad_n2']].loc[data['NObeyesdad'] == 'Overweight_Level_I'].head(3)
data[['NObeyesdad',
      'NObeyesdad_n',
      'NObeyesdad2',
      'NObeyesdad_n2']].loc[data['NObeyesdad'] == 'Overweight_Level_II'].head(3)
data[['NObeyesdad',
      'NObeyesdad_n',
      'NObeyesdad2',
      'NObeyesdad_n2']].loc[data['NObeyesdad'] == 'Obesity_Type_I'].head(3)
data[['NObeyesdad',
      'NObeyesdad_n',
      'NObeyesdad2',
      'NObeyesdad_n2']].loc[data['NObeyesdad'] == 'Obesity_Type_II'].head(3)
data[['NObeyesdad',
      'NObeyesdad_n',
      'NObeyesdad2',
      'NObeyesdad_n2']].loc[data['NObeyesdad'] == 'Obesity_Type_III'].head(3)


# Valider le nettoyage
data.info()


# Réordonner les colonnes pour
# associer les colonnes catégoriques et numériques
data2: List[str] = data.reindex(columns=[
    'Gender',
    'Gender_n',
    'Age',
    'Height',
    'Weight',
    'FHWO',
    'FHWO_n',
    'FAVC',
    'FAVC_n',
    'FCVC',
    'FCVC_n',
    'NCP',
    'NCP_n',
    'CAEC',
    'CAEC_n',
    'SMOKE',
    'SMOKE_n',
    'CH2O',
    'CH2O_n',
    'SCC',
    'SCC_n',
    'FAF',
    'FAF_n',
    'TUE',
    'TUE_n',
    'CALC',
    'CALC_n',
    'MTRANS',
    'Automobile_n',
    'Motorbike_n',
    'Bike_n',
    'Public_Transportation_n',
    'Walking_n',
    'NObeyesdad',
    'NObeyesdad_n',
    'NObeyesdad2',
    'NObeyesdad_n2'])

# Valider le nettoyage
data2.info()
data2.head(3)

# Sauvegarder dans un format binaire (Pickle)
# pour préserver les types
data2.to_pickle('data/ObesityDataSet_raw_and_data_sinthetic2.pkl')
