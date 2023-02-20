from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib


class rf_classification:
    """Classe pour créer une instance de données par défaut
    et de modèle prédictif (l'algorithme de Random Forest); la classe
    permet d'importer de nouvelles de données, de les changer
    dans l'attribut d'instance et d'exécuter le modèle prédictif."""

    def __init__(self) -> Union[pd.DataFrame, RandomForestClassifier]:
        """Instantier des données de départ ou par défaut
        et un modèle prédictif(l'algorithme de Random Forest)"""

        # Constantes pour créer un DataFrame de donnée
        COLS: List[str] = ['Gender_n', 'Age', 'Height', 'FHWO_n', 'FAVC_n',
                           'FCVC_n', 'NCP_n', 'CAEC_n', 'SMOKE_n', 'CH2O_n',
                           'SCC_n', 'FAF_n', 'TUE_n', 'CALC_n',
                           'Automobile_n', 'Motorbike_n', 'Bike_n',
                           'Public_Transportation_n', 'Walking_n',
                           'Nobeyesdad_n', 'Nobeyesdad_n2']
        VALS: List[int, float] = [0, 20, 1.75, 0, 0,
                                  1, 3, 1, 0, 1,
                                  0, 1, 3, 1,
                                  0, 0, 0,
                                  4, 0,
                                  1, 1]

        # Attribut de départ ou par défaut
        self.questionnaire: pd.DataFrame \
            = pd.DataFrame(np.array([VALS]), columns=COLS)
        # Attribut
        self.modele_prediction: RandomForestClassifier \
            = joblib.load('modele/modele_rf.pkl')
        self.scaler: MinMaxScaler \
            = joblib.load('modele/scaler_rf.pkl')

    def importer_changer_donnees(self,
                         nom_fichier_ext: str,
                         nom_feuille: str) -> pd.DataFrame:
        """Importer de nouvelles données et
        changer les données avec les nouvelles pour le modèle prédictif"""

        self.donnees: pd.DataFrame = pd.read_excel(nom_fichier_ext,
                                                   sheet_name=nom_feuille,
                                                   skiprows=24,
                                                   decimal=',')
        self.questionnaire: pd.DataFrame = self.donnees
        return self.questionnaire

    def faire_classification(self) -> str:
        """Faire une classification (0 ou 1) avec le modèle prédictif
        et les données dans les attributs de l'instance"""
        
        donnees: pd.DataFrame = self.questionnaire
        donnees2: pd.DataFrame = \
            donnees.rename(columns={'Nobeyesdad_n': 'NObeyesdad_n',
                                    'Nobeyesdad_n2': 'NObeyesdad_n2'})
        donnees3: np.ndarray = \
            self.scaler.transform(donnees2)
        donnees4: pd.DataFrame = \
            pd.DataFrame(donnees3, columns=donnees2.columns)
        donnees5: pd.DataFrame =\
            donnees4.drop(labels=['NObeyesdad_n', 'NObeyesdad_n2'], axis=1)
    
        self.resultat: RandomForestClassifier \
            = self.modele_prediction.predict(donnees5)
        if int(self.resultat) == 0:
            return "non obèse"
        else:
            return "obèse"


if __name__ == "__main__":

    # Instantier
    jeu: rf_classification = rf_classification()

    # Extraire les attributs pour tester
    jeu.questionnaire
    jeu.modele_prediction
    jeu.scaler
    jeu.scaler.scale_

    # Importer de nouvelles données
    # Changer l'attribut questionnaire avec les nouvelles données
    jeu.importer_changer_donnees(nom_fichier_ext="questionnaire.xlsx",
                                 nom_feuille="donnees")

    # Vérifier l'attribut pour tester
    jeu.questionnaire

    # Faire une prédiction avec les nouvelles données
    print(jeu.faire_classification())
