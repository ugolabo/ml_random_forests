from rf_classification import rf_classification

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
