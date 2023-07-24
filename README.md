# API pour le projet de détection/prédiction de faillite bancaire

## Description
L'API prend en input des données clients et renvoit un score de prédiction qui permet de détecter un risque de faillite bancaire. Elle est utilisée dans le Dashboard de prédiction de faillite bancaire.
Le score provient d'un algorithme de Machine Learning de type régression logistique, le modèle est ici importé. Pour plus de détails sur l'entraînement du modèle vous pouvez allez voir le notebook correspond dans mes repo github. (OC7-Notebooks).

## Découpage des dossiers
API/
│
├── .github\workflows
│ ├── main.yml #fichier qui permet de lancer le worflow de tests unitaires pour assurer un déploiment sans erreur
│
├── bin #fichiers qui permettent de renvoyer le score de prédictions
│ ├── data_FE_columns.sav
│ ├── ids_test.pkl
│ ├── imputer.sav
│ ├── logreg_best.sav
│ ├── scaler.sav
│ ├── shap_values_part1.npy
│ └──shap_values_part2.npy
│
├── .gitignore
├── api.py #code de l'API
├── Procfile #pour déploiement Heroku
├── README.md #fichier descriptif 
├── requirements.txt #liste des packages utilisés
├── runtime.txt #version de python utilisée
└──  test_api.py #tests unitaires appartement au WF GitHub Actions

## Packages utilisés
Cet API fonctionne sur python-3.11.4 et nécessite :
fastapi==0.100.0
matplotlib==3.7.2
numpy==1.24.4
pandas==2.0.3
Requests==2.31.0
scikit_learn==1.3.0
seaborn==0.12.2
shap==0.42.0
streamlit==1.24.1
uvicorn==0.22.0
gunicorn==20.1.0
pytest
httpx

## Utilisation
Pour la lancer en local il faut lancer la commande : 
uvicorn api:app --reload
