

import pandas as pd
import pickle
import numpy as np
import warnings
import uvicorn
import json
from fastapi import FastAPI
import shap
import os
from pathlib import Path

#C:\Users\Me\Documents\1- DATA SCIENCE\Projet7\>uvicorn api:app --reload
#run dans le terminal: uvicorn api:app --reload
#env7
#test 
warnings.filterwarnings("ignore")
data_FE_columns=pickle.load(open('bin/data_FE_columns.sav','rb'))
logreg_best=pickle.load(open("bin/logreg_best.sav", 'rb'))
imputer=pickle.load(open("bin/imputer.sav", 'rb'))
scaler=pickle.load(open("bin/scaler.sav", 'rb'))
shap_values_part1 = np.load('bin/shap_values_part1.npy')
shap_values_part2 = np.load('bin/shap_values_part2.npy')
shap_values = np.concatenate((shap_values_part1, shap_values_part2), axis=0)
ids_test=pickle.load(open("bin/ids_test.pkl", 'rb'))

def feindiv(data):
    #transform data_dict into pd.series
    data = pd.Series(data)
    id_client = data['SK_ID_CURR']
    id_shap=ids_test.loc[ids_test == id_client].index
    #label encoding
    data_col_le=['NAME_CONTRACT_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','EMERGENCYSTATE_MODE']
    mapping = {'NAME_CONTRACT_TYPE': {'Cash loans': 0,'Revolving loans': 1},
    'FLAG_OWN_CAR': {'N': 0,'Y': 1},'FLAG_OWN_REALTY': {'N': 0,'Y': 1},'EMERGENCYSTATE_MODE': {'No': 0,'Yes': 1,'nan': 2}}
    for col in data.index:
        if col in data_col_le:
            encoding_map = mapping[col]
            data[col] = encoding_map.get(data[col],data[col])
    # one-hot encoding of categorical variables
    categorical_cols=['CODE_GENDER', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE'] 
    data = pd.DataFrame(data).transpose()  # Conversion du tableau numpy en DataFrame
    data = pd.get_dummies(data,columns=categorical_cols)
    anom = data[data['DAYS_EMPLOYED'] == 365243]
    non_anom = data[data['DAYS_EMPLOYED'] != 365243]
    # Create an anomalous flag column
    data['DAYS_EMPLOYED_ANOM'] = data["DAYS_EMPLOYED"] == 365243
    # Replace the anomalous values with nan
    data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    #abs data birth
    data['DAYS_BIRTH'] = abs(data['DAYS_BIRTH'])
    #data domain features
    data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
    data['DAYS_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    data.drop(['TARGET','SK_ID_CURR'], axis=1,inplace=True)
    # Get missing columns in the training test
    missing_cols = set( data_FE_columns ) - set( data.columns )
    missing_cols=list(missing_cols)
    missing_data = pd.DataFrame(0, index=data.index, columns=missing_cols)
    data = pd.concat([data, missing_data], axis=1)
    # Ensure the order of column in the test set is in the same order than in train set
    data = data[data_FE_columns]
    #data=data.squeeze()
    data = np.reshape(data.values, (1, -1))
    data = imputer.transform(data)
    data = scaler.transform(data)
    #data=data[0]
    prediction_score = logreg_best.predict_proba(data)[:,1][0]
    prediction_binary = int((prediction_score>0.529).astype(int))
    shap_values_selected=shap_values[id_shap]
    X_shap=pd.DataFrame(data,columns=data_FE_columns)
    expected_value = -0.39871664006890845
    return prediction_score, prediction_binary, X_shap, shap_values_selected, expected_value

app = FastAPI()

@app.post("/prediction")
def make_prediction(data: dict):
    prediction_score, prediction_binary, X_shap, shap_values_selected, expected_value = feindiv(data)
    # Retournez la prédiction et d'autres résultats souhaités
    results = {"prediction": prediction_binary, "score": prediction_score, "X_shap": X_shap.to_json(), "shap_values_selected": shap_values_selected.tolist(), "expected_value": expected_value}
    results=json.dumps(results)
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)