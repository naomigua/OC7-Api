import pytest
from fastapi.testclient import TestClient
import numpy as np
import json
from api import app, feindiv

client = TestClient(app)

import numpy as np


def test_make_prediction1():
    sample_data = {'SK_ID_CURR': 389171, 'TARGET': 0, 'NAME_CONTRACT_TYPE': 'Cash loans', 'CODE_GENDER': 'F', 'FLAG_OWN_CAR': 'N', 'FLAG_OWN_REALTY': 'Y', 'CNT_CHILDREN': 0, 'AMT_INCOME_TOTAL': 247500.0, 'AMT_CREDIT': 254700.0, 'AMT_ANNUITY': 24939.0, 'AMT_GOODS_PRICE': 225000.0, 'NAME_TYPE_SUITE': 'Unaccompanied', 'NAME_INCOME_TYPE': 'State servant', 'NAME_EDUCATION_TYPE': 'Secondary / secondary special', 'NAME_FAMILY_STATUS': 'Widow', 'NAME_HOUSING_TYPE':
'House / apartment', 'REGION_POPULATION_RELATIVE': 0.04622, 'DAYS_BIRTH': -19626, 'DAYS_EMPLOYED': -6982, 'DAYS_REGISTRATION': -11167.0, 'DAYS_ID_PUBLISH': -3158, 'OWN_CAR_AGE': np.nan, 'FLAG_MOBIL': 1, 'FLAG_EMP_PHONE': 1, 'FLAG_WORK_PHONE': 0, 'FLAG_CONT_MOBILE': 1, 'FLAG_PHONE': 0, 'FLAG_EMAIL': 0, 'OCCUPATION_TYPE': 'High skill tech staff', 'CNT_FAM_MEMBERS': 1.0, 'REGION_RATING_CLIENT': 1, 'REGION_RATING_CLIENT_W_CITY': 1, 'WEEKDAY_APPR_PROCESS_START': 'FRIDAY', 'HOUR_APPR_PROCESS_START': 14, 'REG_REGION_NOT_LIVE_REGION': 0, 'REG_REGION_NOT_WORK_REGION': 0, 'LIVE_REGION_NOT_WORK_REGION': 0, 'REG_CITY_NOT_LIVE_CITY': 0, 'REG_CITY_NOT_WORK_CITY': 0, 'LIVE_CITY_NOT_WORK_CITY': 0, 'ORGANIZATION_TYPE': 'Business Entity Type 3', 'EXT_SOURCE_1': np.nan, 'EXT_SOURCE_2': 0.6935206274920988, 'EXT_SOURCE_3': 0.6144143775673561, 'APARTMENTS_AVG': 0.132, 'BASEMENTAREA_AVG': 0.0645, 'YEARS_BEGINEXPLUATATION_AVG': 0.9846, 'YEARS_BUILD_AVG': np.nan, 'COMMONAREA_AVG': np.nan, 'ELEVATORS_AVG': 0.16, 'ENTRANCES_AVG': 0.069, 'FLOORSMAX_AVG': 0.625, 'FLOORSMIN_AVG': np.nan, 'LANDAREA_AVG': np.nan, 'LIVINGAPARTMENTS_AVG': np.nan, 'LIVINGAREA_AVG': 0.1628, 'NONLIVINGAPARTMENTS_AVG': np.nan, 'NONLIVINGAREA_AVG': 0.0022, 'APARTMENTS_MODE': 0.1345, 'BASEMENTAREA_MODE': 0.067, 'YEARS_BEGINEXPLUATATION_MODE': 0.9846, 'YEARS_BUILD_MODE': np.nan, 'COMMONAREA_MODE': np.nan, 'ELEVATORS_MODE': 0.1611, 'ENTRANCES_MODE': 0.069, 'FLOORSMAX_MODE': 0.625, 'FLOORSMIN_MODE': np.nan, 'LANDAREA_MODE': np.nan, 'LIVINGAPARTMENTS_MODE': np.nan, 'LIVINGAREA_MODE': 0.1696, 'NONLIVINGAPARTMENTS_MODE': np.nan, 'NONLIVINGAREA_MODE': 0.0023, 'APARTMENTS_MEDI': 0.1332, 'BASEMENTAREA_MEDI': 0.0645, 'YEARS_BEGINEXPLUATATION_MEDI': 0.9846, 'YEARS_BUILD_MEDI': np.nan, 'COMMONAREA_MEDI': np.nan, 'ELEVATORS_MEDI': 0.16, 'ENTRANCES_MEDI': 0.069,
    'FLOORSMAX_MEDI': 0.625, 'FLOORSMIN_MEDI': np.nan, 'LANDAREA_MEDI': np.nan, 'LIVINGAPARTMENTS_MEDI': np.nan, 'LIVINGAREA_MEDI': 0.1657, 'NONLIVINGAPARTMENTS_MEDI': np.nan, 'NONLIVINGAREA_MEDI': 0.0022, 'FONDKAPREMONT_MODE': np.nan, 'HOUSETYPE_MODE': np.nan, 'TOTALAREA_MODE': 0.1285, 'WALLSMATERIAL_MODE': 'Panel', 'EMERGENCYSTATE_MODE': 'No', 'OBS_30_CNT_SOCIAL_CIRCLE': 0.0, 'DEF_30_CNT_SOCIAL_CIRCLE': 0.0, 'OBS_60_CNT_SOCIAL_CIRCLE': 0.0, 'DEF_60_CNT_SOCIAL_CIRCLE': 0.0, 'DAYS_LAST_PHONE_CHANGE': -2000.0, 'FLAG_DOCUMENT_2': 0, 'FLAG_DOCUMENT_3': 1, 'FLAG_DOCUMENT_4': 0, 'FLAG_DOCUMENT_5': 0, 'FLAG_DOCUMENT_6': 0, 'FLAG_DOCUMENT_7': 0, 'FLAG_DOCUMENT_8': 0, 'FLAG_DOCUMENT_9': 0, 'FLAG_DOCUMENT_10': 0, 'FLAG_DOCUMENT_11': 0, 'FLAG_DOCUMENT_12': 0, 'FLAG_DOCUMENT_13': 0, 'FLAG_DOCUMENT_14': 0, 'FLAG_DOCUMENT_15': 0, 'FLAG_DOCUMENT_16': 0, 'FLAG_DOCUMENT_17': 0, 'FLAG_DOCUMENT_18': 0, 'FLAG_DOCUMENT_19': 0, 'FLAG_DOCUMENT_20': 0, 'FLAG_DOCUMENT_21': 0, 'AMT_REQ_CREDIT_BUREAU_HOUR': 0.0, 'AMT_REQ_CREDIT_BUREAU_DAY': 0.0, 'AMT_REQ_CREDIT_BUREAU_WEEK': 0.0, 'AMT_REQ_CREDIT_BUREAU_MON': 0.0, 'AMT_REQ_CREDIT_BUREAU_QRT': 0.0, 'AMT_REQ_CREDIT_BUREAU_YEAR': 0.0}

    
    response = client.post("/prediction", json=sample_data)
    
    assert response.status_code == 200
    response_dict = json.loads(response.json())
    assert "score" in response_dict
    assert response_dict["score"] == pytest.approx(0.1200169765913633)


def test_make_prediction2():
    sample_data = {'SK_ID_CURR': 384575, 'TARGET': 0, 'NAME_CONTRACT_TYPE': 'Cash loans', 'CODE_GENDER': 'M', 'FLAG_OWN_CAR': 'Y', 'FLAG_OWN_REALTY': 'N', 'CNT_CHILDREN': 2, 'AMT_INCOME_TOTAL': 207000.0, 'AMT_CREDIT': 465457.5, 'AMT_ANNUITY': 52641.0, 'AMT_GOODS_PRICE': 418500.0, 'NAME_TYPE_SUITE': 'Unaccompanied', 'NAME_INCOME_TYPE': 'Commercial associate', 'NAME_EDUCATION_TYPE': 'Secondary / secondary special', 'NAME_FAMILY_STATUS': 'Married', 'NAME_HOUSING_TYPE': 'House / apartment', 'REGION_POPULATION_RELATIVE': 0.00963, 'DAYS_BIRTH': -13297, 'DAYS_EMPLOYED': -762, 'DAYS_REGISTRATION': -637.0, 'DAYS_ID_PUBLISH': -4307, 'OWN_CAR_AGE': 19.0, 'FLAG_MOBIL': 1, 'FLAG_EMP_PHONE': 1, 'FLAG_WORK_PHONE': 0, 'FLAG_CONT_MOBILE': 1, 'FLAG_PHONE': 0, 'FLAG_EMAIL': 0, 'OCCUPATION_TYPE': 'Sales staff', 'CNT_FAM_MEMBERS': 4.0, 'REGION_RATING_CLIENT': 2, 'REGION_RATING_CLIENT_W_CITY': 2, 'WEEKDAY_APPR_PROCESS_START': 'THURSDAY', 'HOUR_APPR_PROCESS_START': 11, 'REG_REGION_NOT_LIVE_REGION': 0, 'REG_REGION_NOT_WORK_REGION': 0, 'LIVE_REGION_NOT_WORK_REGION': 0, 'REG_CITY_NOT_LIVE_CITY': 0, 'REG_CITY_NOT_WORK_CITY': 1, 'LIVE_CITY_NOT_WORK_CITY': 1, 'ORGANIZATION_TYPE': 'Business Entity Type 3', 'EXT_SOURCE_1': 0.6758780692916837, 'EXT_SOURCE_2': 0.6048943383603742, 'EXT_SOURCE_3': 0.0005272652387098, 'APARTMENTS_AVG': np.nan, 'BASEMENTAREA_AVG': np.nan, 'YEARS_BEGINEXPLUATATION_AVG': np.nan, 'YEARS_BUILD_AVG': np.nan, 'COMMONAREA_AVG': np.nan, 'ELEVATORS_AVG': np.nan, 'ENTRANCES_AVG': np.nan, 'FLOORSMAX_AVG': np.nan, 'FLOORSMIN_AVG': np.nan, 'LANDAREA_AVG': np.nan, 'LIVINGAPARTMENTS_AVG': np.nan, 'LIVINGAREA_AVG': np.nan, 'NONLIVINGAPARTMENTS_AVG': np.nan, 'NONLIVINGAREA_AVG': np.nan, 'APARTMENTS_MODE': np.nan, 'BASEMENTAREA_MODE': np.nan, 'YEARS_BEGINEXPLUATATION_MODE': np.nan, 'YEARS_BUILD_MODE': np.nan, 'COMMONAREA_MODE': np.nan, 'ELEVATORS_MODE': np.nan, 'ENTRANCES_MODE': np.nan, 'FLOORSMAX_MODE': np.nan, 'FLOORSMIN_MODE': np.nan, 'LANDAREA_MODE': np.nan, 'LIVINGAPARTMENTS_MODE': np.nan, 'LIVINGAREA_MODE': np.nan, 'NONLIVINGAPARTMENTS_MODE': np.nan, 'NONLIVINGAREA_MODE': np.nan, 'APARTMENTS_MEDI': np.nan, 'BASEMENTAREA_MEDI': np.nan, 'YEARS_BEGINEXPLUATATION_MEDI': np.nan, 'YEARS_BUILD_MEDI': np.nan, 'COMMONAREA_MEDI': np.nan, 'ELEVATORS_MEDI': np.nan, 'ENTRANCES_MEDI': np.nan, 'FLOORSMAX_MEDI': np.nan, 'FLOORSMIN_MEDI': np.nan, 'LANDAREA_MEDI': np.nan, 'LIVINGAPARTMENTS_MEDI': np.nan, 'LIVINGAREA_MEDI': np.nan, 'NONLIVINGAPARTMENTS_MEDI': np.nan, 'NONLIVINGAREA_MEDI': np.nan, 'FONDKAPREMONT_MODE': np.nan, 'HOUSETYPE_MODE': np.nan, 'TOTALAREA_MODE': np.nan, 'WALLSMATERIAL_MODE': np.nan, 'EMERGENCYSTATE_MODE': np.nan, 'OBS_30_CNT_SOCIAL_CIRCLE': 0.0, 'DEF_30_CNT_SOCIAL_CIRCLE': 0.0, 'OBS_60_CNT_SOCIAL_CIRCLE': 0.0, 'DEF_60_CNT_SOCIAL_CIRCLE': 0.0, 'DAYS_LAST_PHONE_CHANGE': -2.0, 'FLAG_DOCUMENT_2': 0, 'FLAG_DOCUMENT_3': 1, 'FLAG_DOCUMENT_4': 0, 'FLAG_DOCUMENT_5': 0, 'FLAG_DOCUMENT_6': 0, 'FLAG_DOCUMENT_7': 0, 'FLAG_DOCUMENT_8': 0, 'FLAG_DOCUMENT_9': 0, 'FLAG_DOCUMENT_10': 0, 'FLAG_DOCUMENT_11': 0, 'FLAG_DOCUMENT_12': 0, 'FLAG_DOCUMENT_13': 0, 'FLAG_DOCUMENT_14': 0, 'FLAG_DOCUMENT_15': 0, 'FLAG_DOCUMENT_16': 0, 'FLAG_DOCUMENT_17': 0, 'FLAG_DOCUMENT_18': 0, 'FLAG_DOCUMENT_19': 0, 'FLAG_DOCUMENT_20': 0, 'FLAG_DOCUMENT_21': 0, 'AMT_REQ_CREDIT_BUREAU_HOUR': 0.0, 'AMT_REQ_CREDIT_BUREAU_DAY': 0.0, 'AMT_REQ_CREDIT_BUREAU_WEEK': 0.0, 'AMT_REQ_CREDIT_BUREAU_MON': 1.0, 'AMT_REQ_CREDIT_BUREAU_QRT': 0.0, 'AMT_REQ_CREDIT_BUREAU_YEAR': 1.0}

    response = client.post("/prediction", json=sample_data)
    
    assert response.status_code == 200
    response_dict = json.loads(response.json())
    assert "score" in response_dict
    assert response_dict["score"] == pytest.approx(0.7314058911073659)

def test_make_prediction3():
    sample_data = {'SK_ID_CURR': 164360, 'TARGET': 0, 'NAME_CONTRACT_TYPE': 'Cash loans', 'CODE_GENDER': 'F', 'FLAG_OWN_CAR': 'Y', 'FLAG_OWN_REALTY': 'Y', 'CNT_CHILDREN': 0, 'AMT_INCOME_TOTAL': 135000.0, 'AMT_CREDIT': 1800000.0, 'AMT_ANNUITY': 47484.0, 'AMT_GOODS_PRICE': 1800000.0, 'NAME_TYPE_SUITE': 'Children', 'NAME_INCOME_TYPE': 'State servant', 'NAME_EDUCATION_TYPE': 'Secondary / secondary special', 'NAME_FAMILY_STATUS': 'Married', 'NAME_HOUSING_TYPE': 'House / apartment', 'REGION_POPULATION_RELATIVE': 0.028663, 'DAYS_BIRTH': -19913, 'DAYS_EMPLOYED': -1627, 'DAYS_REGISTRATION': -11594.0, 'DAYS_ID_PUBLISH': -3284, 'OWN_CAR_AGE': 3.0, 'FLAG_MOBIL': 1, 'FLAG_EMP_PHONE': 1, 'FLAG_WORK_PHONE': 0, 'FLAG_CONT_MOBILE': 1, 'FLAG_PHONE': 1, 'FLAG_EMAIL': 0, 'OCCUPATION_TYPE': 'Core staff', 'CNT_FAM_MEMBERS': 2.0, 'REGION_RATING_CLIENT': 2, 'REGION_RATING_CLIENT_W_CITY': 2, 'WEEKDAY_APPR_PROCESS_START': 'TUESDAY', 'HOUR_APPR_PROCESS_START': 8, 'REG_REGION_NOT_LIVE_REGION': 0, 'REG_REGION_NOT_WORK_REGION': 0, 'LIVE_REGION_NOT_WORK_REGION': 0, 'REG_CITY_NOT_LIVE_CITY': 0, 'REG_CITY_NOT_WORK_CITY': 1, 'LIVE_CITY_NOT_WORK_CITY': 1, 'ORGANIZATION_TYPE': 'Kindergarten', 'EXT_SOURCE_1': np.nan, 'EXT_SOURCE_2': 0.5351669305852306, 'EXT_SOURCE_3': 0.5388627065779676, 'APARTMENTS_AVG': np.nan, 'BASEMENTAREA_AVG': np.nan, 'YEARS_BEGINEXPLUATATION_AVG': np.nan, 'YEARS_BUILD_AVG': np.nan, 'COMMONAREA_AVG': np.nan, 'ELEVATORS_AVG': np.nan, 'ENTRANCES_AVG': np.nan, 'FLOORSMAX_AVG': np.nan, 'FLOORSMIN_AVG': np.nan, 'LANDAREA_AVG': np.nan, 'LIVINGAPARTMENTS_AVG': np.nan, 'LIVINGAREA_AVG': np.nan, 'NONLIVINGAPARTMENTS_AVG': np.nan, 'NONLIVINGAREA_AVG': np.nan, 'APARTMENTS_MODE': np.nan, 'BASEMENTAREA_MODE': np.nan, 'YEARS_BEGINEXPLUATATION_MODE': np.nan, 'YEARS_BUILD_MODE': np.nan, 'COMMONAREA_MODE': np.nan, 'ELEVATORS_MODE': np.nan, 'ENTRANCES_MODE': np.nan, 'FLOORSMAX_MODE': np.nan, 'FLOORSMIN_MODE': np.nan, 'LANDAREA_MODE': np.nan, 'LIVINGAPARTMENTS_MODE': np.nan, 'LIVINGAREA_MODE': np.nan, 'NONLIVINGAPARTMENTS_MODE': np.nan, 'NONLIVINGAREA_MODE': np.nan, 'APARTMENTS_MEDI': np.nan, 'BASEMENTAREA_MEDI': np.nan, 'YEARS_BEGINEXPLUATATION_MEDI': np.nan, 'YEARS_BUILD_MEDI': np.nan, 'COMMONAREA_MEDI': np.nan, 'ELEVATORS_MEDI': np.nan, 'ENTRANCES_MEDI': np.nan, 'FLOORSMAX_MEDI': np.nan, 'FLOORSMIN_MEDI': np.nan, 'LANDAREA_MEDI': np.nan, 'LIVINGAPARTMENTS_MEDI': np.nan, 'LIVINGAREA_MEDI': np.nan, 'NONLIVINGAPARTMENTS_MEDI': np.nan, 'NONLIVINGAREA_MEDI': np.nan, 'FONDKAPREMONT_MODE': np.nan, 'HOUSETYPE_MODE': np.nan, 'TOTALAREA_MODE': np.nan, 'WALLSMATERIAL_MODE': np.nan, 'EMERGENCYSTATE_MODE': np.nan, 'OBS_30_CNT_SOCIAL_CIRCLE': 1.0, 'DEF_30_CNT_SOCIAL_CIRCLE': 0.0, 'OBS_60_CNT_SOCIAL_CIRCLE': 1.0, 'DEF_60_CNT_SOCIAL_CIRCLE': 0.0, 'DAYS_LAST_PHONE_CHANGE': -313.0, 'FLAG_DOCUMENT_2': 0, 'FLAG_DOCUMENT_3': 1, 'FLAG_DOCUMENT_4': 0, 'FLAG_DOCUMENT_5': 0, 'FLAG_DOCUMENT_6': 0, 'FLAG_DOCUMENT_7': 0, 'FLAG_DOCUMENT_8': 0, 'FLAG_DOCUMENT_9': 0, 'FLAG_DOCUMENT_10': 0, 'FLAG_DOCUMENT_11': 0, 'FLAG_DOCUMENT_12': 0, 'FLAG_DOCUMENT_13': 0, 'FLAG_DOCUMENT_14': 0, 'FLAG_DOCUMENT_15': 0, 'FLAG_DOCUMENT_16': 0, 'FLAG_DOCUMENT_17': 0, 'FLAG_DOCUMENT_18': 0, 'FLAG_DOCUMENT_19': 0, 'FLAG_DOCUMENT_20': 0, 'FLAG_DOCUMENT_21': 0, 'AMT_REQ_CREDIT_BUREAU_HOUR': 0.0, 'AMT_REQ_CREDIT_BUREAU_DAY': 0.0, 'AMT_REQ_CREDIT_BUREAU_WEEK': 0.0, 'AMT_REQ_CREDIT_BUREAU_MON': 0.0, 'AMT_REQ_CREDIT_BUREAU_QRT': 2.0, 'AMT_REQ_CREDIT_BUREAU_YEAR': 3.0}

    response = client.post("/prediction", json=sample_data)
    
    assert response.status_code == 200
    response_dict = json.loads(response.json())
    assert "score" in response_dict
    assert response_dict["score"] == pytest.approx(0.21390706563081222)
    
    assert response.status_code == 200
    response_dict = json.loads(response.json())
    assert "score" in response_dict
    assert response_dict["score"] == pytest.approx(0.21390706563081222)
