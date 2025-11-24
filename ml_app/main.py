import pandas as pd
import xgboost as xg
from pydantic import BaseModel, validator
from fastapi import HTTPException, FastAPI
import joblib
from pathlib import Path
import logging
from typing import Literal
from ml_app.app_utils.custom_transformers import (
    FeatureDropper,
    CategoricalAggregator,
    CategoricalConverter,
    NewFeatureEngineerDeploy,
    DaysToYearsTransformer,
    OutlierSkewTransformer,
    TableMerger,
    FeatureInteractionsBorutaDeploy,
    NumericDowncaster,
)


MODEL_PATH = "ml_app/xgb_model.joblib"


class HomeCreditDefaultRiskApp(BaseModel):
    OCCUPATION_TYPE: Literal[
        "Drivers",
        "Manual Labor",
        "Other",
        "Professional",
        "Sales & Admin",
        "Security",
        "Service",
    ]

    NAME_INCOME_TYPE: Literal[
        "Commercial associate", "Employed", "Pensioner", "State servant", "Not-Employed"
    ]

    ORGANIZATION_TYPE: Literal[
        "Business Entity",
        "Education",
        "Finance",
        "Other",
        "Industry",
        "Medicine",
        "Public/Services",
        "Self-employed",
        "Trade",
        "Transport",
        "Unknown",
    ]
    SK_ID_CURR: int
    AMT_CREDIT: int
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    OWN_CAR_AGE: int
    CNT_FAM_MEMBERS: int
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    FLOORSMAX_MEDI: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    DAYS_ID_PUBLISH: int
    FLAG_DOCUMENT_3: int

    @validator("SK_ID_CURR")
    def id_must_be_in_range(cls, v):
        if v < 1e5 or v > 4.563e5:
            raise ValueError("Value must be in a range between 100 000 and 456 300")
        return v

    @validator("FLAG_DOCUMENT_3")
    def flag_must_be_0_or_1(cls, v):
        if v not in (0, 1):
            raise ValueError("Value must be zero or one")
        return v

    @validator(
        "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY", "OWN_CAR_AGE", "CNT_FAM_MEMBERS"
    )
    def bmi_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("Value must non-negative")
        return v

    @validator("DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_ID_PUBLISH")
    def must_be_not_positive(cls, v):
        if v > 0:
            raise ValueError("Must be not positive")
        return v

    @validator("FLOORSMAX_MEDI", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
    def must_be_fraction(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Must be a fraction between 0 and 1")
        return v


class PredictionOut(BaseModel):
    default_proba: float

    @validator("default_proba", pre=True, always=True)
    def round_value(cls, v):
        return round(float(v), 3)


model = joblib.load(MODEL_PATH)

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Welcome to Home Credit Default Risk App", "model_version": 0.2}


@app.post("/v2/predict", response_model=PredictionOut)
def predict(input: HomeCreditDefaultRiskApp):
    try:
        input_data = input.dict()
        cust_df = pd.DataFrame([input_data])

        preds = model.predict_proba(cust_df)[0, 1]
        result = {"default_proba": preds}
        return result
    except Exception as e:
        logging.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))
