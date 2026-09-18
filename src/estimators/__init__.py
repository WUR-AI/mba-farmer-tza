from .base import BaseEstimator, CausalEstimate
from .models import (
    DML_SOIL, 
    DML_SOIL_RATE, 
    DML_SOIL_TIME, 
    DML_SOIL_LEGUME,
    OLS_SOIL,
    OLS_SOIL_RATE,
    OLS_SOIL_TIME,
    OLS_SOIL_LEGUME
)

def get_estimator(name: str):
    estimators = {
        'DML_SOIL': DML_SOIL,
        'DML_SOIL_RATE': DML_SOIL_RATE,
        'DML_SOIL_TIME': DML_SOIL_TIME,
        'DML_SOIL_LEGUME': DML_SOIL_LEGUME,
        'OLS_SOIL': OLS_SOIL,
        'OLS_SOIL_RATE': OLS_SOIL_RATE,
        'OLS_SOIL_TIME': OLS_SOIL_TIME,
        'OLS_SOIL_LEGUME': OLS_SOIL_LEGUME
    }
    if name not in estimators:
        raise ValueError(f"Estimator {name} is not implemented.")
    return estimators[name]
