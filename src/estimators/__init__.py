from .base import BaseEstimator, CausalEstimate
from .models import (
    DML_SOIL, 
    DML_SOIL_RATE,
    DML_SOIL_RATE_SPLINE,
    DML_SOIL_RATE_POLY,
    DML_SOIL_TIME, 
    DML_SOIL_LEGUME,
    DML_SOIL_PFERT,
    OLS_SOIL,
    OLS_SOIL_RATE,
    OLS_SOIL_RATE_SPLINE,
    OLS_SOIL_RATE_POLY,
    OLS_SOIL_TIME,
    OLS_SOIL_LEGUME,
    OLS_SOIL_PFERT
)

def get_estimator(name: str):
    estimators = {
        'DML_SOIL': DML_SOIL,
        'DML_SOIL_RATE': DML_SOIL_RATE,
        'DML_SOIL_RATE_SPLINE': DML_SOIL_RATE_SPLINE,
        'DML_SOIL_RATE_POLY': DML_SOIL_RATE_POLY,
        'DML_SOIL_TIME': DML_SOIL_TIME,
        'DML_SOIL_LEGUME': DML_SOIL_LEGUME,
        'DML_SOIL_PFERT': DML_SOIL_PFERT,
        'OLS_SOIL': OLS_SOIL,
        'OLS_SOIL_RATE': OLS_SOIL_RATE,
        'OLS_SOIL_RATE_SPLINE': OLS_SOIL_RATE_SPLINE,
        'OLS_SOIL_RATE_POLY': OLS_SOIL_RATE_POLY,
        'OLS_SOIL_TIME': OLS_SOIL_TIME,
        'OLS_SOIL_LEGUME': OLS_SOIL_LEGUME,
        'OLS_SOIL_PFERT': OLS_SOIL_PFERT
    }
    if name not in estimators:
        raise ValueError(f"Estimator {name} is not implemented.")
    return estimators[name]
