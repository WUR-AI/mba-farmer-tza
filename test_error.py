import sys
import traceback
import numpy as np

try:
    import pandas as pd
    from main import get_estimator, load_and_preprocess_data
    from src.estimators.base import fit_estimator
    
    data = load_and_preprocess_data(data_path="/data/mba-tza", data_version="260116")
    data = data[data['N_rate'] > 0].dropna(subset=['Y_n', 'N_rate'])
    import geopandas as gpd
    points = gpd.GeoSeries(gpd.points_from_xy(data['lon'], data['lat'], crs='EPSG:32736'), index=data.index)
    
    current_data = data[['Y_n', 'N_rate', 'soil']].copy().dropna()
    EstimatorClass = get_estimator('DML_SOIL_RATE_SPLINE')
    
    model, _, _, _ = fit_estimator(
        estimator_name='DML_SOIL_RATE_SPLINE',
        EstimatorClass=EstimatorClass,
        current_data=current_data,
        raw_data=data,
        adjustment_set=['soil'],
        treatment_node='N',
        outcome_node='Y',
        outcome_var='Y_n',
        treatment_var='N_rate',
        random_seed=43,
        points=points
    )
    
    model.estimate_cate()
except Exception as e:
    traceback.print_exc()
