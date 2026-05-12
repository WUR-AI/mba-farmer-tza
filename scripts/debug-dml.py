import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
plt.style.use('ggplot')
import pickle

import geopandas as gpd
from shapely.geometry import Point

from CAgsalML.causal import  dml_sensitivity_analysis
from CAgsalML.tuning import GPOptimizer
from CAgsalML.transformers import SpatialTransform

from econml.dml import LinearDML
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import SplineTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut


RANDOM_SEED = 43
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)

with open('debug-data/rate-response-dml-package', 'rb') as f:
    model_q, model_treatment, Y, T, W, X_sand, \
        points, index_zone_subset, T_absolute, Y_absolute = pickle.load(f)
    
T_spatial_avg = T_absolute - T
Y_spatial_avg = Y_absolute - Y
groups_cv = np.random.choice(np.arange(5), size=len(T), replace=True)

cubic_spline_transformer = SplineTransformer(
    degree=3, n_knots=4, include_bias=True,
    knots='uniform',
)

T_trans = cubic_spline_transformer.fit_transform(T_absolute.reshape(-1, 1))
T_spline_spatial_avg = np.zeros_like(T_trans)
for i in range(T_trans.ndim):
    t_col = T_trans[:, i]
    spatial_transformer = SpatialTransform(
        points.geometry, maxlag=50e3, esitmator='cressie',
        model='matern', n_lags=50, use_nugget=True
    )
    spatial_transformer.fit(t_col)
    t_trans_col = spatial_transformer.transform_demean() 
    T_spline_spatial_avg[:, i] = t_col - np.where(np.isnan(t_trans_col), 0, t_trans_col)

def spatial_anomaly_transform(y):
    return y - T_spline_spatial_avg

spatial_anomaly_transformer = FunctionTransformer(spatial_anomaly_transform)
treatment_featurizer = Pipeline([
    ('spline', cubic_spline_transformer),
    ('spatial anomaly', spatial_anomaly_transformer)
])

lineardml_amount_estimate = LinearDML(
    model_y=model_q, 
    model_t=model_treatment, 
    treatment_featurizer=treatment_featurizer,
    discrete_treatment=False,
    cv=LeaveOneGroupOut(),
    random_state=RANDOM_SEED,
    fit_cate_intercept=False, 
)
lineardml_amount_estimate = lineardml_amount_estimate.fit(
    Y=Y, T=T_absolute.reshape(-1, 1), W=W, X=X_sand, groups=groups_cv,
    inference='statsmodels', cache_values=True,
)
print(lineardml_amount_estimate.ate_inference(X_sand))

# def spatial_anomaly_transform(y):
#     y = np.array(y)
#     if y.ndim > 1:
#         y_trans = np.zeros_like(y)
#         for i in range(y.ndim):
#             y_col = y[:, i]
#             spatial_transformer = SpatialTransform(
#                 points.geometry, maxlag=50e3, esitmator='cressie',
#                 model='matern', n_lags=50, use_nugget=True
#             )
#             spatial_transformer.fit(y_col)
#             y_col_trans = spatial_transformer.transform_demean()
#             y_trans[:, i] = np.where(np.isnan(y_col_trans), 0, y_col_trans)
#     else:
#         spatial_transformer = SpatialTransform(
#             points.geometry, maxlag=50e3, esitmator='cressie',
#             model='matern', n_lags=50, use_nugget=True
#         )
#         spatial_transformer.fit(y)
#         y_trans = spatial_transformer.transform_demean()
#         y_trans = np.where(np.isnan(y_trans), 0, y_trans)
#     return y_trans