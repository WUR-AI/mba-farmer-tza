import numpy as np
import pandas as pd
from typing import Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn.preprocessing import SplineTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
from CAgsalML.transformers import SpatialTransform
import json

RANDOM_SEED = 43

def fit_estimator(
    estimator_name,
    EstimatorClass,
    current_data,
    raw_data,
    adjustment_set,
    treatment_node,
    outcome_node,
    outcome_var,
    treatment_var,
    random_seed=43,
    shared_model_y=None,
    shared_model_t=None,
    points=None
):
    """
    Unified helper to initialize and fit an estimator with all necessary boilerplate.
    """
    model = EstimatorClass(
        outcome=current_data[outcome_var], 
        treatment=current_data[treatment_var], 
        controls=None, # Controls are resolved in fit
        model_y=shared_model_y,
        model_t=shared_model_t,
        random_state=random_seed
    )
    
    # We pass the default RF models if not already fitted
    if getattr(model, 'model_y', None) is None:
        from sklearn.ensemble import RandomForestRegressor
        model.model_y = RandomForestRegressor(random_state=random_seed, n_jobs=-1, max_depth=3)
        model.model_t = RandomForestRegressor(random_state=random_seed, n_jobs=-1, max_depth=3)
        model.is_first_pass = True
    else:
        model.is_first_pass = False
        
    groups = raw_data.loc[current_data.index, 'ADM2_PCODE'].values if 'ADM2_PCODE' in raw_data.columns else None

    current_transformed_data_subset = raw_data.loc[current_data.index].copy()
    
    kwargs = {}
    if estimator_name in ['DML_SOIL_RATE', 'OLS_SOIL_RATE'] and points is not None:
        kwargs['points'] = points.loc[current_data.index]

    current_data, shared_model_y, shared_model_t = model.fit(
        data=current_data, 
        transformed_data=current_transformed_data_subset, 
        adjustment_set=adjustment_set, 
        treatment_name=treatment_node, 
        outcome_name=outcome_node,
        groups=groups,
        **kwargs
    )
    
    return model, current_data, shared_model_y, shared_model_t

@dataclass
class CausalEstimate:
    """
    Dataclass to hold the results of a causal estimation.
    """
    value: float
    std_error: float
    p_value: float
    ci_lower: float
    ci_upper: float
    estimator_instance: Any  # Reference to the estimator that generated this
    count: Optional[int] = None

class SpatialAnomalyTransformerCallable:
    def __init__(self, T_spline_spatial_avg):
        self.T_spline_spatial_avg = T_spline_spatial_avg
        
    def __call__(self, t):
        return t - self.T_spline_spatial_avg

def get_feature_lists(adjustment_set, transformed_data, treatment_name, outcome_name, x_cols):
    """
    Returns the feature list (all potential), the controls (W), and modifiers (X).
    """
    with open('config/node_variable_map.json', 'r') as f:
        columns_dict = json.load(f)
    
    bad_controls = [
        i for c in sorted(set(columns_dict.keys()) - set(adjustment_set)) 
        for i in columns_dict.get(c, [])
    ]
    
    all_potential_features = [i for v in columns_dict.values() for i in v]
    features = list(sorted(
        list(set(all_potential_features) - set(bad_controls) - {outcome_name, treatment_name})
    ))
    
    # Ensure only columns that exist in the transformed data are used
    features = [f for f in features if f in transformed_data.columns]
    
    # Add explicit x_cols
    for x in x_cols:
        if x in transformed_data.columns and x not in features:
            features.append(x)

            
    # For timing features, they usually start with 'timing_' and 'pfertilizer_'.
    # We should also ensure they are NOT bad controls.
    
    w_names = list(sorted(set(features) - set(x_cols)))
    
    # Let's just return features, w_names, x_names
    return features, w_names, x_cols


def build_treatment_featurizer(T_absolute, points_geom):
    cubic_spline_transformer = SplineTransformer(n_knots=5, degree=3, include_bias=False)
    T_trans = cubic_spline_transformer.fit_transform(T_absolute.reshape(-1, 1))
    
    T_spline_spatial_avg = np.zeros_like(T_trans)
    for i in range(T_trans.ndim):
        t_col = T_trans[:, i]
        spatial_transformer = SpatialTransform(
            points_geom, maxlag=50e3, esitmator='cressie',
            model='matern', n_lags=50, use_nugget=True
        )
        spatial_transformer.fit(t_col)
        t_trans_col = spatial_transformer.transform_demean()
        T_spline_spatial_avg[:, i] = t_col - np.where(np.isnan(t_trans_col), 0, t_trans_col)
        
    spatial_anomaly_transform = SpatialAnomalyTransformerCallable(T_spline_spatial_avg)
    spatial_anomaly_transformer = FunctionTransformer(spatial_anomaly_transform)
    
    treatment_featurizer = Pipeline([
        ('spline', cubic_spline_transformer),
        ('spatial anomaly', spatial_anomaly_transformer)
    ])
    
    return treatment_featurizer, cubic_spline_transformer


class BaseEstimator(ABC):
    """
    Abstract Base Class for Causal Estimators.
    """
    
    def __init__(
        self, 
        outcome: Any, 
        treatment: Any, 
        controls: Any, 
        conditional_nodes: Optional[Any] = None, 
        splits: Optional[Any] = None, 
        **kwargs
    ):
        self.outcome = outcome
        self.treatment = treatment
        self.controls = controls
        self.conditional_nodes = conditional_nodes
        self.splits = splits
        self.kwargs = kwargs
    
    @abstractmethod
    def fit(self):
        """
        Fit the causal estimator on the provided data.
        """
        pass
    
    @abstractmethod
    def estimate_ate(self) -> CausalEstimate:
        """
        Estimate the Average Treatment Effect (ATE).
        """
        pass
        
    def estimate_cate(self) -> Any:
        """
        Estimate the Conditional Average Treatment Effect (CATE).
        """
        raise NotImplementedError("CATE estimation is not implemented for this estimator.")
        
    def estimate_dose_response(self, X_for_inference, T0, T1) -> dict:
        """
        Estimate a dose response or marginal response if the model uses continuous treatment curves.
        """
        raise NotImplementedError("Dose response is not implemented for this estimator.")
