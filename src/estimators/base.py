import numpy as np
import scipy.stats as stats
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
        random_state=random_seed,
        cv=5
    )
    
    # We pass the default RF models if not already fitted
    if getattr(model, 'model_y', None) is None:
        from sklearn.ensemble import RandomForestRegressor
        model.model_y = RandomForestRegressor(random_state=random_seed, n_jobs=-1, max_depth=3)
        model.model_t = RandomForestRegressor(random_state=random_seed, n_jobs=-1, max_depth=3)
        model.is_first_pass = True
    else:
        model.is_first_pass = False
        
    groups = raw_data.loc[current_data.index, 'ADM3_PCODE'].values if 'ADM3_PCODE' in raw_data.columns else None

    current_transformed_data_subset = raw_data.loc[current_data.index].copy()
    
    kwargs = {}
    if 'SOIL_RATE' in estimator_name and points is not None:
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


def build_treatment_featurizer(T_absolute, points_geom, featurizer_type='SPLINE'):
    if featurizer_type == 'SPLINE':
        transformer = SplineTransformer(n_knots=5, degree=3, include_bias=False)
        # from sklearn.preprocessing import PolynomialFeatures
        # transformer = PolynomialFeatures(degree=2, include_bias=False)
    elif featurizer_type == 'POLY':
        from sklearn.preprocessing import PolynomialFeatures
        transformer = PolynomialFeatures(degree=2, include_bias=False)
    else:
        raise ValueError(f"Unknown featurizer_type: {featurizer_type}")
        
    return transformer, transformer


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

from src.estimators.helpers import _aggregate_cates_to_ate


class DoseResponseMixin:
    def estimate_ate(self):
        cates = self.estimate_cate()
        # Filter out the "High Non-Return" points for ATE aggregation
        base_cates = {k: v for k, v in cates.items() if 'High Non-Return' not in k}
        
        if hasattr(self, 'X_counts'):
            weights = {k: self.X_counts[k] for k in base_cates.keys() if k in self.X_counts}
        elif hasattr(self, 'x_names'):
            weights = {name: self.X[:, i].sum() for i, name in enumerate(self.x_names) if name in base_cates}
        else:
            weights = None
            
        return _aggregate_cates_to_ate(base_cates, self, weights=weights)

    def _calculate_continuous_inference(self, T0, T1, is_sandy, param, param_cov_matrix, scale_divide=False):
        T1_arr = np.atleast_1d(T1).reshape(-1, 1)
        T0_arr = np.atleast_1d(T0).reshape(-1, 1)
        if T0_arr.shape[0] == 1 and T1_arr.shape[0] > 1:
            T0_arr = np.full_like(T1_arr, T0_arr[0, 0])
            
        t_change = self.treatment_transformer.transform(T1_arr) - self.treatment_transformer.transform(T0_arr)
        
        if scale_divide:
            scale = T1_arr - T0_arr
            scale = np.where(scale == 0, 1.0, scale)
            t_change = t_change / scale
            
        # Check if we are in DML or OLS
        if hasattr(self, 'cate_column_map'):
            # OLS_SOIL_RATE
            x_final_model = np.zeros((T1_arr.shape[0], len(param)))
            for i in range(t_change.shape[1]):
                col_name = f'feat_{i}_sandy' if is_sandy else f'feat_{i}_nosandy'
                if col_name in param.index:
                    idx = param.index.get_loc(col_name)
                    x_final_model[:, idx] = t_change[:, i]
            
            x_mean = x_final_model.mean(axis=0)
            ate = x_mean @ param
            ate_se = np.sqrt(x_mean @ param_cov_matrix @ x_mean)
        else:
            # DML_SOIL_RATE
            X_for_inference = np.array([[0, 1]]) if is_sandy else np.array([[1, 0]])
            x_final_model = []
            for j in range(t_change.shape[1]):
                x_final_model.append(np.repeat(X_for_inference, T1_arr.shape[0], axis=0) * t_change[:, j:j+1])
            x_final_model = np.hstack(x_final_model)
            
            x_mean = x_final_model.mean(axis=0).reshape(1, -1)
            ate = (x_mean @ param.reshape(-1, 1))[0, 0]
            ate_se = np.sqrt(x_mean @ param_cov_matrix @ x_mean.T)[0, 0]
            
        return ate, ate_se

    def estimate_dose_response(self):
        """
        Calculates marginal response curve for a change from baseline_t to t
        for t spanning the 5th to 95th percentiles.
        """
        if hasattr(self, 'cate_column_map'):
            param_cov_matrix = self.est.cov_params()
            param = self.est.params
        else:
            param_cov_matrix = self.est._ortho_learner_model_final._model_final._model._param_var
            param = self.est._ortho_learner_model_final._model_final._model._param
            
        baseline = getattr(self, 'baseline_t', 'q05')
        if baseline == 'q05':
            baseline = np.percentile(self.T_absolute, 5)
        if baseline == 'mean':
            baseline = np.mean(self.T_absolute)
            
        q5, q95 = np.percentile(self.T_absolute, [5, 95])
        t_grid = np.arange(np.ceil(q5), np.floor(q95) + 1, 1.0)
        
        results = []
        for t in t_grid:
            ate_sandy, se_sandy = self._calculate_continuous_inference(baseline, t, True, param, param_cov_matrix)
            ate_nonsandy, se_nonsandy = self._calculate_continuous_inference(baseline, t, False, param, param_cov_matrix)
            
            results.append({
                'Soil type': 'Sandy',
                'T': t,
                'Mean Lift': ate_sandy,
                'Standard Error': se_sandy,
                'Lower CI': ate_sandy - 1.96 * se_sandy,
                'Upper CI': ate_sandy + 1.96 * se_sandy
            })
            results.append({
                'Soil type': 'Non-sandy',
                'T': t,
                'Mean Lift': ate_nonsandy,
                'Standard Error': se_nonsandy,
                'Lower CI': ate_nonsandy - 1.96 * se_nonsandy,
                'Upper CI': ate_nonsandy + 1.96 * se_nonsandy
            })
            
        df = pd.DataFrame(results).set_index(['Soil type', 'T']).sort_index()
        return df

    def estimate_average_marginal_effect(self, treatment_name=None):
        if treatment_name is None:
            treatment_name = self.treatment.name
        N = 10 if 'N' in treatment_name else 5
        
        if hasattr(self, 'cate_column_map'):
            param_cov_matrix = self.est.cov_params()
            param = self.est.params
        else:
            param_cov_matrix = self.est._ortho_learner_model_final._model_final._model._param_var
            param = self.est._ortho_learner_model_final._model_final._model._param
            
        q5, q95 = np.percentile(self.T_absolute, [5, 95])
        start_t = np.ceil(q5 / N) * N
        end_t = np.floor(q95 / N) * N
        
        t_grid = np.arange(start_t, end_t, N)
        
        results = []
        for t in t_grid:
            ate_sandy, se_sandy = self._calculate_continuous_inference(t, t + N, True, param, param_cov_matrix)
            ate_nonsandy, se_nonsandy = self._calculate_continuous_inference(t, t + N, False, param, param_cov_matrix)
            
            ate_sandy /= N
            se_sandy /= N
            ate_nonsandy /= N
            se_nonsandy /= N
            
            results.append({
                'Soil type': 'Sandy',
                'T': t,
                'Mean Lift': ate_sandy,
                'Standard Error': se_sandy,
                'Lower CI': ate_sandy - 1.96 * se_sandy,
                'Upper CI': ate_sandy + 1.96 * se_sandy
            })
            results.append({
                'Soil type': 'Non-sandy',
                'T': t,
                'Mean Lift': ate_nonsandy,
                'Standard Error': se_nonsandy,
                'Lower CI': ate_nonsandy - 1.96 * se_nonsandy,
                'Upper CI': ate_nonsandy + 1.96 * se_nonsandy
            })
            
        df = pd.DataFrame(results).set_index(['Soil type', 'T']).sort_index()
        return df

    def estimate_cate(self):
        baseline = getattr(self, 'baseline_t', 'q05')
        if baseline == 'q05':
            baseline = np.percentile(self.T_absolute, 5)
        if baseline == 'mean':
            baseline = np.mean(self.T_absolute)
            
        t_target_sandy = self.T_absolute
        t_target_nonsandy = self.T_absolute
        
        if hasattr(self, 'x_names'):
            x_names_list = list(self.x_names)
            if 'Sandy' in x_names_list and 'Non-sandy' in x_names_list:
                sandy_idx = x_names_list.index('Sandy')
                nonsandy_idx = x_names_list.index('Non-sandy')
                t_target_sandy = self.T_absolute[self.X[:, sandy_idx] == 1]
                t_target_nonsandy = self.T_absolute[self.X[:, nonsandy_idx] == 1]
        
        if hasattr(self, 'cate_column_map'):
            param_cov_matrix = self.est.cov_params()
            param = self.est.params
        else:
            param_cov_matrix = self.est._ortho_learner_model_final._model_final._model._param_var
            param = self.est._ortho_learner_model_final._model_final._model._param
            
        ate_sandy, se_sandy = self._calculate_continuous_inference(baseline, t_target_sandy, True, param, param_cov_matrix, scale_divide=True)
        ate_nonsandy, se_nonsandy = self._calculate_continuous_inference(baseline, t_target_nonsandy, False, param, param_cov_matrix, scale_divide=True)
        
        def _build_estimate(ate, se, group_name):
            ci_mean_lower = ate - 1.96 * se
            ci_mean_upper = ate + 1.96 * se
            if se != 0:
                t_stat = ate / se
                p_value = 2 * stats.norm.sf(np.abs(t_stat))
            else:
                p_value = np.nan
            
            count = None
            if hasattr(self, 'X_counts') and group_name in self.X_counts:
                count = int(self.X_counts[group_name])
            elif hasattr(self, 'x_names') and group_name in self.x_names:
                idx = list(self.x_names).index(group_name)
                count = int(self.X[:, idx].sum())
                
            return CausalEstimate(
                value=ate, std_error=se, p_value=p_value, 
                ci_lower=ci_mean_lower, ci_upper=ci_mean_upper, estimator_instance=self, count=count
            )
            
        cates = {
            'Sandy': _build_estimate(ate_sandy, se_sandy, 'Sandy'),
            'Non-sandy': _build_estimate(ate_nonsandy, se_nonsandy, 'Non-sandy')
        }
        
        # High Non-Return Point via Parametric Bootstrap
        try:
            rng = np.random.default_rng(42)
            n_sims = 10000
            param_vals = param.values if hasattr(param, 'values') else np.array(param)
            cov_vals = param_cov_matrix.values if hasattr(param_cov_matrix, 'values') else np.array(param_cov_matrix)
            
            # Ensure it is symmetric positive semi-definite
            cov_vals = (cov_vals + cov_vals.T) / 2
            
            simulated_params = rng.multivariate_normal(param_vals, cov_vals, size=n_sims)
            
            q95 = np.percentile(self.T_absolute, 95)
            q50 = np.percentile(self.T_absolute, 50) # I expect the non-return rate to be on the upper end of the curve
            # Evaluate up to 150% of the 95th percentile
            t_grid = np.arange(q50, q95, 0.5)
            
            # Prepare X_diff arrays for all t in t_grid for both Sandy and Non-Sandy
            X_diff_sandy_list = []
            X_diff_nonsandy_list = []
            
            for t in t_grid:
                t_change = self.treatment_transformer.transform([[t + 0.5]]) - self.treatment_transformer.transform([[t - 0.5]])
                
                if hasattr(self, 'cate_column_map'):
                    x_final_sandy = np.zeros(len(param))
                    x_final_nonsandy = np.zeros(len(param))
                    for i in range(t_change.shape[1]):
                        col_sandy = f'feat_{i}_sandy'
                        col_nonsandy = f'feat_{i}_nosandy'
                        if col_sandy in param.index:
                            x_final_sandy[param.index.get_loc(col_sandy)] = t_change[0, i]
                        if col_nonsandy in param.index:
                            x_final_nonsandy[param.index.get_loc(col_nonsandy)] = t_change[0, i]
                    X_diff_sandy_list.append(x_final_sandy)
                    X_diff_nonsandy_list.append(x_final_nonsandy)
                else:
                    X_sandy = np.array([[0, 1]])
                    X_nonsandy = np.array([[1, 0]])
                    x_final_sandy = []
                    x_final_nonsandy = []
                    for j in range(t_change.shape[1]):
                        x_final_sandy.append(X_sandy * t_change[0, j])
                        x_final_nonsandy.append(X_nonsandy * t_change[0, j])
                    X_diff_sandy_list.append(np.hstack(x_final_sandy).mean(axis=0))
                    X_diff_nonsandy_list.append(np.hstack(x_final_nonsandy).mean(axis=0))
            
            X_diff_sandy = np.array(X_diff_sandy_list) # shape: (len(t_grid), num_params)
            X_diff_nonsandy = np.array(X_diff_nonsandy_list)
            
            for soil, X_diff in [('Sandy', X_diff_sandy), ('Non-sandy', X_diff_nonsandy)]:
                # Compute marginal effects for all simulations and all t
                # simulated_params shape: (n_sims, num_params)
                # X_diff shape: (len(t_grid), num_params)
                # Output shape: (n_sims, len(t_grid))
                ME_sims = np.dot(simulated_params, X_diff.T)
                
                high_non_return_points = []
                for i in range(n_sims):
                    # Find first t where ME <= 0
                    negative_indices = np.where(ME_sims[i] <= 0)[0]
                    if len(negative_indices) > 0:
                        high_non_return_points.append(t_grid[negative_indices[0]])
                    else:
                        # Never crosses 0 in the given grid
                        high_non_return_points.append(np.nan)
                        
                high_non_return_points = np.array(high_non_return_points)
                valid_points = high_non_return_points[~np.isnan(high_non_return_points)]
                
                if len(valid_points) > 0:
                    point_est = np.median(valid_points)
                    ci_lower = np.percentile(valid_points, 2.5)
                    ci_upper = np.percentile(valid_points, 97.5)
                    std_error = np.std(valid_points)
                else:
                    point_est = np.nan
                    ci_lower = np.nan
                    ci_upper = np.nan
                    std_error = np.nan
                    
                count = None
                if hasattr(self, 'X_counts') and soil in self.X_counts:
                    count = int(self.X_counts[soil])
                elif hasattr(self, 'x_names') and soil in self.x_names:
                    idx = list(self.x_names).index(soil)
                    count = int(self.X[:, idx].sum())
                    
                cates[f'{soil}: High Non-Return'] = CausalEstimate(
                    value=point_est, std_error=std_error, p_value=np.nan, 
                    ci_lower=ci_lower, ci_upper=ci_upper, estimator_instance=self,
                    count=count
                )
                
        except Exception as e:
            print(f"Failed to estimate High Non-Return CATE parameters: {e}")
            import traceback
            traceback.print_exc()
            
        return cates
# ==========================================
