import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.estimators.base import BaseEstimator, CausalEstimate, get_feature_lists, build_treatment_featurizer, RANDOM_SEED, DoseResponseMixin
from src.estimators.helpers import _aggregate_cates_to_ate, _build_time_matrix, _build_pfert_matrix, _build_legume_matrix

class BaseOLSEstimator(BaseEstimator):
    heterogeneity_builder = None
    heterogeneity_drop_cols = []

    def __init__(self, outcome, treatment, controls, splits=None, **kwargs):
        super().__init__(outcome, treatment, controls, splits, **kwargs)
        self.est = None
        self.cate_column_map = {}
        self.X_counts = {}

    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None):
        if self.heterogeneity_builder:
            X_het, x_names_het = self.heterogeneity_builder(transformed_data, treatment_name=treatment_name)
        else:
            X_het, x_names_het = None, []

        cols_to_drop = ['soil_sandy'] + self.heterogeneity_drop_cols
        if self.heterogeneity_builder:
            cols_to_drop += x_names_het

        features, _, _ = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, cols_to_drop
        )
        
        ols_data = transformed_data[features].copy()
        ols_data[outcome_name] = self.outcome
        
        is_sandy = transformed_data['soil_sandy'].astype(bool)
        
        self.cate_column_map = {}
        self.X_counts = {}

        if not self.heterogeneity_builder:
            self.X_counts = {
                'Sandy': is_sandy.sum(),
                'Non-sandy': (~is_sandy).sum()
            }
            ols_data['N_sandy'] = self.treatment * is_sandy
            ols_data['N_nosandy'] = self.treatment * (~is_sandy)
            self.cate_column_map = {'Sandy': 'N_sandy', 'Non-sandy': 'N_nosandy'}
        else:
            for c in x_names_het:
                c_val = transformed_data[c]
                
                self.X_counts[f'N_{c}_Sandy'] = (c_val * is_sandy).sum()
                self.X_counts[f'N_{c}_Non-sandy'] = (c_val * ~is_sandy).sum()
                
                ols_data[f'N_{c}_Sandy'] = c_val * self.treatment * is_sandy
                ols_data[f'N_{c}_Non-sandy'] = c_val * self.treatment * ~is_sandy
                
                self.cate_column_map[f'N_{c}_Sandy'] = f'N_{c}_Sandy'
                self.cate_column_map[f'N_{c}_Non-sandy'] = f'N_{c}_Non-sandy'

        for c in cols_to_drop:
            if c in ols_data:
                ols_data = ols_data.drop(columns=[c])
                
        ols_data = ols_data.astype(float)
        return data, *self._fit_ols(ols_data, outcome_name)
        
    def _fit_ols(self, ols_data, outcome_name):
        self.est = sm.OLS(
            endog=ols_data[outcome_name],
            exog=sm.add_constant(ols_data.drop(columns=[outcome_name]))
        ).fit()
        return self.kwargs.get('model_y'), self.kwargs.get('model_t')
        
    def estimate_ate(self):
        cates = self.estimate_cate()
        cates = {k: v for k, v in cates.items() if k in self.X_counts}
        return _aggregate_cates_to_ate(cates, self, weights=self.X_counts)

    def estimate_cate(self):
        conf_ints = self.est.conf_int()
        cates = {}
        for human_name, col_name in self.cate_column_map.items():
            count = self.X_counts.get(human_name)
            cates[human_name] = CausalEstimate(
                value=self.est.params[col_name], std_error=self.est.bse[col_name], 
                p_value=self.est.pvalues[col_name], ci_lower=conf_ints.loc[col_name, 0], 
                ci_upper=conf_ints.loc[col_name, 1], estimator_instance=self, count=count
            )
            
        # Marginalize soil groups if this is a complex estimator
        for soil in ['Non-sandy', 'Sandy']:
            if soil not in cates:
                soil_subgroups = [g for g in cates if g.endswith(f"_{soil}")]
                if soil_subgroups:
                    soil_cates = {g: cates[g] for g in soil_subgroups}
                    soil_weights = {g: cates[g].count for g in soil_subgroups}
                    cates[soil] = _aggregate_cates_to_ate(soil_cates, self, weights=soil_weights)
                    
        return cates

class OLS_SOIL(BaseOLSEstimator):
    """Ordinary Least Squares for heterogeneous effect by soil type."""
    pass
class OLS_SOIL_RATE(DoseResponseMixin, BaseOLSEstimator):
    """Ordinary Least Squares for heterogeneous effect by soil and continuous rate."""
    def __init__(self, *args, featurizer_type='SPLINE', baseline_t='mean', **kwargs):
        super().__init__(*args, **kwargs)
        self.featurizer_type = featurizer_type
        self.baseline_t = baseline_t

    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, points, groups=None):
        features, _, _ = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, ['soil_sandy']
        )
        
        ols_data = transformed_data[features].copy()
        ols_data[outcome_name] = self.outcome
        
        is_sandy = ols_data['soil_sandy'].astype(bool)
        self.X_counts = {
            'Sandy': is_sandy.sum(),
            'Non-sandy': (~is_sandy).sum()
        }
        
        actual_treatment_col = 'fertN_total' if 'N' in treatment_name else 'fertP_total'
        self.T_absolute = data[actual_treatment_col].values
        
        _, self.treatment_transformer = build_treatment_featurizer(
            self.T_absolute, points.geometry, featurizer_type=self.featurizer_type
        )
        T_splined = self.treatment_transformer.fit_transform(self.T_absolute.reshape(-1, 1))
        
        for i in range(T_splined.shape[1]):
            ols_data[f'feat_{i}_sandy'] = T_splined[:, i] * is_sandy
            ols_data[f'feat_{i}_nosandy'] = T_splined[:, i] * (~is_sandy)
            
        ols_data = ols_data.drop(columns=['soil_sandy']).astype(float)
        
        self.cate_column_map = {} 
        
        return data, *self._fit_ols(ols_data, outcome_name)

class OLS_SOIL_RATE_SPLINE(OLS_SOIL_RATE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, featurizer_type='SPLINE', **kwargs)

class OLS_SOIL_RATE_POLY(OLS_SOIL_RATE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, featurizer_type='POLY', **kwargs)


class OLS_SOIL_TIME(BaseOLSEstimator):
    """Ordinary Least Squares for heterogeneous effect by soil and timing."""
    heterogeneity_builder = staticmethod(_build_time_matrix)


class OLS_SOIL_PFERT(BaseOLSEstimator):
    """Ordinary Least Squares for heterogeneous effect by soil and P-fertilizer type."""
    heterogeneity_builder = staticmethod(_build_pfert_matrix)


class OLS_SOIL_LEGUME(BaseOLSEstimator):
    """Ordinary Least Squares for heterogeneous effect by soil and legume history."""
    heterogeneity_builder = staticmethod(_build_legume_matrix)
    heterogeneity_drop_cols = ['history_maizelegume']
