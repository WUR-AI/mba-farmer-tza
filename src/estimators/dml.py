import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from econml.dml import LinearDML
from sklearn.model_selection import GroupKFold
import scipy.stats as stats

from src.estimators.base import BaseEstimator, CausalEstimate, get_feature_lists, build_treatment_featurizer, RANDOM_SEED, DoseResponseMixin
from src.estimators.helpers import trim_and_score_models, get_coefs_df_from_summary, _aggregate_cates_to_ate, _build_soil_matrix, _build_time_matrix, _build_legume_matrix, _build_pfert_matrix

class BaseDMLEstimator(BaseEstimator):
    heterogeneity_builder = None
    heterogeneity_drop_cols = []

    def __init__(self, outcome, treatment, controls, model_y=None, model_t=None, splits=None, random_state=RANDOM_SEED, cv=5, **kwargs):
        super().__init__(outcome, treatment, controls, splits, **kwargs)
        self.random_state = random_state
        self.cv = cv
        self.model_y = model_y
        self.model_t = model_t
        self.est = None
        self.x_names = []
        self.X = None

    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None):
        if self.heterogeneity_builder:
            X_het, x_names_het = self.heterogeneity_builder(transformed_data, treatment_name=treatment_name)
        else:
            X_het, x_names_het = None, []

        cols_to_drop = ['soil_sandy'] + self.heterogeneity_drop_cols
        if self.heterogeneity_builder:
            cols_to_drop += x_names_het

        features, w_names, x_names = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, cols_to_drop
        )

        data, transformed_data, Y, T, W, groups, _ = self._trim_and_get_splits(
            data, transformed_data, w_names, x_names, groups
        )

        X_soil, soil_names = _build_soil_matrix(transformed_data)
        if self.heterogeneity_builder:
            X_het, _ = self.heterogeneity_builder(transformed_data, treatment_name=treatment_name)
                
            self.X = np.hstack([
                X_het * X_soil[:, 0].reshape(-1, 1),
                X_het * X_soil[:, 1].reshape(-1, 1)
            ])
            self.x_names = [f'{j}_{i}' for j in soil_names for i in x_names_het]
        else:
            self.X = X_soil
            self.x_names = soil_names

        self._fit_dml(Y, T, W, groups)
        return data, self.model_y, self.model_t

    def _trim_and_get_splits(self, data, transformed_data, w_names, x_names, groups, points=None):
        Y = self.outcome.values
        T = self.treatment.values
        if getattr(self, 'is_first_pass', False):
            X_train = transformed_data.loc[:, w_names + x_names]
            data, transformed_data, points_out, diagnostics = trim_and_score_models(
                self.model_y, self.model_t, X_train, Y, T, self.cv, data, transformed_data, points=points, groups=groups, random_state=self.random_state
            )
            self.diagnostics = diagnostics
            if groups is not None:
                groups = groups[diagnostics['mask_support']]
        else:
            self.diagnostics = None
            points_out = points

        Y = data[self.outcome.name].values
        T = data[self.treatment.name].values
        W = transformed_data[w_names].values
        return data, transformed_data, Y, T, W, groups, points_out

    def _fit_dml(self, Y, T, W, groups, treatment_featurizer=None):
        if groups is not None:
            from sklearn.model_selection import GroupKFold
            cv_splits = list(GroupKFold(
                n_splits=int(self.cv), shuffle=True, random_state=self.random_state
            ).split(self.X, Y, groups=groups))
        else:
            cv_splits = self.cv
            
        model_t = MultiOutputRegressor(self.model_t) if treatment_featurizer is not None else self.model_t
            
        self.est = LinearDML(
            model_y=self.model_y, 
            model_t=model_t, 
            treatment_featurizer=treatment_featurizer,
            discrete_treatment=False,
            cv=cv_splits,
            random_state=self.random_state, 
            fit_cate_intercept=False,
        )
        
        self.est = self.est.fit(
            Y=Y, T=T, W=W, X=self.X,
            inference='statsmodels', cache_values=True,
        )
        return self.model_y, self.model_t

    def estimate_ate(self):
        if hasattr(self.est, 'ate_inference'):
            try:
                inf = self.est.ate_inference(X=self.X)
                p_val = inf.pvalue()
                if isinstance(p_val, (np.ndarray, list, tuple)):
                    p_val = p_val[0] if len(p_val) > 0 else np.nan
                
                ci = inf.conf_int_mean()
                ci_lower, ci_upper = ci[0], ci[1]
                
                return CausalEstimate(
                    value=inf.mean_point,
                    std_error=inf.stderr_mean,
                    p_value=p_val,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    estimator_instance=self,
                    count=len(self.X)
                )
            except Exception:
                pass
                
        # Fallback for OLS or if ate_inference fails
        cates = self.estimate_cate()
        if hasattr(self, 'x_names'):
            cates = {k: v for k, v in cates.items() if k in self.x_names}
            weights = {name: self.X[:, i].sum() for i, name in enumerate(self.x_names)}
        else:
            weights = None
        return _aggregate_cates_to_ate(cates, self, weights=weights)

    def estimate_cate(self):
        df = get_coefs_df_from_summary(self.est, self.x_names)
        cates = {}
        for group in self.x_names:
            idx = self.x_names.index(group)
            count = int(self.X[:, idx].sum())
            cates[group] = CausalEstimate(
                value=df.loc[group, 'point_estimate'],
                std_error=df.loc[group, 'stderr'],
                p_value=df.loc[group, 'pvalue'],
                ci_lower=df.loc[group, 'ci_lower'],
                ci_upper=df.loc[group, 'ci_upper'],
                estimator_instance=self,
                count=count
            )
        # Marginalize soil groups if this is a complex estimator (like TIME, LEGUME, PFERT)
        for soil in ['Non-sandy', 'Sandy']:
            if soil not in self.x_names:
                # Find all subgroups that start with this soil prefix
                soil_subgroups = [g for g in self.x_names if g.startswith(f"{soil}_")]
                if soil_subgroups:
                    soil_cates = {g: cates[g] for g in soil_subgroups}
                    soil_weights = {g: cates[g].count for g in soil_subgroups}
                    cates[soil] = _aggregate_cates_to_ate(soil_cates, self, weights=soil_weights)
                    
        return cates

# DML ESTIMATORS
# ==========================================

class DML_SOIL(BaseDMLEstimator):
    """Homogeneous effect (linear response) varying by soil type."""
    pass
class DML_SOIL_RATE(DoseResponseMixin, BaseDMLEstimator):
    """Heterogeneous effect (response curve) varying by soil type."""
    def __init__(self, *args, featurizer_type='SPLINE', baseline_t='mean', **kwargs):
        super().__init__(*args, **kwargs)
        self.featurizer_type = featurizer_type
        self.baseline_t = baseline_t

    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, points, groups=None):
        features, w_names, x_names = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, ['soil_sandy']
        )
        
        data, transformed_data, Y, _, W, groups, points_out = self._trim_and_get_splits(
            data, transformed_data, w_names, x_names, groups, points=points
        )
        
        self.X, self.x_names = _build_soil_matrix(transformed_data)
        
        actual_treatment_col = 'fertN_total' if 'N' in treatment_name else 'fertP_total'
        self.T_absolute = data[actual_treatment_col].values
        
        treatment_featurizer, self.treatment_transformer = build_treatment_featurizer(
            self.T_absolute, points_out.geometry, featurizer_type=self.featurizer_type
        )
        
        self._fit_dml(Y, self.T_absolute.reshape(-1, 1), W, groups, treatment_featurizer=treatment_featurizer)
        return data, self.model_y, self.model_t

class DML_SOIL_RATE_SPLINE(DML_SOIL_RATE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, featurizer_type='SPLINE', **kwargs)

class DML_SOIL_RATE_POLY(DML_SOIL_RATE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, featurizer_type='POLY', **kwargs)


class DML_SOIL_TIME(BaseDMLEstimator):
    """Heterogeneous effect varying by soil type and timing strategy."""
    heterogeneity_builder = staticmethod(_build_time_matrix)

class DML_SOIL_LEGUME(BaseDMLEstimator):
    """Heterogeneous effect varying by soil type and legume history."""
    heterogeneity_builder = staticmethod(_build_legume_matrix)
    heterogeneity_drop_cols = ['history_maizelegume']

class DML_SOIL_PFERT(BaseDMLEstimator):
    """Heterogeneous effect varying by soil type and P-fertilizer type."""
    heterogeneity_builder = staticmethod(_build_pfert_matrix)

