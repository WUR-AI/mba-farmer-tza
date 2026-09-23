import numpy as np
import pandas as pd
from typing import Any, Optional
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from econml.dml import LinearDML
import scipy.stats as stats
from sklearn.model_selection import cross_val_predict, cross_val_score, GroupKFold

from src.estimators.base import (
    BaseEstimator, 
    CausalEstimate, 
    get_feature_lists, 
    build_treatment_featurizer,
    RANDOM_SEED
)

def trim_and_score_models(model_y, model_t, X_train, Y, T, cv, data, transformed_data, points=None, groups=None):
    if groups is not None:
        cv_gen = list(GroupKFold(n_splits=cv).split(X_train, Y, groups=groups))
    else:
        cv_gen = cv

    print("Evaluating models with cross_val_score...")
    r2_t = cross_val_score(model_t, X_train, T, cv=cv_gen, groups=groups, scoring='r2', n_jobs=-1)
    r2_y = cross_val_score(model_y, X_train, Y, cv=cv_gen, groups=groups, scoring='r2', n_jobs=-1)
    print(f"Treatment model R2: {np.mean(r2_t):.4f} (+/- {np.std(r2_t):.4f}) [{min(r2_t):.4f} - {max(r2_t):.4f}]")
    print(f"Outcome model R2: {np.mean(r2_y):.4f} (+/- {np.std(r2_y):.4f}) [{min(r2_y):.4f} - {max(r2_y):.4f}]")
    
    print("Performing GPS trimming...")
    T_pred = cross_val_predict(model_t, X_train, T, cv=cv_gen, groups=groups, n_jobs=-1)
    res_var = np.var(T - T_pred)
    gps_density = stats.norm.pdf(T, loc=T_pred, scale=np.sqrt(res_var))
    
    threshold = np.quantile(gps_density, 0.05)
    mask = gps_density > threshold
    
    trimmed_data = data[mask].copy()
    trimmed_transformed = transformed_data[mask].copy()
    trimmed_points = points[mask].copy() if points is not None else None
    
    print(f"GPS trimming: retained {mask.sum()} / {len(data)} samples.")
    diagnostics = {'T_pred': T_pred, 'mask_support': mask, 'T': T}
    return trimmed_data, trimmed_transformed, trimmed_points, diagnostics

def get_coefs_df_from_summary(est, x_names):
    """Helper to extract coefficients from LinearDML summary."""
    sm_summary = est.summary(feature_names=x_names)
    table_idx = len(sm_summary.tables) - 1
    df = pd.read_html(sm_summary.tables[table_idx].as_html(), header=0, index_col=0)[0]
    return df

def _aggregate_cates_to_ate(cates, estimator_instance, weights=None):
    if weights is None:
        ate = np.mean([c.value for c in cates.values()])
        ate_se = np.sqrt(np.sum([c.std_error**2 for c in cates.values()])) / len(cates)
    else:
        weight_sum = np.sum([weights[k] for k in cates.keys()])
        norm_weights = {k: weights[k]/weight_sum for k in cates.keys()}
        ate = np.sum([c.value * norm_weights[k] for k, c in cates.items()])
        ate_se = np.sqrt(np.sum([(c.std_error * norm_weights[k])**2 for k, c in cates.items()]))
        
    ci_lower = ate - 1.96 * ate_se
    ci_upper = ate + 1.96 * ate_se
    t_stat = ate / ate_se if ate_se != 0 else np.nan
    p_value = 2 * stats.norm.sf(np.abs(t_stat)) if not np.isnan(t_stat) else np.nan

    return CausalEstimate(
        value=ate, std_error=ate_se, p_value=p_value, 
        ci_lower=ci_lower, ci_upper=ci_upper, estimator_instance=estimator_instance
    )

def _build_soil_matrix(transformed_data):
    sandy_series = transformed_data['soil_sandy'].to_numpy()
    X_soil = np.vstack([~sandy_series.astype(bool), sandy_series]).T
    x_names = ['Non-sandy', 'Sandy']
    return X_soil, x_names

def _build_time_matrix(transformed_data, treatment_name):
    fertilizer_type = 'N' if 'N' in treatment_name else 'P'
    basal_col = f'fert{fertilizer_type}_basal'
    tops_cols = [
        f'fert{fertilizer_type}_kneehigh', f'fert{fertilizer_type}_56leaves', 
        f'fert{fertilizer_type}_810leaves', f'fert{fertilizer_type}_tasselingsilking'
    ]
    
    has_basal = transformed_data[basal_col] > 0
    n_tops = (transformed_data[tops_cols] > 0).sum(axis=1)
    
    strategy = pd.Series(index=transformed_data.index, dtype=str)
    strategy[~has_basal] = "Top-Only"
    strategy[has_basal & (n_tops < 1)] = "Basal-Only"
    strategy[has_basal & (n_tops == 1)] = "Basal+Single"
    strategy[has_basal & (n_tops > 1)] = "Basal+Split"
    
    x_names_time = ["Top-Only", "Basal-Only", "Basal+Single", "Basal+Split"]
    for cat in x_names_time:
        transformed_data[cat] = (strategy == cat).astype(float)
        
    X_time = transformed_data[x_names_time].to_numpy()
    return X_time, x_names_time

def _build_legume_matrix(transformed_data):
    is_legume = transformed_data['history_maizelegume'].to_numpy().astype(bool)
    X_legume = np.vstack([~is_legume, is_legume]).T
    x_names_legume = ['Non-legume', 'Legume']
    return X_legume, x_names_legume

def _build_pfert_matrix(transformed_data):
    p_types_dict = {
        'DAP': ['DAP'],
        'Other-NP': [
            'Yara-Mila-Cereal', 'Yara-Mila-OTESHA', 'NPK-20|10|10', 
            'NPK-14|23|14', 'NPS', 'NPSZinc'
        ]
    }
    
    def assign_p_fertilizer_group(row):
        basal_types = row['basaltypes'].split() if pd.notna(row['basaltypes']) else []
        top1types = row['top1types'].split() if pd.notna(row['top1types']) else []
        all_types = basal_types + top1types
        types = []
        for group, fertilizers in p_types_dict.items():
            if any(fert in all_types for fert in fertilizers):
                types.append(group)
        if types == []:
            return 'N-Only'
        elif len(types) > 1:
            return 'Other-NP'
        else:
            return types[0]

    strategy = transformed_data.apply(assign_p_fertilizer_group, axis=1)
    
    x_names_pfert = ['N-Only', 'DAP', 'Other-NP']
    for cat in x_names_pfert:
        transformed_data[cat] = (strategy == cat).astype(float)
        
    X_pfert = transformed_data[x_names_pfert].to_numpy()
    return X_pfert, x_names_pfert


# ==========================================
# BASE CLASSES
# ==========================================

class BaseDMLEstimator(BaseEstimator):
    def __init__(self, outcome, treatment, controls, model_y=None, model_t=None, splits=None, random_state=RANDOM_SEED, cv=5, **kwargs):
        super().__init__(outcome, treatment, controls, splits, **kwargs)
        self.random_state = random_state
        self.cv = cv
        self.model_y = model_y
        self.model_t = model_t
        self.est = None
        self.x_names = []
        self.X = None

    def _trim_and_get_splits(self, data, transformed_data, w_names, x_names, groups, points=None):
        Y = self.outcome.values
        T = self.treatment.values
        if getattr(self, 'is_first_pass', False):
            X_train = transformed_data.loc[:, w_names + x_names]
            data, transformed_data, points_out, diagnostics = trim_and_score_models(
                self.model_y, self.model_t, X_train, Y, T, self.cv, data, transformed_data, points=points, groups=groups
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
            cv_splits = list(GroupKFold(n_splits=self.cv).split(self.X, Y, groups=groups))
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
        return cates


class BaseOLSEstimator(BaseEstimator):
    def __init__(self, outcome, treatment, controls, splits=None, **kwargs):
        super().__init__(outcome, treatment, controls, splits, **kwargs)
        self.est = None
        self.cate_column_map = {}
        self.X_counts = {}
        
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
        return cates

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
            np.random.seed(42)
            n_sims = 10000
            param_vals = param.values if hasattr(param, 'values') else np.array(param)
            cov_vals = param_cov_matrix.values if hasattr(param_cov_matrix, 'values') else np.array(param_cov_matrix)
            
            # Ensure it is symmetric positive semi-definite
            cov_vals = (cov_vals + cov_vals.T) / 2
            
            simulated_params = np.random.multivariate_normal(param_vals, cov_vals, size=n_sims)
            
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
                    
                cates[f'{soil}: High Non-Return'] = CausalEstimate(
                    value=point_est, std_error=std_error, p_value=np.nan, 
                    ci_lower=ci_lower, ci_upper=ci_upper, estimator_instance=self
                )
                
        except Exception as e:
            print(f"Failed to estimate High Non-Return CATE parameters: {e}")
            import traceback
            traceback.print_exc()
            
        return cates
# ==========================================
# DML ESTIMATORS
# ==========================================

class DML_SOIL(BaseDMLEstimator):
    """Homogeneous effect (linear response) varying by soil type."""
    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None):
        features, w_names, x_names = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, ['soil_sandy']
        )
        
        data, transformed_data, Y, T, W, groups, _ = self._trim_and_get_splits(
            data, transformed_data, w_names, x_names, groups
        )
        
        self.X, self.x_names = _build_soil_matrix(transformed_data)
        
        self._fit_dml(Y, T, W, groups)
        return data, self.model_y, self.model_t


class DML_SOIL_RATE(DoseResponseMixin, BaseDMLEstimator):
    """Heterogeneous effect (response curve) varying by soil type."""
    def __init__(self, *args, featurizer_type='SPLINE', baseline_t='q05', **kwargs):
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
    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None):
        X_time, x_names_time = _build_time_matrix(transformed_data, treatment_name)
        features, w_names, x_names = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, x_names_time + ['soil_sandy']
        )
        
        data, transformed_data, Y, T, W, groups, _ = self._trim_and_get_splits(
            data, transformed_data, w_names, x_names, groups
        )
        
        X_time, _ = _build_time_matrix(transformed_data, treatment_name)
        X_soil, _ = _build_soil_matrix(transformed_data)
        
        self.X = np.hstack([
            X_time * X_soil[:, 0].reshape(-1, 1),
            X_time * X_soil[:, 1].reshape(-1, 1)
        ])
        
        self.x_names = [f'{j}_{i}' for j in ('Non-sandy', 'Sandy') for i in x_names_time]
        
        self._fit_dml(Y, T, W, groups)
        return data, self.model_y, self.model_t


class DML_SOIL_LEGUME(BaseDMLEstimator):
    """Heterogeneous effect varying by soil type and legume history."""
    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None):
        features, w_names, x_names = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, ['history_maizelegume', 'soil_sandy']
        )
        
        data, transformed_data, Y, T, W, groups, _ = self._trim_and_get_splits(
            data, transformed_data, w_names, x_names, groups
        )
        
        X_soil, _ = _build_soil_matrix(transformed_data)
        X_legume, _ = _build_legume_matrix(transformed_data)
        
        self.X = np.hstack([
            X_legume * X_soil[:, 0].reshape(-1, 1),
            X_legume * X_soil[:, 1].reshape(-1, 1)
        ])
        
        self.x_names = [f'{j}_{i}' for j in ('Non-sandy', 'Sandy') for i in ('Non-legume', 'Legume')]
        
        self._fit_dml(Y, T, W, groups)
        return data, self.model_y, self.model_t


class DML_SOIL_PFERT(BaseDMLEstimator):
    """Heterogeneous effect varying by soil type and P-fertilizer type."""
    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None):
        X_pfert, x_names_pfert = _build_pfert_matrix(transformed_data)
        features, w_names, x_names = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, x_names_pfert + ['soil_sandy']
        )
        
        data, transformed_data, Y, T, W, groups, _ = self._trim_and_get_splits(
            data, transformed_data, w_names, x_names, groups
        )
        
        X_pfert, _ = _build_pfert_matrix(transformed_data)
        X_soil, _ = _build_soil_matrix(transformed_data)
        
        self.X = np.hstack([
            X_pfert * X_soil[:, 0].reshape(-1, 1),
            X_pfert * X_soil[:, 1].reshape(-1, 1)
        ])
        
        self.x_names = [f'{j}_{i}' for j in ('Non-sandy', 'Sandy') for i in x_names_pfert]
        
        self._fit_dml(Y, T, W, groups)
        return data, self.model_y, self.model_t

# ==========================================
# OLS ESTIMATORS
# ==========================================

class OLS_SOIL(BaseOLSEstimator):
    """Ordinary Least Squares for heterogeneous effect by soil type."""
    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None):
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
        
        ols_data['N_sandy'] = self.treatment * is_sandy
        ols_data['N_nosandy'] = self.treatment * (~is_sandy)
        ols_data = ols_data.drop(columns=['soil_sandy']).astype(float)
        
        self.cate_column_map = {'Sandy': 'N_sandy', 'Non-sandy': 'N_nosandy'}
        return data, *self._fit_ols(ols_data, outcome_name)


class OLS_SOIL_RATE(DoseResponseMixin, BaseOLSEstimator):
    """Ordinary Least Squares for heterogeneous effect by soil and continuous rate."""
    def __init__(self, *args, featurizer_type='SPLINE', baseline_t='q05', **kwargs):
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
        T_splined = self.treatment_transformer.transform(self.T_absolute.reshape(-1, 1))
        
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
    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None):
        X_time, x_names_time = _build_time_matrix(transformed_data, treatment_name)
        features, _, _ = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, x_names_time + ['soil_sandy']
        )
        
        ols_data = transformed_data[features].copy()
        ols_data[outcome_name] = self.outcome
        
        is_sandy = ols_data['soil_sandy'].astype(bool)
        
        self.cate_column_map = {}
        for c in x_names_time:
            c_val = ols_data[c]
            
            self.X_counts[f'N_{c}_Sandy'] = (c_val * is_sandy).sum()
            self.X_counts[f'N_{c}_Non-sandy'] = (c_val * ~is_sandy).sum()
            
            ols_data[f'N_{c}_Sandy'] = c_val * self.treatment * is_sandy
            ols_data[f'N_{c}_Non-sandy'] = c_val * self.treatment * ~is_sandy
            
            self.cate_column_map[f'N_{c}_Sandy'] = f'N_{c}_Sandy'
            self.cate_column_map[f'N_{c}_Non-sandy'] = f'N_{c}_Non-sandy'
            
        ols_data = ols_data.drop(columns=['soil_sandy'] + x_names_time).astype(float)
        return data, *self._fit_ols(ols_data, outcome_name)


class OLS_SOIL_PFERT(BaseOLSEstimator):
    """Ordinary Least Squares for heterogeneous effect by soil and P-fertilizer type."""
    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None):
        X_pfert, x_names_pfert = _build_pfert_matrix(transformed_data)
        features, _, _ = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, x_names_pfert + ['soil_sandy']
        )
        
        ols_data = transformed_data[features].copy()
        ols_data[outcome_name] = self.outcome
        
        is_sandy = ols_data['soil_sandy'].astype(bool)
        
        self.cate_column_map = {}
        for c in x_names_pfert:
            c_val = ols_data[c]
            
            self.X_counts[f'N_{c}_Sandy'] = (c_val * is_sandy).sum()
            self.X_counts[f'N_{c}_Non-sandy'] = (c_val * ~is_sandy).sum()
            
            ols_data[f'N_{c}_Sandy'] = c_val * self.treatment * is_sandy
            ols_data[f'N_{c}_Non-sandy'] = c_val * self.treatment * ~is_sandy
            
            self.cate_column_map[f'N_{c}_Sandy'] = f'N_{c}_Sandy'
            self.cate_column_map[f'N_{c}_Non-sandy'] = f'N_{c}_Non-sandy'
            
        ols_data = ols_data.drop(columns=['soil_sandy'] + x_names_pfert).astype(float)
        
        return data, *self._fit_ols(ols_data, outcome_name)


class OLS_SOIL_LEGUME(BaseOLSEstimator):
    """Ordinary Least Squares for heterogeneous effect by soil and legume history."""
    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None):
        features, _, _ = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, []
        )
        
        ols_data = transformed_data[features].copy()
        ols_data[outcome_name] = self.outcome
        
        is_sandy = ols_data['soil_sandy'].astype(bool)
        is_legume = ols_data['history_maizelegume'].astype(bool)
        
        self.X_counts = {
            'N_Legume_Sandy': (is_legume * is_sandy).sum(),
            'N_Non-legume_Sandy': (~is_legume * is_sandy).sum(),
            'N_Legume_Non-sandy': (is_legume * ~is_sandy).sum(),
            'N_Non-legume_Non-sandy': (~is_legume * ~is_sandy).sum(),
        }
        
        ols_data['N_Legume_Sandy'] = self.treatment * is_legume * is_sandy
        ols_data['N_Non-legume_Sandy'] = self.treatment * ~is_legume * is_sandy
        ols_data['N_Legume_Non-sandy'] = self.treatment * is_legume * ~is_sandy
        ols_data['N_Non-legume_Non-sandy'] = self.treatment * ~is_legume * ~is_sandy
        
        self.cate_column_map = {k: k for k in self.X_counts.keys()}
        
        ols_data = ols_data.drop(columns=['soil_sandy', 'history_maizelegume']).astype(float)
        return data, *self._fit_ols(ols_data, outcome_name)
