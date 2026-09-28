import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats

from src.estimators.base import BaseEstimator, CausalEstimate, get_feature_lists, RANDOM_SEED

# ==========================================

class RF_PREDICTIVE(BaseEstimator):
    """
    Naive predictive RF baseline (S-learner / G-computation).
    Fits a single RandomForestRegressor on (W + T) -> Y, then derives 
    causal estimands via interventional predictions.
    """
    def __init__(self, outcome, treatment, controls, model_y=None, model_t=None, 
                 splits=None, random_state=RANDOM_SEED, **kwargs):
        super().__init__(outcome, treatment, controls, splits, **kwargs)
        self.random_state = random_state
        self.rf_model = None
        self.feature_names = None
        self.treatment_col_idx = None
        self.T_absolute = None
        self.X_rf = None
        self.is_sandy = None
        self.X_counts = {}
        
    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None):
        # Resolve features from the adjustment set
        features, w_names, x_names = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, ['soil_sandy']
        )
        
        # The treatment variable in the data
        treatment_var = 'fertN_total' if 'N' in treatment_name else 'fertP_total'
        
        # Build feature matrix: all adjustment-set controls + treatment
        rf_feature_names = sorted(set(w_names + x_names))
        # Add treatment if not already in there
        if treatment_var not in rf_feature_names:
            rf_feature_names.append(treatment_var)
        
        # Drop rows with NaN in any feature or outcome
        outcome_var = self.outcome.name
        subset = transformed_data[rf_feature_names].join(data[[outcome_var]]).dropna()
        data = data.loc[subset.index]
        transformed_data = transformed_data.loc[subset.index]
        
        self.feature_names = rf_feature_names
        self.treatment_col_idx = rf_feature_names.index(treatment_var)
        self.T_absolute = transformed_data[treatment_var].values
        self.is_sandy = transformed_data['soil_sandy'].astype(bool).values
        self.X_counts = {
            'Sandy': int(self.is_sandy.sum()),
            'Non-sandy': int((~self.is_sandy).sum())
        }
        
        X = transformed_data[rf_feature_names].values
        Y = data[outcome_var].values
        
        self.X_rf = X
        self.Y = Y
        
        # Fit the RF
        self.rf_model = RandomForestRegressor(
            n_estimators=500, max_depth=None, 
            random_state=self.random_state, n_jobs=-1
        )
        self.rf_model.fit(X, Y)
        
        # Cross-validated R2 for diagnostics
        from sklearn.model_selection import cross_val_score
        cv_r2 = cross_val_score(
            RandomForestRegressor(n_estimators=500, max_depth=None, 
                                  random_state=self.random_state, n_jobs=-1),
            X, Y, cv=5, scoring='r2', n_jobs=-1
        )
        print(f"RF_PREDICTIVE CV R2: {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f}) [{cv_r2.min():.4f} - {cv_r2.max():.4f}]")
        
        return data, None, None
    
    def _predict_at_treatment(self, t_values):
        """Predict Y for each observation with treatment replaced by t_values."""
        X_mod = self.X_rf.copy()
        X_mod[:, self.treatment_col_idx] = t_values
        return self.rf_model.predict(X_mod)
    
    def _predict_at_treatment_per_tree(self, t_values):
        """Per-tree predictions with treatment replaced by t_values."""
        X_mod = self.X_rf.copy()
        X_mod[:, self.treatment_col_idx] = t_values
        # Returns shape (n_trees, n_samples)
        return np.array([tree.predict(X_mod) for tree in self.rf_model.estimators_])
    
    def _compute_effects_and_se(self, mask=None):
        """
        Compute individual per-unit-of-treatment effects:
        effect_i = [f̂(W_i, T_i) - f̂(W_i, t_baseline)] / (T_i - t_baseline)
        averaged across observations, with SE from per-tree variance.
        """
        t_baseline = np.mean(self.T_absolute)
        
        # Dose above baseline for each farmer
        dose = self.T_absolute - t_baseline
        
        # Per-tree predictions at observed T and at baseline
        tree_preds_obs = self._predict_at_treatment_per_tree(self.T_absolute)    # (n_trees, N)
        tree_preds_base = self._predict_at_treatment_per_tree(
            np.full_like(self.T_absolute, t_baseline)
        )  # (n_trees, N)
        
        # Per-tree individual lift
        tree_lifts = tree_preds_obs - tree_preds_base  # (n_trees, N)
        
        # Exclude farmers at the baseline (dose == 0) to avoid division by zero
        valid = dose > 0
        if mask is not None:
            valid = valid & mask
        
        tree_lifts = tree_lifts[:, valid]
        dose_valid = dose[valid]
        
        # Normalize: per-unit effects
        tree_effects = tree_lifts / dose_valid[np.newaxis, :]  # (n_trees, N_valid)
        
        # Per-tree average effect
        tree_avg_effects = tree_effects.mean(axis=1)  # (n_trees,)
        
        # Point estimate = ensemble mean
        mean_effect = tree_avg_effects.mean()
        # SE from variance across trees
        se = tree_avg_effects.std(ddof=1)
        
        return mean_effect, se

    def estimate_ate(self):
        mean_effect, se = self._compute_effects_and_se(mask=None)
        
        if se > 0:
            t_stat = mean_effect / se
            p_value = 2 * stats.norm.sf(np.abs(t_stat))
        else:
            p_value = np.nan
            
        return CausalEstimate(
            value=mean_effect,
            std_error=se,
            p_value=p_value,
            ci_lower=mean_effect - 1.96 * se,
            ci_upper=mean_effect + 1.96 * se,
            estimator_instance=self,
            count=len(self.T_absolute)
        )
    
    def estimate_cate(self):
        cates = {}
        for group_name, mask in [('Sandy', self.is_sandy), ('Non-sandy', ~self.is_sandy)]:
            mean_effect, se = self._compute_effects_and_se(mask=mask)
            
            if se > 0:
                t_stat = mean_effect / se
                p_value = 2 * stats.norm.sf(np.abs(t_stat))
            else:
                p_value = np.nan
                
            cates[group_name] = CausalEstimate(
                value=mean_effect,
                std_error=se,
                p_value=p_value,
                ci_lower=mean_effect - 1.96 * se,
                ci_upper=mean_effect + 1.96 * se,
                estimator_instance=self,
                count=int(mask.sum())
            )
        return cates
    
    def estimate_dose_response(self):
        """
        PDP-based dose-response curve per soil group.
        For each t in the grid, replace all farmers' treatment with t,
        predict, and average within each soil group. Report lift from baseline.
        """
        t_baseline = np.mean(self.T_absolute)
        q5, q95 = np.percentile(self.T_absolute, [5, 95])
        t_grid = np.arange(np.ceil(q5), np.floor(q95) + 1, 1.0)
        
        # Baseline predictions per soil group
        Y_baseline = self._predict_at_treatment(np.full(len(self.T_absolute), t_baseline))
        baseline_sandy = Y_baseline[self.is_sandy].mean()
        baseline_nonsandy = Y_baseline[~self.is_sandy].mean()
        
        # Per-tree baseline predictions for SE
        tree_baseline = self._predict_at_treatment_per_tree(
            np.full(len(self.T_absolute), t_baseline)
        )
        tree_base_sandy = tree_baseline[:, self.is_sandy].mean(axis=1)
        tree_base_nonsandy = tree_baseline[:, ~self.is_sandy].mean(axis=1)
        
        results = []
        for t in t_grid:
            Y_t = self._predict_at_treatment(np.full(len(self.T_absolute), t))
            tree_preds_t = self._predict_at_treatment_per_tree(
                np.full(len(self.T_absolute), t)
            )
            
            for group_name, mask, base_val, tree_base_group in [
                ('Sandy', self.is_sandy, baseline_sandy, tree_base_sandy),
                ('Non-sandy', ~self.is_sandy, baseline_nonsandy, tree_base_nonsandy)
            ]:
                lift = Y_t[mask].mean() - base_val
                
                # Per-tree lifts for SE
                tree_lifts = tree_preds_t[:, mask].mean(axis=1) - tree_base_group
                se = tree_lifts.std(ddof=1)
                
                results.append({
                    'Soil type': group_name,
                    'T': t,
                    'Mean Lift': lift,
                    'Standard Error': se,
                    'Lower CI': lift - 1.96 * se,
                    'Upper CI': lift + 1.96 * se
                })
        
        return pd.DataFrame(results).set_index(['Soil type', 'T']).sort_index()
    
    def estimate_average_marginal_effect(self, treatment_name=None):
        """
        Average marginal effect: for each t in a coarse grid,
        compute [DR(t + step) - DR(t)] / step per soil group.
        """
        if treatment_name is None:
            treatment_name = self.treatment.name
        step = 10 if 'N' in treatment_name else 5
        
        q5, q95 = np.percentile(self.T_absolute, [5, 95])
        start_t = np.ceil(q5 / step) * step
        end_t = np.floor(q95 / step) * step
        t_grid = np.arange(start_t, end_t, step)
        
        results = []
        for t in t_grid:
            for group_name, mask in [('Sandy', self.is_sandy), ('Non-sandy', ~self.is_sandy)]:
                Y_t = self._predict_at_treatment(np.full(len(self.T_absolute), t))
                Y_t_plus = self._predict_at_treatment(np.full(len(self.T_absolute), t + step))
                
                ame = (Y_t_plus[mask].mean() - Y_t[mask].mean()) / step
                
                # Per-tree AME for SE
                tree_t = self._predict_at_treatment_per_tree(np.full(len(self.T_absolute), t))
                tree_t_plus = self._predict_at_treatment_per_tree(np.full(len(self.T_absolute), t + step))
                tree_ame = (tree_t_plus[:, mask].mean(axis=1) - tree_t[:, mask].mean(axis=1)) / step
                se = tree_ame.std(ddof=1)
                
                results.append({
                    'Soil type': group_name,
                    'T': t,
                    'Mean Lift': ame,
                    'Standard Error': se,
                    'Lower CI': ame - 1.96 * se,
                    'Upper CI': ame + 1.96 * se
                })
        
        return pd.DataFrame(results).set_index(['Soil type', 'T']).sort_index()
