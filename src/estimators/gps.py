import numpy as np
import pandas as pd
import scipy.stats as stats
from src.estimators.base import BaseEstimator, CausalEstimate, get_feature_lists
from src.estimators.helpers import _build_soil_matrix

class GPS_SOIL_RATE(BaseEstimator):
    """
    Pooled Hirano & Imbens (2004) cell-means interacted specification for GPS.
    """
    
    def __init__(self, outcome, treatment, controls, **kwargs):
        super().__init__(outcome, treatment, controls, **kwargs)
        self.treatment_name = None
        self.outcome_name = None
        self.boot_alpha_hats = []
        self.boot_beta_hats = []
        self.boot_sigma_sq_hats = []
        
    def _fit_single_sample(self, Y, T, G, X):
        N = len(T)
        G_mat = G.reshape(-1, 1)
        
        # W_treat = [X, G * X]
        W_treat = np.hstack([X, G_mat * X])
        
        # GPS model
        beta_hat, _, _, _ = np.linalg.lstsq(W_treat, T, rcond=None)
        residuals = T - W_treat @ beta_hat
        sigma_sq_hat = np.sum(residuals**2) / max(1, (N - W_treat.shape[1]))
        
        mu = W_treat @ beta_hat
        R_hat = (1.0 / np.sqrt(2 * np.pi * sigma_sq_hat)) * np.exp(- (T - mu)**2 / (2 * sigma_sq_hat))
        
        # Outcome model
        phi = np.vstack([
            np.ones(N),
            T,
            T**2,
            R_hat,
            R_hat**2,
            T * R_hat
        ]).T
        
        W_0 = (1 - G_mat) * phi
        W_1 = G_mat * phi
        W_outcome = np.hstack([W_0, W_1])
        
        alpha_hat, _, _, _ = np.linalg.lstsq(W_outcome, Y, rcond=None)
        
        return beta_hat, sigma_sq_hat, alpha_hat

    def fit(self, data, transformed_data, adjustment_set, treatment_name, outcome_name, groups=None, **kwargs):
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        
        Y = self.outcome.to_numpy(dtype=float)
        T = self.treatment.to_numpy(dtype=float)
        
        X_soil, soil_names = _build_soil_matrix(transformed_data)
        G = X_soil[:, 1].astype(float)  # Sandy is G=1
        
        features, w_names, x_cols = get_feature_lists(
            adjustment_set, transformed_data, treatment_name, outcome_name, []
        )
        X_df = transformed_data[features].copy()
        X_df.insert(0, 'Intercept', 1.0)
        X = X_df.to_numpy(dtype=float)
        
        self.Y = Y
        self.T = T
        self.G = G
        self.X = X
        
        self.beta_hat, self.sigma_sq_hat, self.alpha_hat = self._fit_single_sample(Y, T, G, X)
        
        # Bootstrap
        B = 100
        idx_0 = np.where(G == 0)[0]
        idx_1 = np.where(G == 1)[0]
        
        np.random.seed(self.kwargs.get('random_state', 43))
        
        for b in range(B):
            samp_0 = np.random.choice(idx_0, size=len(idx_0), replace=True)
            samp_1 = np.random.choice(idx_1, size=len(idx_1), replace=True)
            boot_idx = np.concatenate([samp_0, samp_1])
            
            beta_b, sigma_b, alpha_b = self._fit_single_sample(
                Y[boot_idx], T[boot_idx], G[boot_idx], X[boot_idx]
            )
            self.boot_beta_hats.append(beta_b)
            self.boot_sigma_sq_hats.append(sigma_b)
            self.boot_alpha_hats.append(alpha_b)
            
        return data, None, None

    def _predict_dose_response(self, t_eval, beta, sigma_sq, alpha, G, X):
        N = len(G)
        G_mat = G.reshape(-1, 1)
        W_treat = np.hstack([X, G_mat * X])
        mu = W_treat @ beta
        
        idx_0 = np.where(G == 0)[0]
        idx_1 = np.where(G == 1)[0]
        
        mu_overall = np.zeros(len(t_eval))
        mu_0 = np.zeros(len(t_eval))
        mu_1 = np.zeros(len(t_eval))
        
        for k, t in enumerate(t_eval):
            r_hyp = (1.0 / np.sqrt(2 * np.pi * sigma_sq)) * np.exp(- (t - mu)**2 / (2 * sigma_sq))
            
            phi = np.vstack([
                np.ones(N),
                np.full(N, t),
                np.full(N, t**2),
                r_hyp,
                r_hyp**2,
                t * r_hyp
            ]).T
            
            W_0 = (1 - G_mat) * phi
            W_1 = G_mat * phi
            W_outcome = np.hstack([W_0, W_1])
            
            y_pred = W_outcome @ alpha
            
            mu_overall[k] = np.mean(y_pred)
            mu_0[k] = np.mean(y_pred[idx_0]) if len(idx_0) > 0 else np.nan
            mu_1[k] = np.mean(y_pred[idx_1]) if len(idx_1) > 0 else np.nan
            
        return mu_overall, mu_0, mu_1

    def estimate_dose_response(self):
        baseline_t = np.percentile(self.T, 5)
        q5, q95 = np.percentile(self.T, [5, 95])
        t_eval = np.arange(np.ceil(q5), np.floor(q95) + 1, 1.0)
        
        t_eval_with_ref = np.concatenate([[baseline_t], t_eval])
        
        mu_overall, mu_0, mu_1 = self._predict_dose_response(
            t_eval_with_ref, self.beta_hat, self.sigma_sq_hat, self.alpha_hat, self.G, self.X
        )
        
        lift_overall = mu_overall[1:] - mu_overall[0]
        lift_0 = mu_0[1:] - mu_0[0]
        lift_1 = mu_1[1:] - mu_1[0]
        
        boot_lift_overall = []
        boot_lift_0 = []
        boot_lift_1 = []
        
        for b in range(len(self.boot_alpha_hats)):
            m_o, m_0, m_1 = self._predict_dose_response(
                t_eval_with_ref, self.boot_beta_hats[b], self.boot_sigma_sq_hats[b], self.boot_alpha_hats[b], self.G, self.X
            )
            boot_lift_overall.append(m_o[1:] - m_o[0])
            boot_lift_0.append(m_0[1:] - m_0[0])
            boot_lift_1.append(m_1[1:] - m_1[0])
            
        boot_lift_overall = np.array(boot_lift_overall)
        boot_lift_0 = np.array(boot_lift_0)
        boot_lift_1 = np.array(boot_lift_1)
        
        records = []
        for k, t in enumerate(t_eval):
            records.append({
                'Soil type': 'Overall', 'T': t, 'Mean Lift': lift_overall[k],
                'Lower CI': np.percentile(boot_lift_overall[:, k], 2.5),
                'Upper CI': np.percentile(boot_lift_overall[:, k], 97.5)
            })
            records.append({
                'Soil type': 'Non-sandy', 'T': t, 'Mean Lift': lift_0[k],
                'Lower CI': np.percentile(boot_lift_0[:, k], 2.5),
                'Upper CI': np.percentile(boot_lift_0[:, k], 97.5)
            })
            records.append({
                'Soil type': 'Sandy', 'T': t, 'Mean Lift': lift_1[k],
                'Lower CI': np.percentile(boot_lift_1[:, k], 2.5),
                'Upper CI': np.percentile(boot_lift_1[:, k], 97.5)
            })
            
        return pd.DataFrame(records).set_index(['Soil type', 'T']).sort_index()

    def estimate_average_marginal_effect(self):
        N_step = 10 if 'N' in self.treatment_name else 5
        q5, q95 = np.percentile(self.T, [5, 95])
        start_t = np.ceil(q5 / N_step) * N_step
        end_t = np.floor(q95 / N_step) * N_step
        t_grid = np.arange(start_t, end_t, N_step)
        
        # We need Y(t + N_step) and Y(t)
        t_eval_base = t_grid
        t_eval_next = t_grid + N_step
        t_eval_combined = np.concatenate([t_eval_base, t_eval_next])
        
        mu_overall, mu_0, mu_1 = self._predict_dose_response(
            t_eval_combined, self.beta_hat, self.sigma_sq_hat, self.alpha_hat, self.G, self.X
        )
        
        half = len(t_grid)
        ME_overall = (mu_overall[half:] - mu_overall[:half]) / N_step
        ME_0 = (mu_0[half:] - mu_0[:half]) / N_step
        ME_1 = (mu_1[half:] - mu_1[:half]) / N_step
        
        boot_ME_overall = []
        boot_ME_0 = []
        boot_ME_1 = []
        
        for b in range(len(self.boot_alpha_hats)):
            m_o, m_0, m_1 = self._predict_dose_response(
                t_eval_combined, self.boot_beta_hats[b], self.boot_sigma_sq_hats[b], self.boot_alpha_hats[b], self.G, self.X
            )
            boot_ME_overall.append((m_o[half:] - m_o[:half]) / N_step)
            boot_ME_0.append((m_0[half:] - m_0[:half]) / N_step)
            boot_ME_1.append((m_1[half:] - m_1[:half]) / N_step)
            
        boot_ME_overall = np.array(boot_ME_overall)
        boot_ME_0 = np.array(boot_ME_0)
        boot_ME_1 = np.array(boot_ME_1)
        
        records = []
        for k, t in enumerate(t_grid):
            records.append({
                'Soil type': 'Overall', 'T': t, 'Mean Lift': ME_overall[k],
                'Lower CI': np.percentile(boot_ME_overall[:, k], 2.5),
                'Upper CI': np.percentile(boot_ME_overall[:, k], 97.5)
            })
            records.append({
                'Soil type': 'Non-sandy', 'T': t, 'Mean Lift': ME_0[k],
                'Lower CI': np.percentile(boot_ME_0[:, k], 2.5),
                'Upper CI': np.percentile(boot_ME_0[:, k], 97.5)
            })
            records.append({
                'Soil type': 'Sandy', 'T': t, 'Mean Lift': ME_1[k],
                'Lower CI': np.percentile(boot_ME_1[:, k], 2.5),
                'Upper CI': np.percentile(boot_ME_1[:, k], 97.5)
            })
            
        return pd.DataFrame(records).set_index(['Soil type', 'T']).sort_index()

    def _calc_effects(self, beta, sigma_sq, alpha):
        t_ref = np.percentile(self.T, 5)
        
        G_mat = self.G.reshape(-1, 1)
        W_treat = np.hstack([self.X, G_mat * self.X])
        mu = W_treat @ beta
        
        r_ref = (1.0 / np.sqrt(2 * np.pi * sigma_sq)) * np.exp(- (t_ref - mu)**2 / (2 * sigma_sq))
        phi_ref = np.vstack([
            np.ones(len(self.T)),
            np.full(len(self.T), t_ref),
            np.full(len(self.T), t_ref**2),
            r_ref,
            r_ref**2,
            t_ref * r_ref
        ]).T
        W_0_ref = (1 - G_mat) * phi_ref
        W_1_ref = G_mat * phi_ref
        y_ref = np.hstack([W_0_ref, W_1_ref]) @ alpha
        
        r_obs = (1.0 / np.sqrt(2 * np.pi * sigma_sq)) * np.exp(- (self.T - mu)**2 / (2 * sigma_sq))
        phi_obs = np.vstack([
            np.ones(len(self.T)),
            self.T,
            self.T**2,
            r_obs,
            r_obs**2,
            self.T * r_obs
        ]).T
        W_0_obs = (1 - G_mat) * phi_obs
        W_1_obs = G_mat * phi_obs
        y_obs = np.hstack([W_0_obs, W_1_obs]) @ alpha
        
        scales = self.T - t_ref
        valid_idx = scales > 0
        
        eff = (y_obs[valid_idx] - y_ref[valid_idx]) / scales[valid_idx]
        g_valid = self.G[valid_idx]
        
        return np.mean(eff), np.mean(eff[g_valid == 0]), np.mean(eff[g_valid == 1])

    def estimate_cate(self):
        ate, cate_0, cate_1 = self._calc_effects(self.beta_hat, self.sigma_sq_hat, self.alpha_hat)
        
        boot_ate = []
        boot_cate_0 = []
        boot_cate_1 = []
        for b in range(len(self.boot_alpha_hats)):
            b_ate, b_c0, b_c1 = self._calc_effects(self.boot_beta_hats[b], self.boot_sigma_sq_hats[b], self.boot_alpha_hats[b])
            boot_ate.append(b_ate)
            boot_cate_0.append(b_c0)
            boot_cate_1.append(b_c1)
            
        def _build_est(val, boots, name):
            std_err = np.std(boots)
            t_stat = val / std_err if std_err != 0 else np.nan
            p_val = 2 * stats.norm.sf(np.abs(t_stat)) if not np.isnan(t_stat) else np.nan
            count = np.sum(self.G == (1 if name == 'Sandy' else 0))
            return CausalEstimate(
                value=val, std_error=std_err, p_value=p_val,
                ci_lower=np.percentile(boots, 2.5), ci_upper=np.percentile(boots, 97.5),
                estimator_instance=self, count=count
            )
            
        cates = {
            'Sandy': _build_est(cate_1, boot_cate_1, 'Sandy'),
            'Non-sandy': _build_est(cate_0, boot_cate_0, 'Non-sandy')
        }
        
        # High Non-Return Point via Empirical Bootstrap
        try:
            q50 = np.percentile(self.T, 50)
            q95 = np.percentile(self.T, 95)
            t_grid_hnr = np.arange(q50, q95, 0.5)
            
            t_plus = t_grid_hnr + 0.5
            t_minus = t_grid_hnr - 0.5
            
            boot_ME_0 = []
            boot_ME_1 = []
            
            for b in range(len(self.boot_alpha_hats)):
                _, m_0_p, m_1_p = self._predict_dose_response(t_plus, self.boot_beta_hats[b], self.boot_sigma_sq_hats[b], self.boot_alpha_hats[b], self.G, self.X)
                _, m_0_m, m_1_m = self._predict_dose_response(t_minus, self.boot_beta_hats[b], self.boot_sigma_sq_hats[b], self.boot_alpha_hats[b], self.G, self.X)
                boot_ME_0.append(m_0_p - m_0_m)
                boot_ME_1.append(m_1_p - m_1_m)
                
            boot_ME_0 = np.array(boot_ME_0)
            boot_ME_1 = np.array(boot_ME_1)
            
            for name, ME_sims, g_val in [('Non-sandy', boot_ME_0, 0), ('Sandy', boot_ME_1, 1)]:
                high_non_return_points = []
                for i in range(len(ME_sims)):
                    neg_idx = np.where(ME_sims[i] <= 0)[0]
                    if len(neg_idx) > 0:
                        high_non_return_points.append(t_grid_hnr[neg_idx[0]])
                    else:
                        high_non_return_points.append(np.nan)
                        
                valid_points = np.array([p for p in high_non_return_points if not np.isnan(p)])
                
                if len(valid_points) > 0:
                    point_est = np.median(valid_points)
                    ci_lower = np.percentile(valid_points, 2.5)
                    ci_upper = np.percentile(valid_points, 97.5)
                    std_error = np.std(valid_points)
                else:
                    point_est, ci_lower, ci_upper, std_error = np.nan, np.nan, np.nan, np.nan
                    
                cates[f'{name}: High Non-Return'] = CausalEstimate(
                    value=point_est, std_error=std_error, p_value=np.nan,
                    ci_lower=ci_lower, ci_upper=ci_upper, estimator_instance=self,
                    count=np.sum(self.G == g_val)
                )
        except Exception as e:
            print(f"Failed to estimate High Non-Return for GPS_SOIL_RATE: {e}")
            
        return cates

    def estimate_ate(self):
        ate, _, _ = self._calc_effects(self.beta_hat, self.sigma_sq_hat, self.alpha_hat)
        
        boot_ate = []
        for b in range(len(self.boot_alpha_hats)):
            b_ate, _, _ = self._calc_effects(self.boot_beta_hats[b], self.boot_sigma_sq_hats[b], self.boot_alpha_hats[b])
            boot_ate.append(b_ate)
            
        std_err = np.std(boot_ate)
        t_stat = ate / std_err if std_err != 0 else np.nan
        p_val = 2 * stats.norm.sf(np.abs(t_stat)) if not np.isnan(t_stat) else np.nan
        
        return CausalEstimate(
            value=ate, std_error=std_err, p_value=p_val,
            ci_lower=np.percentile(boot_ate, 2.5), ci_upper=np.percentile(boot_ate, 97.5),
            estimator_instance=self, count=len(self.T)
        )
