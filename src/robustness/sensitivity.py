import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def compute_ucc_sensitivity(estimate, se, count, grid_cd=None, grid_cy=None, rho=1.0, alpha=0.05, benchmark_covariates=None):
    """
    Computes sensitivity bounds for Unobserved Common Confounder (UCC).
    Formula: Bias = rho * sqrt((cy * cd) / (1 - cd)) * SE * sqrt(df)
    """
    if grid_cd is None:
        grid_cd = np.linspace(0.0, 0.5, 50)
    if grid_cy is None:
        grid_cy = np.linspace(0.0, 0.5, 50)
        
    cd_mesh, cy_mesh = np.meshgrid(grid_cd, grid_cy)
    
    # df approximation for DML
    df = count
    
    # Calculate Bias
    # To avoid division by zero when cd=1.0, we use np.clip
    cd_clipped = np.clip(cd_mesh, 0, 0.999)
    bias = rho * np.sqrt((cy_mesh * cd_clipped) / (1 - cd_clipped)) * se * np.sqrt(df)
    
    adj_estimate = estimate - bias
    
    # Calculate confidence limits
    # The standard error is rescaled in rigorous implementations, but the first-order approximation
    # just shifts the point estimate.
    import scipy.stats as stats
    z = stats.norm.ppf(1 - alpha / 2)
    
    ci_lower = adj_estimate - z * se
    ci_upper = adj_estimate + z * se
    
    return {
        'grid_cd': cd_mesh,
        'grid_cy': cy_mesh,
        'adj_estimate': adj_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'baseline_estimate': estimate,
        'baseline_se': se,
        'rho': rho,
        'alpha': alpha,
        'benchmarks': benchmark_covariates
    }


def estimate_empirical_benchmarks(data, outcome_var, treatment_var, node_variable_map, adjustment_set, random_state=43):
    """
    Estimates the partial R^2 for both treatment and outcome for each node in the adjustment set.
    """
    benchmarks = {}
    
    # Extract all controls W
    all_w = []
    for node in adjustment_set:
        all_w.extend(node_variable_map.get(node, []))
    all_w = [c for c in all_w if c in data.columns]
    
    Y = data[outcome_var]
    T = data[treatment_var]
    
    # Full models
    rf_Y_full = RandomForestRegressor(random_state=random_state, n_jobs=-1, max_depth=3)
    rf_T_full = RandomForestRegressor(random_state=random_state, n_jobs=-1, max_depth=3)
    
    X_full = data[all_w]
    rf_Y_full.fit(X_full, Y)
    rf_T_full.fit(X_full, T)
    
    mse_Y_full = np.mean((Y - rf_Y_full.predict(X_full))**2)
    mse_T_full = np.mean((T - rf_T_full.predict(X_full))**2)
    
    for node in adjustment_set:
        node_vars = node_variable_map.get(node, [])
        node_vars = [v for v in node_vars if v in data.columns]
        
        if not node_vars:
            continue
            
        w_minus_j = [c for c in all_w if c not in node_vars]
        if not w_minus_j:
            # If removing this node leaves no controls, the partial R2 is just the full R2
            cd = 1 - mse_T_full / np.var(T)
            cy = 1 - mse_Y_full / np.var(Y)
            benchmarks[node] = (max(0, cd), max(0, cy))
            continue
            
        X_reduced = data[w_minus_j]
        
        rf_Y_reduced = RandomForestRegressor(random_state=random_state, n_jobs=-1, max_depth=3)
        rf_T_reduced = RandomForestRegressor(random_state=random_state, n_jobs=-1, max_depth=3)
        
        rf_Y_reduced.fit(X_reduced, Y)
        rf_T_reduced.fit(X_reduced, T)
        
        mse_Y_reduced = np.mean((Y - rf_Y_reduced.predict(X_reduced))**2)
        mse_T_reduced = np.mean((T - rf_T_reduced.predict(X_reduced))**2)
        
        # Partial R^2 = (MSE_reduced - MSE_full) / MSE_reduced
        cd = (mse_T_reduced - mse_T_full) / mse_T_reduced if mse_T_reduced > 0 else 0
        cy = (mse_Y_reduced - mse_Y_full) / mse_Y_reduced if mse_Y_reduced > 0 else 0
        
        benchmarks[node] = (max(0, cd), max(0, cy))
        
    return benchmarks


def run_sensitivity_analysis(estimator_instance):
    """
    Runs sensitivity analysis on the fitted estimator.
    (Kept for compatibility with original code)
    """
    if hasattr(estimator_instance, 'est') and hasattr(estimator_instance.est, 'sensitivity_summary'):
        summary_str = str(estimator_instance.est.sensitivity_summary())
        
        rv_theta = None
        rv_ci = None
        
        lines = summary_str.split('\n')
        for i, line in enumerate(lines):
            if "Robustness Value (Theta)" in line and "Robustness Value (CI)" in line:
                for j in range(1, 4):
                    if i + j < len(lines):
                        vals = re.findall(r'0\.\d+', lines[i+j])
                        if len(vals) >= 2:
                            rv_theta = float(vals[0])
                            rv_ci = float(vals[1])
                            break
                break
                
        return {'Robustness Value (Theta)': rv_theta, 'Robustness Value (CI)': rv_ci}
    else:
        return None
