import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def compute_ucc_sensitivity(estimator_instance, estimate, se, count, grid_cd=None, grid_cy=None, rho=1.0, alpha=0.05, benchmark_covariates=None):
    """
    Computes sensitivity bounds for Unobserved Common Confounder (UCC).
    It uses Chernozhukov's method to compute upper bounds on the bias of the estimator
    as implement by EconML
    """
    if grid_cd is None:
        grid_cd = np.linspace(0.0, 0.5, 50)
    if grid_cy is None:
        grid_cy = np.linspace(0.0, 0.5, 50)
        
    cd_mesh, cy_mesh = np.meshgrid(grid_cd, grid_cy)
    
    adj_estimate = np.zeros_like(cd_mesh)
    ci_lower = np.zeros_like(cd_mesh)
    ci_upper = np.zeros_like(cd_mesh)
    
    # Iterate over the grid because EconML's sensitivity_interval does not support vectorized c_y and c_t
    rows, cols = cd_mesh.shape
    for i in range(rows):
        for j in range(cols):
            c_d_val = cd_mesh[i, j]
            c_y_val = cy_mesh[i, j]
            
            # Point estimate bounds
            lb_theta, _ = estimator_instance.est.sensitivity_interval(
                c_y=c_y_val, c_t=c_d_val, rho=rho, alpha=alpha, interval_type='theta'
            )
            # CI bounds
            lb_ci, ub_ci = estimator_instance.est.sensitivity_interval(
                c_y=c_y_val, c_t=c_d_val, rho=rho, alpha=alpha, interval_type='ci'
            )
            
            # For rho=1.0 and positive estimate, lb_theta is the conservative shrink towards zero.
            adj_estimate[i, j] = lb_theta
            ci_lower[i, j] = lb_ci
            ci_upper[i, j] = ub_ci
    
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
