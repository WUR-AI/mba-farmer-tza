import numpy as np
import pandas as pd
import copy

def run_placebo_test(estimator_instance, n_runs=10):
    """
    Runs a placebo test by randomly shuffling the treatment variable and refitting the model.
    """
    original_model = estimator_instance.est
    
    # We need to re-fit the exact same estimator, but with shuffled T.
    # Note: For this to work robustly, we could just copy the estimator instance, 
    # but some internal states might be tied up.
    
    # We will just yield a summary or return a boolean if the effect becomes insignificant.
    # Since doing a full loop takes a lot of time, we'll do n_runs (default 10).
    
    Y = estimator_instance.outcome.values
    W = estimator_instance.controls.values if estimator_instance.controls is not None else None
    
    results = []
    
    for _ in range(n_runs):
        # We need the original T
        if hasattr(estimator_instance, 'T_absolute'):
            T_original = estimator_instance.T_absolute
        else:
            T_original = estimator_instance.treatment.values
            
        T_placebo = np.random.choice(T_original, size=len(T_original), replace=True)
        
        # Fit a new clone of the estimator's base est
        # EconML models don't clone easily, but we can try sklearn.base.clone
        from sklearn.base import clone
        try:
            placebo_est = clone(original_model)
        except:
            import copy
            placebo_est = copy.deepcopy(original_model)
            
        if hasattr(estimator_instance, 'cubic_spline_transformer'):
            from src.estimators.base import build_treatment_featurizer
            import geopandas as gpd
            # mock points for featurizer if needed, this is tricky to extract
            # so we just use the original treatment_featurizer which has spatial_avg fixed
            placebo_est = placebo_est.fit(
                Y=Y, T=estimator_instance.est.featurizer.transform(T_placebo.reshape(-1, 1)) if hasattr(estimator_instance.est, 'featurizer') and estimator_instance.est.featurizer is not None else T_placebo, 
                W=W, X=estimator_instance.X,
                inference='statsmodels'
            )
        else:
            placebo_est = placebo_est.fit(
                Y=Y, T=T_placebo, W=W, X=estimator_instance.X,
                inference='statsmodels'
            )
            
        # Get ATE
        ate = placebo_est.ate(estimator_instance.X)
        results.append(ate)
        
    return np.mean(results), np.std(results)
