import numpy as np
import pandas as pd
from tqdm import tqdm
from src.estimators.base import fit_estimator

def run_random_placebo_treatment(
    N_PLACEBO_RUNS,
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
    ate_results = []
    cate_results = []
    dose_results = []
    
    for i in tqdm(range(N_PLACEBO_RUNS), desc=f"RPT: {estimator_name}"):
        rng = np.random.default_rng(random_seed + i)
        placebo_data = current_data.copy()
        placebo_data[treatment_var] = rng.choice(
            current_data[treatment_var], size=len(current_data), replace=True
        )
        
        try:
            model, _, _, _ = fit_estimator(
                estimator_name=estimator_name,
                EstimatorClass=EstimatorClass,
                current_data=placebo_data,
                raw_data=raw_data,
                adjustment_set=adjustment_set,
                treatment_node=treatment_node,
                outcome_node=outcome_node,
                outcome_var=outcome_var,
                treatment_var=treatment_var,
                random_seed=random_seed + i,
                shared_model_y=shared_model_y,
                shared_model_t=shared_model_t,
                points=points
            )
            
            # ATE
            try:
                ate = model.estimate_ate()
                ate_results.append({
                    'Run': i,
                    'Value': ate.value,
                    'Std_Error': ate.std_error,
                    'P_Value': ate.p_value
                })
            except:
                pass
                
            # CATE
            try:
                cates = model.estimate_cate()
                if isinstance(cates, dict):
                    for k, v in cates.items():
                        cate_results.append({
                            'Run': i,
                            'Group': k,
                            'Value': v.value,
                            'Std_Error': v.std_error,
                            'P_Value': v.p_value
                        })
            except:
                pass
                
            # DOSE
            if "RATE" in estimator_name:
                try:
                    dose_results.append(model.estimate_dose_response())
                except:
                    pass
        except Exception as e:
            print(f"RPT run {i} failed: {e}")
            
    ate_df = pd.DataFrame(ate_results) if ate_results else pd.DataFrame()
    cate_df = pd.DataFrame(cate_results) if cate_results else pd.DataFrame()
    
    return ate_df, cate_df, dose_results


def run_pretreatment_placebo_outcome(
    pre_treatment_vars,
    estimator_name,
    EstimatorClass,
    current_data,
    raw_data,
    adjustment_set,
    treatment_node,
    outcome_node,
    treatment_var,
    random_seed=43,
    shared_model_y=None,
    shared_model_t=None,
    points=None
):
    results = []
    
    for placebo_var in tqdm(pre_treatment_vars, desc=f"PTPO: {estimator_name}"):
        if placebo_var not in raw_data.columns:
            continue
            
        placebo_data = current_data.copy()
        
        # Add the placebo outcome variable if it isn't in current_data already
        if placebo_var not in placebo_data.columns:
            placebo_data[placebo_var] = raw_data.loc[placebo_data.index, placebo_var]
            
        placebo_data = placebo_data.dropna(subset=[placebo_var])
        
        # To prevent feature leakage, we must explicitly drop the placebo variable 
        # from raw_data so that get_feature_lists doesn't add it back into X or W.
        placebo_raw_data = raw_data.copy()
        if placebo_var in placebo_raw_data.columns:
            placebo_raw_data = placebo_raw_data.drop(columns=[placebo_var])
        
        try:
            model, _, _, _ = fit_estimator(
                estimator_name=estimator_name,
                EstimatorClass=EstimatorClass,
                current_data=placebo_data,
                raw_data=placebo_raw_data,
                adjustment_set=adjustment_set,
                treatment_node=treatment_node,
                outcome_node=placebo_var, # Just use the variable string as the outcome node dummy
                outcome_var=placebo_var, # Treat placebo_var as the new outcome
                treatment_var=treatment_var,
                random_seed=random_seed,
                shared_model_y=shared_model_y,
                shared_model_t=shared_model_t,
                points=points
            )
            
            try:
                ate = model.estimate_ate()
                results.append({
                    'Placebo_Outcome': placebo_var,
                    'Parameter': 'ATE',
                    'Value': ate.value,
                    'Std_Error': ate.std_error,
                    'P_Value': ate.p_value
                })
            except:
                pass
                
            try:
                cates = model.estimate_cate()
                if isinstance(cates, dict):
                    for k, v in cates.items():
                        results.append({
                            'Placebo_Outcome': placebo_var,
                            'Parameter': f"CATE: {k}",
                            'Value': v.value,
                            'Std_Error': v.std_error,
                            'P_Value': v.p_value
                        })
            except:
                pass
                
        except Exception as e:
            print(f"PTPO run failed for {placebo_var}: {e}")

    return pd.DataFrame(results)
