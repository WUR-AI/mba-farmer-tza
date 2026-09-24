import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import cross_val_predict, cross_val_score, GroupKFold

from src.estimators.base import CausalEstimate

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

    total_count = sum(c.count for c in cates.values() if c.count is not None)
    total_count = total_count if total_count > 0 else None

    return CausalEstimate(
        value=ate, std_error=ate_se, p_value=p_value, 
        ci_lower=ci_lower, ci_upper=ci_upper, estimator_instance=estimator_instance,
        count=total_count
    )

def _build_soil_matrix(transformed_data, treatment_name=None):
    sandy_series = transformed_data['soil_sandy'].to_numpy()
    X_soil = np.vstack([~sandy_series.astype(bool), sandy_series]).T
    x_names = ['Non-sandy', 'Sandy']
    return X_soil, x_names

def _build_time_matrix(transformed_data, treatment_name=None):
    fertilizer_type = 'N' if 'N' in treatment_name else 'P'
    basal_col = f'fert{fertilizer_type}_basal'
    tops_cols = [
        f'fert{fertilizer_type}_kneehigh', f'fert{fertilizer_type}_56leaves', 
        f'fert{fertilizer_type}_810leaves', f'fert{fertilizer_type}_tasselingsilking'
    ]
    
    has_basal = transformed_data[basal_col] > 0
    n_tops = (transformed_data[tops_cols] > 0).sum(axis=1)
    
    strategy = pd.Series(index=transformed_data.index, dtype=str)
    
    if fertilizer_type == 'N':
        strategy[~has_basal] = "Top-Only"
        strategy[has_basal & (n_tops < 1)] = "Basal-Only"
        strategy[has_basal & (n_tops == 1)] = "Basal+Single"
        strategy[has_basal & (n_tops > 1)] = "Basal+Split"
        x_names_time = ["Top-Only", "Basal-Only", "Basal+Single", "Basal+Split"]
    else:
        strategy[~has_basal] = "Top-Only"
        strategy[has_basal] = "Basal-Only"
        x_names_time = ["Top-Only", "Basal-Only"]
        
    for cat in x_names_time:
        transformed_data[cat] = (strategy == cat).astype(float)
        
    X_time = transformed_data[x_names_time].to_numpy()
    return X_time, x_names_time

def _build_legume_matrix(transformed_data, treatment_name=None):
    is_legume = transformed_data['history_maizelegume'].to_numpy().astype(bool)
    transformed_data['Non-legume'] = (~is_legume).astype(float)
    transformed_data['Legume'] = is_legume.astype(float)
    x_names_legume = ['Non-legume', 'Legume']
    X_legume = transformed_data[x_names_legume].to_numpy()
    return X_legume, x_names_legume

def _build_pfert_matrix(transformed_data, treatment_name=None):
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
