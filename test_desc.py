def generate_descriptive_statistics():
    import pickle
    import json
    import pandas as pd
    import numpy as np
    from src.estimators.helpers import _build_time_matrix, _build_legume_matrix, _build_pfert_matrix, _build_soil_matrix

    with open('outputs/cache/data_preprocessed_260116-transformed.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('config/node_variable_map.json', 'r') as f:
        node_map = json.load(f)

    # Nodes for N analysis
    nodes = ['fertilizerAmountN', 'outcome', 'fertilizerAmountP', 'fertilizerTiming', 'fieldHistory', 'otherManagement', 'soil', 'variety', 'weatherSeason']
    
    rows = []
    for node in nodes:
        vars_in_node = [v for v in node_map.get(node, []) if v in data.columns]
        for var in vars_in_node:
            s = data[var]
            rows.append({
                'Node': node,
                'Variable': var,
                'Average': round(s.mean(), 2),
                'Std': round(s.std(), 2),
                'Min - Max': f"{round(s.min(), 2)} - {round(s.max(), 2)}"
            })
            
    df_desc = pd.DataFrame(rows).set_index(['Node', 'Variable'])
    print("--- Descriptive Statistics (N Analysis Variables) ---")
    print(df_desc.to_markdown())
    print("\n")

    groups = {}
    
    # Soil groups
    X_soil, soil_names = _build_soil_matrix(data.copy())
    groups['Soil: Non-sandy'] = X_soil[:, 0].astype(bool)
    groups['Soil: Sandy'] = X_soil[:, 1].astype(bool)
    
    # N Time groups
    X_time_n, time_n_names = _build_time_matrix(data.copy(), treatment_name='N')
    for i, name in enumerate(time_n_names):
        groups[f'Time N: {name}'] = X_time_n[:, i].astype(bool)
        
    # P Time groups
    X_time_p, time_p_names = _build_time_matrix(data.copy(), treatment_name='P')
    for i, name in enumerate(time_p_names):
        groups[f'Time P: {name}'] = X_time_p[:, i].astype(bool)
        
    # Legume groups
    X_legume, legume_names = _build_legume_matrix(data.copy())
    for i, name in enumerate(legume_names):
        groups[f'Legume: {name}'] = X_legume[:, i].astype(bool)
        
    # P-Fertilizer groups
    X_pfert, pfert_names = _build_pfert_matrix(data.copy())
    for i, name in enumerate(pfert_names):
        groups[f'P-Fert: {name}'] = X_pfert[:, i].astype(bool)
        
    rows_strata = []
    
    for group_name, mask in groups.items():
        subset = data[mask]
        n_val = subset['fertN_total']
        y_val = subset['maizeprodkg_ha']
        
        rows_strata.append({
            'Strata': group_name,
            'Treatment Avg': round(n_val.mean(), 2) if len(n_val) > 0 else np.nan,
            'Treatment Std': round(n_val.std(), 2) if len(n_val) > 0 else np.nan,
            'Treatment Min-Max': f"{round(n_val.min(), 2)} - {round(n_val.max(), 2)}" if len(n_val) > 0 else "-",
            'Outcome Avg': round(y_val.mean(), 2) if len(y_val) > 0 else np.nan,
            'Outcome Std': round(y_val.std(), 2) if len(y_val) > 0 else np.nan,
            'Outcome Min-Max': f"{round(y_val.min(), 2)} - {round(y_val.max(), 2)}" if len(y_val) > 0 else "-",
            'N': len(subset)
        })
        
    df_strata = pd.DataFrame(rows_strata).set_index('Strata')
    print("--- Descriptive Statistics by Strata (N Treatment and Outcome) ---")
    print(df_strata.to_markdown())
    print("\n")

if __name__ == '__main__':
    generate_descriptive_statistics()
