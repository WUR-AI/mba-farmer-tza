import argparse
import pickle
import os
import json
import pandas as pd
import tomllib
import scipy.stats as st

from src.data.preprocessing import load_and_preprocess_data, spatial_demean_data
from src.identification.adjustment import identify_adjustment_set
from src.estimators import get_estimator
from src.plots.diagnostics import plot_gps_support
from src.plots.estimates import plot_dose_response_curve, plot_average_marginal_effect, plot_point_estimates, plot_ucc_contours
from src.estimators.base import fit_estimator
from src.robustness.sensitivity import run_sensitivity_analysis, compute_ucc_sensitivity
from src.robustness.placebo import run_random_placebo_treatment, run_pretreatment_placebo_outcome
import geopandas as gpd

def parse_args():
    parser = argparse.ArgumentParser(description="MBA Tanzania Causal Pipeline")
    parser.add_argument('--config', type=str, required=True, help="Path to TOML configuration file")
    parser.add_argument('--experiment', type=str, required=True, help="Name of the experiment to run")
    parser.add_argument('--data', action='store_true', help="Run data preprocessing step")
    parser.add_argument('--identify', action='store_true', help="Run identification step")
    parser.add_argument('--estimate', action='store_true', help="Run estimation step")
    parser.add_argument('--plot', action='store_true', help='Generate plots from saved results')
    parser.add_argument('--robustness', action='store_true', help="Run robustness checks")
    return parser.parse_args()

def load_config(config_path, experiment_name):
    with open(config_path, "rb") as f:
        if hasattr(tomllib, 'load'):
            config_data = tomllib.load(f)
        else:
            config_data = tomllib.loads(f.read().decode("utf-8"))

    if 'experiments' not in config_data or experiment_name not in config_data['experiments']:
        raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")
        
    exp_config = config_data['experiments'][experiment_name]
    global_config = config_data.get('global', {})
    return config_data, exp_config, global_config

def run_data_preprocessing(data_cache_path, data_path, data_version, treatment_var, outcome_var, conditional_vars):
    print("\n--- Running Data Preprocessing ---")
    if os.path.exists(data_cache_path):
        print(f"Loading preprocessed data from {data_cache_path}...")
        data = pd.read_pickle(data_cache_path)
    else:
        data = load_and_preprocess_data(data_path, data_version=data_version)
        os.makedirs(os.path.dirname(data_cache_path), exist_ok=True)
        data.to_pickle(data_cache_path)
        print(f"Saved preprocessed data to {data_cache_path}")

    assert set([treatment_var, outcome_var]+conditional_vars).issubset(set(data.columns))
    data = data.dropna(subset=[treatment_var, outcome_var])
    print("Data summary:")
    print(f"  Rows: {len(data)}")
    print(f"  Outcome summary:\n{data[outcome_var].describe()}")
    print(f"  Treatment summary:\n{data[treatment_var].describe()}")

def run_identification(dag_path, treatment_node, outcome_node, conditional_nodes, adj_set_type, adj_cache_path):
    print("\n--- Running Identification ---")
    with open(dag_path, 'r') as f:
        edges_list = json.load(f)
    assert set([treatment_node, outcome_node]+conditional_nodes).issubset({item for sublist in edges_list for item in sublist})
    
    if os.path.exists(adj_cache_path):
        print(f"Loading adjustment set from {adj_cache_path}...")
        with open(adj_cache_path, 'r') as f:
            adjustment_set = json.load(f)
    else:
        adjustment_set = identify_adjustment_set(edges_list, treatment_node, outcome_node, adj_set_type, conditional_nodes)
        os.makedirs(os.path.dirname(adj_cache_path), exist_ok=True)
        with open(adj_cache_path, 'w') as f:
            json.dump(adjustment_set, f, indent=4)
        print(f"Saved adjustment set to {adj_cache_path}")
        
    print("Adjustment Set:")
    print(adjustment_set)

def load_adjustment_sets(adj_cache_path):
    if not os.path.exists(adj_cache_path):
        raise FileNotFoundError(f"Adjustment set not found at {adj_cache_path}. Run --identify first.")
    with open(adj_cache_path, 'r') as f:
        adjustment_sets_raw = json.load(f)
        
    if len(adjustment_sets_raw) > 0 and isinstance(adjustment_sets_raw[0], str):
        return [adjustment_sets_raw]
    return adjustment_sets_raw

def load_experiment_data(data_path, data_version, treatment_var, outcome_var, node_variable_map):
    print("Loading data for estimation...")
    data = load_and_preprocess_data(data_path=data_path, data_version=data_version)
    
    raw_data = data.copy()
    data = data[data[treatment_var] > 0]
    data = data.dropna(subset=[outcome_var, treatment_var])
    coords = raw_data.loc[data.index, ['lat', 'lon']].copy()
    points = gpd.GeoSeries(gpd.points_from_xy(coords['lon'], coords['lat'], crs='EPSG:32736'), index=coords.index)
    
    return data, raw_data, coords, points

def run_estimation(args, exp_config, global_config, data, raw_data, points, adjustment_sets_list, node_variable_map, dag_basename, adj_set_type, treatment_node, outcome_node, treatment_var, outcome_var, fertilizer):
    print("\n--- Running Estimation ---")
    estimator_names = exp_config.get('estimators', [])
    assert len(estimator_names) > 0
    random_seed = global_config.get('random_seed', 43)

    for i, adjustment_set in enumerate(adjustment_sets_list):
        control_columns = [item for node in adjustment_set for item in node_variable_map[node]]
        current_data = data[[outcome_var, treatment_var] + control_columns].copy().dropna()

        shared_model_y = None
        shared_model_t = None
        
        for estimator_name in estimator_names:
            suffix = f"_adj_{i+1}" if len(adjustment_sets_list) > 1 else "_adj_1"
            model_cache_path = f"outputs/estimators/{estimator_name}_{dag_basename}_{treatment_node}_{outcome_node}_{adj_set_type}{suffix}.pkl"
            
            print(f"Fitting {estimator_name} for adjustment set {i+1}/{len(adjustment_sets_list)}...")
            EstimatorClass = get_estimator(estimator_name)
            
            model, current_data, shared_model_y, shared_model_t = fit_estimator(
                estimator_name=estimator_name,
                EstimatorClass=EstimatorClass,
                current_data=current_data,
                raw_data=raw_data,
                adjustment_set=adjustment_set,
                treatment_node=treatment_node,
                outcome_node=outcome_node,
                outcome_var=outcome_var,
                treatment_var=treatment_var,
                random_seed=random_seed,
                shared_model_y=shared_model_y,
                shared_model_t=shared_model_t,
                points=points
            )
            
            os.makedirs(os.path.dirname(model_cache_path), exist_ok=True)
            with open(model_cache_path, 'wb') as f:
                pickle.dump(model, f)
            
            if getattr(model, 'is_first_pass', False) and hasattr(model, 'diagnostics') and model.diagnostics is not None:
                plot_filename = f"plots/{fertilizer}-{args.experiment}/fertilizer-gps-support-strata.pdf"
                print(f"Generating GPS support plot: {plot_filename}")
                plot_gps_support(model.diagnostics, current_data, raw_data, treatment_var, fertilizer, plot_filename)
            
            print(f"\n--- Results for {estimator_name} (Adj {i+1}) ---")
            print(f"{'Parameter':<25} | {'Mean Value':<12} | {'Std Error':<10} | {'P-Value':<10} | {'CI Lower':<10} | {'CI Upper':<10}")
            print("-" * 90)
            
            causal_records = []
            
            try:
                ate = model.estimate_ate()
                print(f"{'ATE':<25} | {ate.value:<12.4f} | {ate.std_error:<10.4f} | {ate.p_value:<10.4f} | {ate.ci_lower:<10.4f} | {ate.ci_upper:<10.4f}")
                causal_records.append({
                    'Parameter': 'ATE', 'Mean Value': ate.value, 'Std Error': ate.std_error,
                    'P-Value': ate.p_value, 'CI Lower': ate.ci_lower, 'CI Upper': ate.ci_upper, 'Count': ate.count
                })
            except NotImplementedError:
                print(f"{'ATE':<25} | Not implemented")
            except Exception as e:
                print(f"{'ATE':<25} | Error: {str(e)[:30]}")
            
            if 'SOIL_RATE' in estimator_name or estimator_name == 'RF_PREDICTIVE':
                try:
                    dose_resp_df = model.estimate_dose_response()
                    csv_path = f"outputs/estimates/{fertilizer}-{args.experiment}-{estimator_name}{suffix}-dose_response.csv"
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    dose_resp_df.to_csv(csv_path)
                    
                    ame_df = model.estimate_average_marginal_effect()
                    ame_csv = f"outputs/estimates/{fertilizer}-{args.experiment}-{estimator_name}{suffix}-marginal_effect.csv"
                    ame_df.to_csv(ame_csv)
                    print(f"Saved dose response and marginal effect outputs for {estimator_name}.")
                except Exception as e:
                    print(f"Failed to generate dose response/marginal effect for {estimator_name}: {e}")
                
            try:
                cates = model.estimate_cate()
                if isinstance(cates, dict):
                    for group, cate in cates.items():
                        param_name = f"CATE: {group}"[:25]
                        print(f"{param_name:<25} | {cate.value:<12.4f} | {cate.std_error:<10.4f} | {cate.p_value:<10.4f} | {cate.ci_lower:<10.4f} | {cate.ci_upper:<10.4f}")
                        causal_records.append({
                            'Parameter': f"CATE: {group}", 'Mean Value': cate.value, 'Std Error': cate.std_error,
                            'P-Value': cate.p_value, 'CI Lower': cate.ci_lower, 'CI Upper': cate.ci_upper, 'Count': cate.count
                        })
            except NotImplementedError:
                print(f"{'CATE':<25} | Not implemented")
            except Exception as e:
                print(f"{'CATE':<25} | Error: {str(e)[:30]}")
                
            if causal_records:
                causal_df = pd.DataFrame(causal_records)
                causal_csv = f"outputs/estimates/{fertilizer}-{args.experiment}-{estimator_name}{suffix}-causal_estimates.csv"
                os.makedirs(os.path.dirname(causal_csv), exist_ok=True)
                causal_df.to_csv(causal_csv, index=False)
            print("\n")
            
    print("Model estimation complete.")

def generate_plots(args, exp_config, data, adjustment_sets_list, fertilizer, treatment_var):
    print("\n--- Generating Plots ---")
    estimator_names = exp_config.get('estimators', [])
    T_absolute = data[treatment_var]
    
    for i, _ in enumerate(adjustment_sets_list):
        suffix = f"_adj_{i+1}" if len(adjustment_sets_list) > 1 else "_adj_1"
        for estimator_name in estimator_names:
            if 'SOIL_RATE' in estimator_name or estimator_name == 'RF_PREDICTIVE':
                try:
                    dose_csv = f"outputs/estimates/{fertilizer}-{args.experiment}-{estimator_name}{suffix}-dose_response.csv"
                    if os.path.exists(dose_csv):
                        dose_resp_df = pd.read_csv(dose_csv, index_col=['Soil type', 'T'])
                        pdf_path = f"plots/{fertilizer}-{args.experiment}/{estimator_name}{suffix}-dose_response.pdf"
                        plot_dose_response_curve(dose_resp_df, pdf_path, fertilizer)
                        print(f"Generated {pdf_path}")
                        
                    ame_csv = f"outputs/estimates/{fertilizer}-{args.experiment}-{estimator_name}{suffix}-marginal_effect.csv"
                    if os.path.exists(ame_csv):
                        ame_df = pd.read_csv(ame_csv, index_col=['Soil type', 'T'])
                        ame_pdf = f"plots/{fertilizer}-{args.experiment}/{estimator_name}{suffix}-marginal_effect.pdf"
                        plot_average_marginal_effect(ame_df, T_absolute, ame_pdf, fertilizer)
                        print(f"Generated {ame_pdf}")
                except Exception as e:
                    print(f"Failed plotting RATE estimates for {estimator_name} {suffix}: {e}")
            else:
                try:
                    causal_csv = f"outputs/estimates/{fertilizer}-{args.experiment}-{estimator_name}{suffix}-causal_estimates.csv"
                    if os.path.exists(causal_csv):
                        causal_df = pd.read_csv(causal_csv)
                        causal_df['Estimator'] = estimator_name
                        pdf_path = f"plots/{fertilizer}-{args.experiment}/{estimator_name}{suffix}-point_estimates.pdf"
                        plot_point_estimates(causal_df, pdf_path, fertilizer)
                        print(f"Generated {pdf_path}")
                except Exception as e:
                    print(f"Failed plotting point estimates for {estimator_name} {suffix}: {e}")
                    
            try:
                import glob
                ucc_files = glob.glob(f"outputs/robustness/{fertilizer}-{args.experiment}/{estimator_name}{suffix}_*-ucc_data.pkl")
                if not ucc_files:
                    ucc_files = glob.glob(f"outputs/robustness/{fertilizer}-{args.experiment}/{estimator_name}{suffix}-ucc_data.pkl")
                    
                for ucc_data_path in ucc_files:
                    try:
                        with open(ucc_data_path, 'rb') as f_ucc:
                            ucc_data = pickle.load(f_ucc)
                        
                        basename = os.path.basename(ucc_data_path)
                        prefix = f"{estimator_name}{suffix}_"
                        if basename.startswith(prefix):
                            group_str = basename[len(prefix):].replace('-ucc_data.pkl', '')
                            title = f"UCC Sensitivity for {estimator_name} ({group_str})"
                            ucc_pdf = f"plots/{fertilizer}-{args.experiment}/{estimator_name}{suffix}_{group_str}-ucc_contour.pdf"
                        else:
                            title = f"UCC Sensitivity for {estimator_name}"
                            ucc_pdf = f"plots/{fertilizer}-{args.experiment}/{estimator_name}{suffix}-ucc_contour.pdf"
                            
                        plot_ucc_contours(ucc_data, save_path=ucc_pdf, title=title)
                    except Exception as e:
                        print(f"Failed to plot UCC contour for {ucc_data_path}: {e}")
            except Exception as e:
                print(f"Failed plotting UCC contours for {estimator_name} {suffix}: {e}")

def run_robustness(args, exp_config, global_config, data, raw_data, points, adjustment_sets_list, node_variable_map, dag_basename, adj_set_type, treatment_node, outcome_node, treatment_var, outcome_var, fertilizer):
    print("\n--- Running Robustness Checks ---")
    robustness_tests = exp_config.get('robustness', {})
    if isinstance(robustness_tests, list):
        robustness_tests = {k: exp_config.get('estimators', []) for k in robustness_tests}
    
    estimator_names = exp_config.get('estimators', [])
    random_seed = global_config.get('random_seed', 43)
    robustness_dir = f"outputs/robustness/{fertilizer}-{args.experiment}/"
    os.makedirs(robustness_dir, exist_ok=True)
    sensitivity_records = []

    for i, adjustment_set in enumerate(adjustment_sets_list):
        control_columns = [item for node in adjustment_set for item in node_variable_map[node]]
        current_data = data[[outcome_var, treatment_var] + control_columns].copy().dropna()

        for estimator_name in estimator_names:
            suffix = f"_adj_{i+1}" if len(adjustment_sets_list) > 1 else "_adj_1"
            model_cache_path = f"outputs/estimators/{estimator_name}_{dag_basename}_{treatment_node}_{outcome_node}_{adj_set_type}{suffix}.pkl"
            
            if os.path.exists(model_cache_path):
                print(f"\n--- Robustness for {estimator_name} (Adj {i+1}) ---")
                with open(model_cache_path, 'rb') as f:
                    model = pickle.load(f)
                    
                if estimator_name in robustness_tests.get("SensitivityAnalysis", []) and "RATE" not in estimator_name:
                    print("Running Sensitivity Analysis...")
                    summary = run_sensitivity_analysis(model)
                    if summary:
                        summary['Estimator'] = estimator_name
                        summary['Adj_Set'] = i+1
                        sensitivity_records.append(summary)
                        
                if estimator_name in robustness_tests.get("UnobservedCommonConfounder", []):
                    print("Running UCC Sensitivity Analysis...")
                    try:
                        cates = model.estimate_cate()
                        if isinstance(cates, dict):
                            cates_to_run = cates.copy()
                            try:
                                ate_obj = model.estimate_ate()
                                if ate_obj is not None:
                                    cates_to_run['ATE'] = ate_obj
                            except Exception as e:
                                print(f"Failed to append ATE for {estimator_name}: {e}")
                                
                            for group, estimate_obj in cates_to_run.items():
                                try:
                                    if estimate_obj.count is None:
                                        print(f"Skipping UCC for {group} due to missing count")
                                        continue
                                        
                                    ucc_data = compute_ucc_sensitivity(
                                        estimate=estimate_obj.value, se=estimate_obj.std_error,
                                        count=estimate_obj.count, benchmark_covariates=None
                                    )
                                    safe_group = group.replace(' ', '_').replace('/', '_').replace(':', '_')
                                    ucc_data_path = f"{robustness_dir}/{estimator_name}{suffix}_{safe_group}-ucc_data.pkl"
                                    with open(ucc_data_path, 'wb') as f_ucc:
                                        pickle.dump(ucc_data, f_ucc)
                                    print(f"Saved UCC data to {ucc_data_path}")
                                except Exception as e:
                                    print(f"Failed to run UCC Sensitivity Analysis for {estimator_name} {suffix} (group {group}): {e}")
                    except Exception as e:
                        print(f"Failed to run UCC Sensitivity Analysis for {estimator_name} {suffix}: {e}")
                    
                if estimator_name in robustness_tests.get("PlaceboTest", []):
                    print("Running Random Placebo Treatment (RPT)...")
                    EstimatorClass = get_estimator(estimator_name)
                    ate_df, cate_df, dose_results = run_random_placebo_treatment(
                        N_PLACEBO_RUNS=100, estimator_name=estimator_name, EstimatorClass=EstimatorClass,
                        current_data=current_data, raw_data=raw_data, adjustment_set=adjustment_set,
                        treatment_node=treatment_node, outcome_node=outcome_node, outcome_var=outcome_var,
                        treatment_var=treatment_var, random_seed=random_seed, shared_model_y=None,
                        shared_model_t=None, points=points
                    )
                    
                    if not ate_df.empty:
                        ate_df.to_csv(f"{robustness_dir}/{estimator_name}{suffix}-RPT_ATE.csv", index=False)
                    if not cate_df.empty:
                        cate_df.to_csv(f"{robustness_dir}/{estimator_name}{suffix}-RPT_CATE.csv", index=False)
                    if dose_results:
                        with open(f"{robustness_dir}/{estimator_name}{suffix}-RPT_DOSE.pkl", 'wb') as f:
                            pickle.dump(dose_results, f)
                            
                    print(f"\n--- RPT Summary for {estimator_name} ---")
                    if not ate_df.empty:
                        mean_val = ate_df['Value'].mean()
                        std_val = ate_df['Value'].std()
                        _, pval = st.ttest_1samp(ate_df['Value'].dropna(), 0.0)
                        print(f"ATE Placebo Estimate: {mean_val:.4f} (Std: {std_val:.4f}, Mean=0 P-Value: {pval:.4f})")
                    if not cate_df.empty:
                        for group, group_df in cate_df.groupby('Group'):
                            mean_val = group_df['Value'].mean()
                            std_val = group_df['Value'].std()
                            _, pval = st.ttest_1samp(group_df['Value'].dropna(), 0.0)
                            print(f"CATE [{group}] Placebo Estimate: {mean_val:.4f} (Std: {std_val:.4f}, Mean=0 P-Value: {pval:.4f})")
                    print("-------------------\n")
                            
                if estimator_name in robustness_tests.get("PreTreatmentPlacebo", []):
                    print("Running Pre-treatment Placebo Outcome (PTPO)...")
                    EstimatorClass = get_estimator(estimator_name)
                    dag_path = os.path.join("config", "dags", exp_config.get('dag'))
                    with open(dag_path, 'r') as f:
                        dag_edges = json.load(f)
                    causal_ancestor_nodes = [edge[0] for edge in dag_edges if edge[1] == treatment_node]
                    
                    pre_treatment_vars = [v for node in causal_ancestor_nodes for v in node_variable_map.get(node, [])]
                    pre_treatment_vars = [v for v in pre_treatment_vars if v in current_data.columns and ("season" not in v or v.startswith("0to100"))]
                    
                    ptpo_df = run_pretreatment_placebo_outcome(
                        pre_treatment_vars=pre_treatment_vars, estimator_name=estimator_name, EstimatorClass=EstimatorClass,
                        current_data=current_data, raw_data=raw_data, adjustment_set=adjustment_set,
                        treatment_node=treatment_node, outcome_node=outcome_node, treatment_var=treatment_var,
                        random_seed=random_seed, shared_model_y=None, shared_model_t=None, points=points
                    )
                    if not ptpo_df.empty:
                        ptpo_df.to_csv(f"{robustness_dir}/{estimator_name}{suffix}-PTPO_results.csv", index=False)

    if sensitivity_records:
        pd.DataFrame(sensitivity_records).to_csv(f"{robustness_dir}/sensitivity_results.csv", index=False)
        print(f"Saved sensitivity results to {robustness_dir}/sensitivity_results.csv")

def main():
    args = parse_args()
    config_data, exp_config, global_config = load_config(args.config, args.experiment)
    
    treatment_node = global_config.get('treatment_node', False)
    outcome_node = global_config.get('outcome_node', False)
    conditional_nodes = global_config.get('conditional_nodes', [])
    
    print(f"=== Running Experiment: {args.experiment} ===")
    dag_path = f"config/dags/{exp_config.get('dag')}"
    dag_basename = os.path.splitext(os.path.basename(exp_config.get('dag')))[0]
    
    DATA_PATH = "/data/mba-tza"
    DATA_VERSION = "260116"
    
    fertilizer = treatment_node[-1:] if treatment_node else 'N'
    assert fertilizer in ["N", "P"]
    
    with open('config/node_variable_map.json', 'r') as f:
        node_variable_map = json.load(f)
        
    outcome_var = node_variable_map[outcome_node][0]
    treatment_var = node_variable_map[treatment_node][0]
    conditional_vars = [node_variable_map[node][0] for node in conditional_nodes]
    
    data_cache_path = f"outputs/cache/data_preprocessed_{DATA_VERSION}-transformed.pkl"
    adj_set_type = exp_config.get('adjustment_set', 'OPTIMAL_MINIMUM_ADJ')
    adj_cache_path = f"outputs/adjustment_sets/adj_{dag_basename}_{treatment_node}_{outcome_node}_{adj_set_type}.json"

    if args.data:
        run_data_preprocessing(data_cache_path, DATA_PATH, DATA_VERSION, treatment_var, outcome_var, conditional_vars)

    if args.identify:
        run_identification(dag_path, treatment_node, outcome_node, conditional_nodes, adj_set_type, adj_cache_path)

    if args.estimate or args.plot or args.robustness:
        adjustment_sets_list = load_adjustment_sets(adj_cache_path)
        data, raw_data, coords, points = load_experiment_data(DATA_PATH, DATA_VERSION, treatment_var, outcome_var, node_variable_map)
        
        if args.estimate:
            run_estimation(args, exp_config, global_config, data, raw_data, points, adjustment_sets_list, node_variable_map, dag_basename, adj_set_type, treatment_node, outcome_node, treatment_var, outcome_var, fertilizer)
            
        if args.plot:
            generate_plots(args, exp_config, data, adjustment_sets_list, fertilizer, treatment_var)
            
        if args.robustness:
            run_robustness(args, exp_config, global_config, data, raw_data, points, adjustment_sets_list, node_variable_map, dag_basename, adj_set_type, treatment_node, outcome_node, treatment_var, outcome_var, fertilizer)

if __name__ == "__main__":
    main()
