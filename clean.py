import argparse
import os
import glob
import shutil
import tomllib
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Purge outputs for MBA Tanzania Causal Pipeline")
    parser.add_argument('--config', type=str, required=True, help="Path to TOML configuration file")
    parser.add_argument('--experiment', type=str, required=True, help="Name of the experiment to clean")
    parser.add_argument('--data', action='store_true', help="Purge data preprocessing outputs")
    parser.add_argument('--identify', action='store_true', help="Purge identification outputs")
    parser.add_argument('--estimate', action='store_true', help="Purge estimation outputs")
    parser.add_argument('--plot', action='store_true', help="Purge plot outputs")
    parser.add_argument('--robustness', action='store_true', help="Purge robustness check outputs")
    return parser.parse_args()

def remove_files(pattern):
    """Finds and removes files matching a glob pattern."""
    files = glob.glob(pattern)
    for file_path in files:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Removed directory: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

def main():
    args = parse_args()
    
    with open(args.config, "rb") as f:
        config_data = tomllib.load(f)

    if 'experiments' not in config_data or args.experiment not in config_data['experiments']:
        raise ValueError(f"Experiment '{args.experiment}' not found in {args.config}")
        
    exp_config = config_data['experiments'][args.experiment]
    global_config = config_data.get('global', {})
    
    treatment_node = exp_config.get('treatment_node', global_config.get('treatment_node', False))
    outcome_node = exp_config.get('outcome_node', global_config.get('outcome_node', False))
    fertilizer = global_config.get('treatment_node', None)[-1:]
    
    dag_basename = os.path.splitext(os.path.basename(exp_config.get('dag')))[0]
    adj_set_type = exp_config.get('adjustment_set', 'OPTIMAL_MINIMUM_ADJ')
    estimator_names = exp_config.get('estimators', [])
    
    print(f"=== Cleaning Experiment: {args.experiment} ===")
    
    if args.data:
        print("\n--- Purging Data Preprocessing ---")
        # DATA_VERSION = "260116" used in main.py
        remove_files("outputs/cache/data_preprocessed_260116-transformed.pkl")
        
    if args.identify:
        print("\n--- Purging Identification ---")
        adj_cache_path = f"outputs/adjustment_sets/adj_{dag_basename}_{treatment_node}_{outcome_node}_{adj_set_type}.json"
        remove_files(adj_cache_path)
        
    if args.estimate:
        print("\n--- Purging Estimation ---")
        # To determine how many adjustment sets we had, we would check the adj_cache_path
        # But for cleaning, we can just glob wildcard the suffix `*`
        for estimator_name in estimator_names:
            # Model caches
            remove_files(f"outputs/estimators/{estimator_name}_{dag_basename}_{treatment_node}_{outcome_node}_{adj_set_type}*.pkl")
            # Estimate CSVs
            remove_files(f"outputs/estimates/{fertilizer}-{args.experiment}-{estimator_name}*.csv")

    if args.plot:
        print("\n--- Purging Plots ---")
        # We can just remove the whole folder for this experiment
        plot_dir = f"plots/{fertilizer}-{args.experiment}"
        if os.path.exists(plot_dir):
            remove_files(f"{plot_dir}/*")
            
    if args.robustness:
        print("\n--- Purging Robustness ---")
        # We can just remove the whole folder for this experiment
        robustness_dir = f"outputs/robustness/{fertilizer}-{args.experiment}"
        if os.path.exists(robustness_dir):
            remove_files(f"{robustness_dir}/*")
            
    print("\nCleaning complete.")

if __name__ == "__main__":
    main()
