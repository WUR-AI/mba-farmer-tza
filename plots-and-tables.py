from PIL.Image import enum
import argparse
import os

import tomllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from src.plots.estimates import plot_point_estimates, plot_dose_response_curve

def overlay_exhaustive_dose_response(ax, fertilizer, base_estimator_name, experiment_name):
    exhaustive_exp = experiment_name.replace('Main', 'ExahustiveAdj')
    
    eff_path = f"outputs/adjustment_sets/adj_main-rev_fertilizerAmount{fertilizer}_outcome_BACKDOOR_EFFICIENT.json"
    exh_path = f"outputs/adjustment_sets/adj_main-rev_fertilizerAmount{fertilizer}_outcome_BACKDOOR_EXHAUSTIVE.json"
    
    if not os.path.exists(eff_path) or not os.path.exists(exh_path):
        return
        
    with open(eff_path, 'r') as f:
        eff_set = json.load(f)
    with open(exh_path, 'r') as f:
        exh_sets = json.load(f)
        
    eff_set_sorted = sorted(eff_set)
    skip_index = -1
    for i, s in enumerate(exh_sets):
        if sorted(s) == eff_set_sorted:
            skip_index = i + 1
            break
            
    first_plot = True
    for i in range(1, len(exh_sets) + 1):
        if i == skip_index:
            continue
            
        csv_path = f"outputs/estimates/{fertilizer}-{exhaustive_exp}-{base_estimator_name}_adj_{i}-dose_response.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'Soil type' in df.columns and 'T' in df.columns:
                df = df.set_index(['Soil type', 'T'])
            
            label_val = None if first_plot else "_nolegend_"
            
            plot_dose_response_curve(
                df, save_path=None, fertilizer=fertilizer, ax=ax, 
                label=label_val, plot_ci=False, plot_legend=False,
                linewidth=1.2, alpha=0.5, linestyle='-'
            )
            first_plot = False

def overlay_exhaustive_estimates(ax, fertilizer, base_estimator_name, experiment_name):
    if 'DML' not in experiment_name:
        return
        
    exhaustive_exp = experiment_name.replace('Main', 'ExahustiveAdj')
    
    eff_path = f"outputs/adjustment_sets/adj_main-rev_fertilizerAmount{fertilizer}_outcome_BACKDOOR_EFFICIENT.json"
    exh_path = f"outputs/adjustment_sets/adj_main-rev_fertilizerAmount{fertilizer}_outcome_BACKDOOR_EXHAUSTIVE.json"
    
    if not os.path.exists(eff_path) or not os.path.exists(exh_path):
        return
        
    with open(eff_path, 'r') as f:
        eff_set = json.load(f)
    with open(exh_path, 'r') as f:
        exh_sets = json.load(f)
        
    eff_set_sorted = sorted(eff_set)
    skip_index = -1
    for i, s in enumerate(exh_sets):
        if sorted(s) == eff_set_sorted:
            skip_index = i + 1
            break
            
    first_plot = True
    for i in range(1, len(exh_sets) + 1):
        if i == skip_index:
            continue
            
        csv_path = f"outputs/estimates/{fertilizer}-{exhaustive_exp}-{base_estimator_name}_adj_{i}-causal_estimates.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['Estimator'] = base_estimator_name
            if 'Count' in df.columns:
                df['Count'] = np.nan
                
            label_val = None if first_plot else "_nolegend_"
            
            plot_point_estimates(
                df, save_path=None, fertilizer=fertilizer, ax=ax, 
                label=label_val, plot_ci=True, 
                alpha=0.2,
            )
            first_plot = False

def plot_soil_time_estimates(experiment_name, save_path, plot_exhaustive=False, plot_main=True):
    print(f"\n--- Generating SOIL_TIME Figure for {experiment_name} ---")
    
    if 'DML' in experiment_name:
        estimator_name = 'DML_SOIL_TIME'
    elif 'OLS' in experiment_name:
        estimator_name = 'OLS_SOIL_TIME'
    else:
        print(f"Estimator for SOIL_TIME not determined for {experiment_name}")
        return
        
    try:
        plt.style.use('publication.mplstyle')
    except:
        pass
        
    fig, axes = plt.subplots(2, 1, figsize=(3, 4.5), height_ratios=(1, 0.5))
    
    csv_n = f"outputs/estimates/N-{experiment_name}-{estimator_name}_adj_1-causal_estimates.csv"
    if os.path.exists(csv_n):
        ax = axes[0]
        df_n = pd.read_csv(csv_n)
        df_n['Estimator'] = estimator_name
        
        if 'OLS' in estimator_name:
            df_n.loc[df_n['Parameter'] == 'CATE: N_Basal-Only_Sandy', 'Count'] = np.nan
        if plot_main:
            plot_point_estimates(df_n, save_path=None, fertilizer='N', ax=ax)
        if plot_exhaustive:
            overlay_exhaustive_estimates(ax, 'N', estimator_name, experiment_name)
        
        ax.text(
            0, 1, 'a', transform=ax.transAxes, va='center', ha='center', fontsize=8,
            fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k')
        )
        
        ax.set_xticks(np.arange(-40, 90, 10))
        ax.set_xticks(np.arange(-40, 90, 2), minor=True)
        # if 'OLS' in estimator_name:
        #     ax.set_xlim(-10, 85)
        # else:
        #     ax.set_xlim(-40, 55)
        ax.set_xlim(-40, 55)
    else:
        print(f"File not found for N: {csv_n}")
        
    csv_p = f"outputs/estimates/P-{experiment_name}-{estimator_name}_adj_1-causal_estimates.csv"
    if os.path.exists(csv_p):
        ax = axes[1]
        df_p = pd.read_csv(csv_p)
        df_p['Estimator'] = estimator_name
        if plot_main:
            plot_point_estimates(df_p, save_path=None, fertilizer='P', ax=ax, xlim=(-50, 150))
        if plot_exhaustive:
            overlay_exhaustive_estimates(ax, 'P', estimator_name, experiment_name)
        
        ax.text(
            0, 1, 'b', transform=ax.transAxes, va='center', ha='center', fontsize=8,
            fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k')
        )
        ax.set_xticks(np.arange(-40, 150, 50))
        ax.set_xticks(np.arange(-40, 150, 10), minor=True)
        ax.set_xlim(-40, 150)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels, loc='lower center', bbox_to_anchor=(0.5, -1., 0., .0), ncol=2,
            title="Soil type"
        )
    else:
        print(f"File not found for P: {csv_p}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {save_path}")

def plot_legume_pfert_estimates(experiment_name, save_path, plot_exhaustive=False, plot_main=True):
    print(f"\n--- Generating LEGUME and PFERT Figure for {experiment_name} ---")
    
    if 'DML' in experiment_name:
        estimator_pfert = 'DML_SOIL_PFERT'
        estimator_legume = 'DML_SOIL_LEGUME'
    elif 'OLS' in experiment_name:
        estimator_pfert = 'OLS_SOIL_PFERT'
        estimator_legume = 'OLS_SOIL_LEGUME'
    else:
        print(f"Estimators for LEGUME/PFERT not determined for {experiment_name}")
        return
        
    try:
        plt.style.use('publication.mplstyle')
    except:
        pass
        
    fig, axes = plt.subplots(2, 1, figsize=(3, 4), height_ratios=(1, 0.66))
    
    csv_pfert = f"outputs/estimates/N-{experiment_name}-{estimator_pfert}_adj_1-causal_estimates.csv"
    if os.path.exists(csv_pfert):
        ax = axes[0]
        df_pfert = pd.read_csv(csv_pfert)
        df_pfert['Estimator'] = estimator_pfert
        if plot_main:
            plot_point_estimates(df_pfert, save_path=None, fertilizer='N', ax=ax)
        if plot_exhaustive:
            overlay_exhaustive_estimates(ax, 'N', estimator_pfert, experiment_name)
        ax.text(
            0, 1, 'a', transform=ax.transAxes, va='center', ha='center', fontsize=8,
            fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k')
        )
        
        ax.set_xticks(np.arange(-10, 40, 10))
        ax.set_xticks(np.arange(-10, 40, 5), minor=True)
        ax.set_xlim(-10, 40)
        ax.set_title('Fertilizer type')
    else:
        print(f"File not found for PFERT: {csv_pfert}")
        
    csv_legume = f"outputs/estimates/N-{experiment_name}-{estimator_legume}_adj_1-causal_estimates.csv"
    if os.path.exists(csv_legume):
        ax = axes[1]
        df_legume = pd.read_csv(csv_legume)
        df_legume['Estimator'] = estimator_legume
        if plot_main:
            plot_point_estimates(df_legume, save_path=None, fertilizer='N', ax=ax)
        if plot_exhaustive:
            overlay_exhaustive_estimates(ax, 'N', estimator_legume, experiment_name)
        ax.text(
            0, 1, 'b', transform=ax.transAxes, va='center', ha='center', fontsize=8,
            fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k')
        )
        ax.set_xticks(np.arange(-10, 60, 10))
        ax.set_xticks(np.arange(-10, 60, 5), minor=True)
        ax.set_xlim(-10, 55)
        ax.set_title('Legume rotation')
        ALT_ADJ_LEGEND = False
        handles, labels = ax.get_legend_handles_labels()
        legend_title = "Soil type"
        if not ("DML" in estimator_legume): 
            pass
        elif not ALT_ADJ_LEGEND:
            handles = [h for n, h in enumerate(handles) if "sandy" in labels[n].lower()]
            labels = [l for n, l in enumerate(labels) if "sandy" in l.lower()]
        else:
            legend_title = None
        ax.legend(
            handles, labels, loc='lower center', bbox_to_anchor=(0.5, -1, 0., .0), ncol=2, 
            title=legend_title
        )
    else:
        print(f"File not found for LEGUME: {csv_legume}")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {save_path}")

def plot_dose_response_curves(experiment_name, save_path, plot_exhaustive=False, plot_main=True):
    print(f"\n--- Generating Dose Response Curves for {experiment_name} ---")
    
    if 'DML' in experiment_name:
        estimator_name = 'DML_SOIL_RATE_SPLINE'
    elif 'OLS' in experiment_name:
        estimator_name = 'OLS_SOIL_RATE_SPLINE'
    else:
        print(f"Estimator for RATE_SPLINE not determined for {experiment_name}")
        return
        
    try:
        plt.style.use('publication.mplstyle')
    except:
        pass
        
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    
    # Nitrogen
    csv_n = f"outputs/estimates/N-{experiment_name}-{estimator_name}_adj_1-dose_response.csv"
    if os.path.exists(csv_n):
        ax = axes[0]
        df_n = pd.read_csv(csv_n)
        if 'Soil type' in df_n.columns and 'T' in df_n.columns:
            df_n = df_n.set_index(['Soil type', 'T'])
        if plot_main:
            plot_dose_response_curve(df_n, save_path=None, fertilizer='N', ax=ax, plot_legend=False)
        if plot_exhaustive:
            overlay_exhaustive_dose_response(ax, 'N', estimator_name, experiment_name)
            
        handles, labels = ax.get_legend_handles_labels()
        legend_title = "Soil type"
        ax.legend(handles, labels, loc='upper left', title=legend_title, frameon=False)
        ax.text(
            0, 1, 'a', transform=ax.transAxes, va='center', ha='center', fontsize=8,
            fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k'), zorder=6
        )
        ax.set_xlim(38, 177)
    else:
        print(f"File not found for N: {csv_n}")
        
    # Phosphorus
    csv_p = f"outputs/estimates/P-{experiment_name}-{estimator_name}_adj_1-dose_response.csv"
    if os.path.exists(csv_p):
        ax = axes[1]
        df_p = pd.read_csv(csv_p)
        if 'Soil type' in df_p.columns and 'T' in df_p.columns:
            df_p = df_p.set_index(['Soil type', 'T'])
        if plot_main:
            plot_dose_response_curve(df_p, save_path=None, fertilizer='P', ax=ax, plot_legend=False)
        if plot_exhaustive:
            overlay_exhaustive_dose_response(ax, 'P', estimator_name, experiment_name)
            
        handles, labels = ax.get_legend_handles_labels()
        legend_title = "Soil type"
        ax.legend(handles, labels, loc='upper left', title=legend_title, frameon=False)
        ax.text(
            0, 1, 'b', transform=ax.transAxes, va='center', ha='center', fontsize=8,
            fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k'), zorder=6
        )
        # ax.set_ylim(-2000, 2000)
        ax.set_ylim(-10, 2900)
        ax.set_xlim(14, 52)
        ax.set_ylabel("")
    else:
        print(f"File not found for P: {csv_p}")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {save_path}")

def plot_other_dose_response_curves(save_path, plot_exhaustive=False):
    print("\n--- Generating Dose Response Curves for OLS, GPS, RF ---")
    
    try:
        plt.style.use('publication.mplstyle')
    except:
        pass
        
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharey='row')
    
    models = [
        ('MainOLS', 'OLS_SOIL_RATE_SPLINE'),
        ('MainGPS', 'GPS_SOIL_RATE'),
        ('MainRF', 'RF_PREDICTIVE')
    ]
    model_labels = ['OLS', 'GPS', 'RF']
    
    for row, fertilizer in enumerate(['N', 'P']):
        for col, (exp_name, est_name) in enumerate(models):
            ax = axes[row, col]
            
            csv_path = f"outputs/estimates/{fertilizer}-{exp_name}-{est_name}_adj_1-dose_response.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'Soil type' in df.columns and 'T' in df.columns:
                    df = df.set_index(['Soil type', 'T'])
                
                plot_dose_response_curve(df, save_path=None, fertilizer=fertilizer, ax=ax, plot_legend=False, plot_ci=False)
                
                dml_csv_path = f"outputs/estimates/{fertilizer}-MainDML-DML_SOIL_RATE_SPLINE_adj_1-dose_response.csv"
                if os.path.exists(dml_csv_path):
                    dml_df = pd.read_csv(dml_csv_path)
                    if 'Soil type' in dml_df.columns and 'T' in dml_df.columns:
                        dml_df = dml_df.set_index(['Soil type', 'T'])
                    plot_dose_response_curve(dml_df, save_path=None, fertilizer=fertilizer, ax=ax, plot_legend=False, plot_ci=False, linestyle='--', label="_nolegend_")
                
                if plot_exhaustive:
                    overlay_exhaustive_dose_response(ax, fertilizer, est_name, exp_name)
                    
                ALT_ADJ_LEGEND = False
                handles, labels = ax.get_legend_handles_labels()
                
                if not ALT_ADJ_LEGEND and len(handles) > 2:
                    handles = handles[:-1]
                    labels = labels[:-1]
                
                ax.legend(handles, labels, loc='upper left', title="Soil type" if not ALT_ADJ_LEGEND else None, frameon=False)
                
                letter = chr(ord('a') + row*3 + col)
                ax.text(
                    0, 1, letter, transform=ax.transAxes, va='center', ha='center', fontsize=8,
                    fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k'), zorder=6
                )
                
                if fertilizer == 'N':
                    # ax.set_ylim(-2000, 2000)
                    ax.set_ylim(-10, 2900)
                else:
                    # ax.set_ylim(-2000, 2000)
                    ax.set_ylim(-10, 2900)
                if col > 0:
                    ax.set_ylabel("")
                
                title_fert = 'Nitrogen' if fertilizer == 'N' else 'Phosphorus'
                ax.set_title(f"{title_fert} - {model_labels[col]}")
            else:
                print(f"File not found for {fertilizer} {exp_name}: {csv_path}")
                
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {save_path}")

def plot_placebo_dose_response_curves(experiment_name, save_path):
    print(f"\n--- Generating Placebo Dose Response Curves for {experiment_name} ---")
    
    try:
        plt.style.use('publication.mplstyle')
    except:
        pass
        
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.5), sharey='row')
    
    estimator_name = {
        "MainDML":'DML_SOIL_RATE_SPLINE',
        "MainGPS": "GPS_SOIL_RATE",
        "MainOLS": "OLS_SOIL_RATE_SPLINE"
    }[experiment_name]
    soil_types = ['Non-sandy', 'Sandy']
    
    for row, fertilizer in enumerate(['N', 'P']):
        true_csv_path = f"outputs/estimates/{fertilizer}-{experiment_name}-{estimator_name}_adj_1-dose_response.csv"
        true_df = None
        if os.path.exists(true_csv_path):
            true_df = pd.read_csv(true_csv_path)
            if 'Soil type' in true_df.columns and 'T' in true_df.columns:
                true_df = true_df.set_index(['Soil type', 'T'])
                
        placebo_pkl_path = f"outputs/robustness/{fertilizer}-{experiment_name}/{estimator_name}_adj_1-RPT_DOSE.pkl"
        placebo_dfs = []
        if os.path.exists(placebo_pkl_path):
            import pickle
            with open(placebo_pkl_path, 'rb') as f:
                placebo_dfs = pickle.load(f)
                
        for col, soil in enumerate(soil_types):
            ax = axes[row, col]
            placebo_label = "Placebo effect"
            for p_df in placebo_dfs:
                if soil in p_df.index.get_level_values('Soil type'):
                    sub_p = p_df.xs(soil, level='Soil type')
                    ax.plot(
                        sub_p.index, sub_p['Mean Lift'], color='black', alpha=0.25, linewidth=1, zorder=1,
                        label=placebo_label
                    )
                    placebo_label = None
                    
            if true_df is not None:
                if soil in true_df.index.get_level_values('Soil type'):
                    sub_t = true_df.xs(soil, level='Soil type')
                    ax.plot(sub_t.index, sub_t['Mean Lift'], color='red', linewidth=1.5, zorder=5, label='Estimated Effect')
                    
            letter = chr(ord('a') + row*2 + col)
            ax.text(
                0, 1, letter, transform=ax.transAxes, va='center', ha='center', fontsize=8,
                fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k'), zorder=10
            )
            
            ax.set_xlabel(f'{fertilizer}-rate (kg/ha)')
            if col == 0:
                ax.set_ylabel('Change in yield compared to using\nthe reference nutrient rate (kg/ha)')
                
            title_fert = 'Nitrogen' if fertilizer == 'N' else 'Phosphorus'
            ax.set_title(f"{title_fert} - {soil}")
            
            if row == 0 and col == 0:
                ax.legend(frameon=False, loc='upper left')
                
            if fertilizer == 'N':
                # ax.set_ylim(-1990, 1990)
                ax.set_ylim(-1000, 2900)
                ax.set_xlim(35, 180)
            else:
                # ax.set_ylim(-1990, 1990)
                ax.set_ylim(-1000, 2900)
                ax.set_xlim( 14, 52)
                
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {save_path}")

def generate_point_estimates_table(experiment_name, estimators, config_n, config_p):
    print(f"\n--- Point Estimates Table for {experiment_name} ---")
    
    elements = ['N', 'P']
    estimates = ['ATE', 'CATE: Sandy', 'CATE: Non-sandy']
    
    index = pd.MultiIndex.from_product([elements, estimates], names=['Element', 'Estimate'])
    df = pd.DataFrame(index=index, columns=estimators)
    
    for fertilizer in elements:
        valid_estimators = config_n['experiments'].get(experiment_name, {}).get('estimators', []) if fertilizer == 'N' else config_p['experiments'].get(experiment_name, {}).get('estimators', [])
        for estimator in estimators:
            if estimator not in valid_estimators:
                continue
            
            csv_path = f"outputs/estimates/{fertilizer}-{experiment_name}-{estimator}_adj_1-causal_estimates.csv"
            if not os.path.exists(csv_path):
                continue
            
            try:
                res_df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue
            
            for param in estimates:
                row = res_df[res_df['Parameter'] == param]
                if not row.empty:
                    mean_val = row['Mean Value'].values[0]
                    se_val = row['Std Error'].values[0]
                    p_val = row['P-Value'].values[0]
                    
                    stars = ""
                    if pd.notnull(p_val):
                        if p_val < 0.01:
                            stars = "**"
                        elif p_val < 0.05:
                            stars = "*"
                    
                    formatted_str = f"{mean_val:.1f} ({se_val:.1f}){stars}"
                    df.loc[(fertilizer, param), estimator] = formatted_str
                    
    df = df.rename(index={'CATE: Sandy': 'CATE-Sandy', 'CATE: Non-sandy': 'CATE-NonSandy'}, level='Estimate')
    df = df.fillna("-")
    
    print(df.to_markdown())

def generate_alt_point_estimates_table(experiment_name, estimators, config_n, config_p):
    print(f"\n--- Alternative Point Estimates Range Table for {experiment_name} ---")
    
    if 'DML' not in experiment_name:
        print("Only applicable to DML experiments currently.")
        return
        
    exhaustive_exp = experiment_name.replace('Main', 'ExahustiveAdj')
    elements = ['N', 'P']
    estimates = ['ATE', 'CATE: Sandy', 'CATE: Non-sandy']
    
    index = pd.MultiIndex.from_product([elements, estimates], names=['Element', 'Estimate'])
    df = pd.DataFrame(index=index, columns=estimators)
    
    for fertilizer in elements:
        valid_estimators = config_n['experiments'].get(experiment_name, {}).get('estimators', []) if fertilizer == 'N' else config_p['experiments'].get(experiment_name, {}).get('estimators', [])
        
        eff_path = f"outputs/adjustment_sets/adj_main-rev_fertilizerAmount{fertilizer}_outcome_BACKDOOR_EFFICIENT.json"
        exh_path = f"outputs/adjustment_sets/adj_main-rev_fertilizerAmount{fertilizer}_outcome_BACKDOOR_EXHAUSTIVE.json"
        
        if not os.path.exists(eff_path) or not os.path.exists(exh_path):
            continue
            
        with open(eff_path, 'r') as f:
            eff_set = json.load(f)
        with open(exh_path, 'r') as f:
            exh_sets = json.load(f)
            
        eff_set_sorted = sorted(eff_set)
        skip_index = -1
        for i, s in enumerate(exh_sets):
            if sorted(s) == eff_set_sorted:
                skip_index = i + 1
                break
                
        for estimator in estimators:
            if estimator not in valid_estimators:
                continue
                
            param_vals = {p: [] for p in estimates}
            
            for i in range(1, len(exh_sets) + 1):
                if i == skip_index:
                    continue
                    
                csv_path = f"outputs/estimates/{fertilizer}-{exhaustive_exp}-{estimator}_adj_{i}-causal_estimates.csv"
                if not os.path.exists(csv_path):
                    continue
                    
                try:
                    res_df = pd.read_csv(csv_path)
                    for param in estimates:
                        row = res_df[res_df['Parameter'] == param]
                        if not row.empty:
                            param_vals[param].append(row['Mean Value'].values[0])
                except Exception as e:
                    pass
                    
            for param in estimates:
                vals = param_vals[param]
                if vals:
                    min_val = min(vals)
                    max_val = max(vals)
                    df.loc[(fertilizer, param), estimator] = f"[{min_val:.1f}, {max_val:.1f}]"
                    
    df = df.rename(index={'CATE: Sandy': 'CATE-Sandy', 'CATE: Non-sandy': 'CATE-NonSandy'}, level='Estimate')
    df = df.fillna("-")
    
    print(df.to_markdown())

def generate_rpt_table(experiment_name, estimators, fertilizer='N'):
    print(f"\n--- RPT Results Table for {fertilizer}-{experiment_name} ---")
    rows = []
    
    for est in estimators:
        causal_csv = f"outputs/estimates/{fertilizer}-{experiment_name}-{est}_adj_1-causal_estimates.csv"
        if not os.path.exists(causal_csv):
            continue
            
        true_estimates = pd.read_csv(causal_csv)
        
        for _, row in true_estimates.iterrows():
            param = row['Parameter']
            true_val = row['Mean Value']
            true_se = row['Std Error']
            
            if param == 'ATE':
                rpt_csv = f"outputs/robustness/{fertilizer}-{experiment_name}/{est}_adj_1-RPT_ATE.csv"
            elif param.startswith('CATE:'):
                rpt_csv = f"outputs/robustness/{fertilizer}-{experiment_name}/{est}_adj_1-RPT_CATE.csv"
            else:
                continue
                
            rpt_mean = np.nan
            rpt_se = np.nan
            rpt_p_val = np.nan
            
            if os.path.exists(rpt_csv):
                rpt_df = pd.read_csv(rpt_csv)
                if param == 'ATE':
                    placebo_vals = rpt_df['Value'].values
                    placebo_ses = rpt_df['Std_Error'].values
                else:
                    group = param.replace('CATE: ', '').strip()
                    if 'Group' in rpt_df.columns:
                        subset = rpt_df[rpt_df['Group'] == group]
                        placebo_vals = subset['Value'].values
                        placebo_ses = subset['Std_Error'].values
                    else:
                        placebo_vals = []
                        placebo_ses = []
                
                if len(placebo_vals) > 0:
                    import scipy.stats as stats
                    rpt_mean = np.mean(placebo_vals)
                    rpt_se = np.mean(placebo_ses)
                    _, rpt_p_val = stats.ttest_1samp(placebo_vals, 0.0)
                    
            rows.append({
                'Estimator': est,
                'Effect': param,
                f'{fertilizer}-AE': rpt_mean,
                'SE': rpt_se,
                'p-value': rpt_p_val
            })
            
    if not rows:
        print("No RPT data found.")
        return
        
    df_res = pd.DataFrame(rows)
    df_res = df_res.set_index(['Estimator', 'Effect'])
    
    df_res[f'{fertilizer}-AE'] = df_res[f'{fertilizer}-AE'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "-")
    df_res['SE'] = df_res['SE'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "-")
    df_res['p-value'] = df_res['p-value'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
    
    print(df_res.to_markdown())

def generate_ptpo_table(experiment_name, estimators, fertilizer='N'):
    print(f"\n--- PTPO Results Table for {fertilizer}-{experiment_name} ---")
    rows = []
    
    for est in estimators:
        ptpo_csv = f"outputs/robustness/{fertilizer}-{experiment_name}/{est}_adj_1-PTPO_results.csv"
        if not os.path.exists(ptpo_csv):
            continue
            
        ptpo_df = pd.read_csv(ptpo_csv)
        for _, row in ptpo_df.iterrows():
            if row['Parameter'] == 'ATE':
                rows.append({
                    'Estimator': est,
                    'Placebo Outcome': row['Placebo_Outcome'],
                    'Parameter': row['Parameter'],
                    'Effect': row['Value'],
                    'SE': row['Std_Error'],
                    'p-value': row['P_Value']
                })
            
    if not rows:
        # print("No PTPO data found.")
        return
        
    df_res = pd.DataFrame(rows)
    df_res = df_res.set_index(['Estimator', 'Placebo Outcome', 'Parameter'])
    
    df_res['Effect'] = df_res['Effect'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
    df_res['SE'] = df_res['SE'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
    df_res['p-value'] = df_res['p-value'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
    
    print(df_res.to_markdown())

def plot_ucc(experiment_name, estimator_name):
    from src.plots.estimates import plot_ucc_contours
    import pickle
    
    fig, axes = plt.subplots(1, 2, figsize=(6., 3.0), sharey=True)
    
    # Phosphorus on the left
    p_path = f"outputs/robustness/P-{experiment_name}/{estimator_name}_adj_1_ATE-ucc_data.pkl"
    if os.path.exists(p_path):
        with open(p_path, 'rb') as f:
            p_data = pickle.load(f)
        plot_ucc_contours(p_data, ax=axes[1], title='Phosphorus', levels=np.arange(-80, 60, 5))
        axes[1].text(
            0, 1, 'b', transform=axes[1].transAxes, va='center', ha='center', fontsize=8,
            fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k'), zorder=6
        )
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Residual treatment variance\nexplained by UCC") 
    else:
        print(f"UCC data not found for P: {p_path}")
        
    # Nitrogen on the right
    n_path = f"outputs/robustness/N-{experiment_name}/{estimator_name}_adj_1_ATE-ucc_data.pkl"
    if os.path.exists(n_path):
        with open(n_path, 'rb') as f:
            n_data = pickle.load(f)
        plot_ucc_contours(n_data, ax=axes[0], title='Nitrogen', levels=np.arange(-20, 20, 1))
        axes[0].text(
            0, 1, 'a', transform=axes[0].transAxes, va='center', ha='center', fontsize=8,
            fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k'), zorder=6
        )
        axes[0].set_ylabel("Residual outcome variance\nexplained by UCC") 
        axes[0].set_xlabel("Residual treatment variance\nexplained by UCC") 
    else:
        print(f"UCC data not found for N: {n_path}")
        
    plt.tight_layout()
    save_path = f"plots/publication/UCC_{experiment_name}_{estimator_name}_ATE.pdf"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved UCC plot to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="MBA Tanzania - Plots and Tables Generator")
    return parser.parse_args()

def generate_descriptive_statistics():
    import pickle
    import json
    import pandas as pd
    import numpy as np
    from src.estimators.helpers import _build_time_matrix, _build_legume_matrix, _build_pfert_matrix, _build_soil_matrix

    data_path = 'outputs/cache/data_preprocessed_260116-transformed.pkl'
    if not os.path.exists(data_path):
        print("Preprocessed data not found.")
        return
        
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open('config/node_variable_map.json', 'r') as f:
        node_map = json.load(f)

    # All nodes in the DAG
    nodes = list(node_map.keys())
    
    # First, get all variables used in the DAG to filter the dataframe
    all_n_vars = []
    for node in nodes:
        all_n_vars.extend([v for v in node_map.get(node, []) if v in data.columns])
        
    # Drop rows with any NA values in these columns to exactly match the analysis dataset
    data = data.dropna(subset=all_n_vars)
    data = data.loc[data['fertN_total'] > 0]
    
    rows = []
    for node in nodes:
        vars_in_node = [v for v in node_map.get(node, []) if v in data.columns]
        for var in vars_in_node:
            s = data[var].astype(float)
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
        n_val = subset['fertN_total'].astype(float)
        y_val = subset['maizeprodkg_ha'].astype(float)
        
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

def literature_comparison_plot(save_path):
    plt.style.use('publication.mplstyle') # mpl style for publication-quality figures

    nae_estimates = {
        'DML Soil': {'mean': 10., 'se': 1.0},
        'OLS Soil': {'mean': 8.7, 'se': 0.7},
        'DML Soil+Rate': {'mean': 13.2, 'se': 2.3},
        'OLS Soil+Rate': {'mean': 9.5, 'se': 2.1},
        'GPS Soil+Rate': {'mean': 15.3, 'se': 1.4},
        'RF Soil+Rate': {'mean': 7.2, 'se': 3.2},
    }
    nae_lit = {
        'Rurinda-TZ': {'mean': 13}, # 300 Sites, NOT, close to optimum (100-140) # Point-level between 0-40
        'Rurinda-TZ-SH': {'mean': 14.7}, # Nutrient omision trials, variable N-rates are an optimum region-based
        'Wortmann-TZ': {'mean': 16} # 50kg N-AE, 9 Sites. Used the overal TZ value, 
    }

    pae_estimates = {
        'DML Soil': {'mean': 52.3, 'se': 5.7},
        'OLS Soil': {'mean': 38.3, 'se': 3.0},
        'DML Soil+Rate': {'mean': 102.0, 'se': 12.9},
        'OLS Soil+Rate': {'mean': 77.2, 'se': 9.6},
        'GPS Soil+Rate': {'mean': 45.1, 'se': 5.2},
        'RF Soil+Rate': {'mean': 73.0, 'se': 30.9},
    }
    pae_lit = {
        'Wortmann-TZ': {'mean': 0},# 10kg, 9 Sites. Used the overal TZ value 
        'Rurinda-TZ': {'mean': 14.8}, # 300 Sites, NOT, close to optimum (100-140)
        'Rurinda-TZ-SH': {'mean': 24.1}, # Nutrient omision trials, variable N=P-rates close to optimum
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6, 3.5))

    # Plot NAE estimates
    for method, data in nae_estimates.items():
        ax1.errorbar(
            data['mean'], method, xerr = 1.96*data.get('se', 0), 
            fmt = 'o' if method in ['OLS', 'DML'] else 's', 
            capsize = 3,elinewidth=2, ms=5, color='black', zorder=2
        )
        ax1.set_xlabel('Average N-AE (kg/kg)')
        ax1.text(
            0, 1, 'a', transform=ax1.transAxes, va='center', ha='center', fontsize=8,
            fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k')
        )
        ax1.set_ylim(-.5, 5.5)
        ax1.set_title('Nitrogen')
    linestyles = {
        'Wortmann-TZ': ":",
        'Rurinda-TZ': "--",
        'Rurinda-TZ-SH': "-."
    }
    for n, (ref, data) in enumerate(nae_lit.items()):
        ax1.axvline(
            data['mean'], linestyle=linestyles[ref], color='gray', zorder=1
        )

    # plot PAE estimates
    for method, data in pae_estimates.items():
        ax2.errorbar(
            data['mean'], method, xerr = 1.96*data.get('se', 0), 
            fmt = 'o' if method in ['OLS', 'DML'] else 's',  
            capsize = 3,elinewidth=2, ms=5, color='black',
            zorder=2
        )
        ax2.set_xlabel('Average P-AE (kg/kg)')
        ax2.text(
            0, 1, 'b', transform=ax2.transAxes, va='center', ha='center', fontsize=8,
            fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k')
        )
        ax2.set_ylim(-.5, 5.5)
        ax2.set_title('Phosphorus')

    for n, (ref, data) in enumerate(pae_lit.items()):
        ax2.axvline(
            data['mean'], linestyle=linestyles[ref], color='gray',
            zorder=1
        )
    # fig.subplots_adjust(top=.8, left=.15, wspace=.50, right=.95)
    descriptions = {
        'Wortmann-TZ': "Wortmann et al. (2018) estimate for Tanzania",
        'Rurinda-TZ': "Rurinda et al. (2020) estimate for Tanzania",
        'Rurinda-TZ-SH': "Rurinda et al. (2020) estimate for the Southern Highlands of Tanzania"
    }
    import matplotlib.lines as mlines
    handles = []
    for ref, desc in descriptions.items():
        handles.append(
            mlines.Line2D([], [], color='gray', linestyle=linestyles[ref], label=desc)
        )
    fig.legend(handles=handles, loc='lower center', title='Reference values', fontsize=7, ncol=1)
    plt.subplots_adjust(bottom=.3, left=.15, right=0.98, wspace=.5)
    plt.savefig(save_path)
    print(f"Saved literature comparison plot to {save_path}")

def plot_sensitivity_dose_curves(experiment_name, save_path):
    print("\n--- Generating Dose Response Curves for SensitivityDose ---")
    estimator_name = {
        "MainDML": 'DML_SOIL_RATE_SPLINE',
        "MainGPS": 'GPS_SOIL_RATE'
    }[experiment_name]
    
    plt.style.use('publication.mplstyle')
        
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6), sharex='col', sharey='row')
    
    for col, fertilizer in enumerate(['N', 'P']):
        
        summary_csv = f"outputs/robustness/{fertilizer}-{experiment_name}/{estimator_name}_adj_1-SensitivityDose_summary.csv"
        r2_ut_cd = r2_ut_cy = r2_uy_cd = r2_uy_cy = 0.0
        if os.path.exists(summary_csv):
            df_summary = pd.read_csv(summary_csv)
            for _, row in df_summary.iterrows():
                if row['Simulated_U'] == 'U_T':
                    r2_ut_cd = row['Partial_R2_T'] * 100
                    r2_ut_cy = row['Partial_R2_Y'] * 100
                elif row['Simulated_U'] == 'U_Y':
                    r2_uy_cd = row['Partial_R2_T'] * 100
                    r2_uy_cy = row['Partial_R2_Y'] * 100
                    
        def get_ate(csv_path):
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                row_df = df[df['Parameter'] == 'ATE']
                if not row_df.empty:
                    return row_df['Mean Value'].values[0]
            return np.nan
            
        csv_orig_ate = f"outputs/estimates/{fertilizer}-{experiment_name}-{estimator_name}_adj_1-causal_estimates.csv"
        csv_ut_ate = f"outputs/robustness/{fertilizer}-{experiment_name}/{estimator_name}_adj_1-U_T-causal_estimates.csv"
        csv_uy_ate = f"outputs/robustness/{fertilizer}-{experiment_name}/{estimator_name}_adj_1-U_Y-causal_estimates.csv"
        
        ate_orig = get_ate(csv_orig_ate)
        ate_ut = get_ate(csv_ut_ate)
        ate_uy = get_ate(csv_uy_ate)
        
        csv_orig = f"outputs/estimates/{fertilizer}-{experiment_name}-{estimator_name}_adj_1-dose_response.csv"
        csv_ut = f"outputs/robustness/{fertilizer}-{experiment_name}/{estimator_name}_adj_1-U_T-dose_response.csv"
        csv_uy = f"outputs/robustness/{fertilizer}-{experiment_name}/{estimator_name}_adj_1-U_Y-dose_response.csv"
        
        df_orig = pd.read_csv(csv_orig).set_index(['Soil type', 'T']) if os.path.exists(csv_orig) else None
        df_ut = pd.read_csv(csv_ut).set_index(['Soil type', 'T']) if os.path.exists(csv_ut) else None
        df_uy = pd.read_csv(csv_uy).set_index(['Soil type', 'T']) if os.path.exists(csv_uy) else None
        
        soil_types = ['Sandy', 'Non-sandy']
        for row, soil in enumerate(soil_types):
            ax = axes[row, col]
            
            def subset_soil(df, s):
                if df is not None and s in df.index.get_level_values('Soil type'):
                    return df.loc[[s]]
                return None
                
            df_orig_soil = subset_soil(df_orig, soil)
            df_ut_soil = subset_soil(df_ut, soil)
            df_uy_soil = subset_soil(df_uy, soil)
            
            if df_orig_soil is not None:
                plot_dose_response_curve(
                    df_orig_soil, save_path=None, fertilizer=fertilizer, ax=ax, 
                    plot_legend=False, plot_ci=False, label="_nolegend_", 
                    color_mapping={'Sandy': 'black', 'Non-sandy': 'black'}
                )
            if df_ut_soil is not None:
                plot_dose_response_curve(
                    df_ut_soil, save_path=None, fertilizer=fertilizer, ax=ax, 
                    plot_legend=False, plot_ci=False, label="_nolegend_", linestyle='--', 
                    color_mapping={'Sandy': 'black', 'Non-sandy': 'black'}
                )
            if df_uy_soil is not None:
                plot_dose_response_curve(
                    df_uy_soil, save_path=None, fertilizer=fertilizer, ax=ax, 
                    plot_legend=False, plot_ci=False, label="_nolegend_", linestyle=':', 
                    color_mapping={'Sandy': 'black', 'Non-sandy': 'black'}
                )
            
            ate_text = f"ATE No UCC: {ate_orig:.1f}\nATE $U_T$ UCC: {ate_ut:.1f}\nATE $U_Y$ UCC: {ate_uy:.1f}"
            if row == 1:
                ax.text(
                    0.05, 0.95, ate_text, transform=ax.transAxes, va='top', ha='left', 
                    fontsize=7, zorder=10
                )
            
            if row == 0:
                import matplotlib.lines as mlines
                line_orig = mlines.Line2D([], [], color='black', linestyle='-', label='No UCC')
                line_ut = mlines.Line2D(
                    [], [], color='black', linestyle='--', 
                    label=f'UCC: $U_T$ ($c_d={r2_ut_cd:.1f}$%, $c_y={r2_ut_cy:.1f}$%)'
                )
                line_uy = mlines.Line2D(
                    [], [], color='black', linestyle=':', 
                    label=f'UCC: $U_Y$ ($c_d={r2_uy_cd:.1f}$%, $c_y={r2_uy_cy:.1f}$%)'
                )
                ax.legend(handles=[line_orig, line_ut, line_uy], loc='upper left', frameon=False, fontsize=7)
                
            letter = chr(ord('a') + row * 2 + col)
            ax.text(
                0, 1, letter, transform=ax.transAxes, va='center', ha='center', fontsize=8,
                fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k'), zorder=6
            )
            
            if fertilizer == 'N':
                ax.set_xlim(38, 177)
                ax.set_ylim(-10, 2900)
                ax.set_title(f"Nitrogen - {soil}")
            else:
                ax.set_xlim(14, 52)
                ax.set_title(f"Phosphorus - {soil}")
                
            if col == 1:
                ax.set_ylabel("")
                
            if row == 0:
                ax.set_xlabel("")
            else:
                ax.set_xlabel(f'{fertilizer}-rate (kg/ha)')
                
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved SensitivityDose figure to {save_path}")

def main():
    # Load both N and P configs to get the list of estimators for MainDML
    with open("config/experiments_N.toml", "rb") as f:
        if hasattr(tomllib, 'load'):
            config_n = tomllib.load(f)
        else:
            config_n = tomllib.loads(f.read().decode("utf-8"))
            
    with open("config/experiments_P.toml", "rb") as f:
        if hasattr(tomllib, 'load'):
            config_p = tomllib.load(f)
        else:
            config_p = tomllib.loads(f.read().decode("utf-8"))
            
    print("=== Generating Plots and Tables ===")
    experiments_to_run = ["MainDML", "MainOLS", "MainRF", "MainGPS", "MainDMLBoosting"]
    for exp in experiments_to_run:
        estimators_n = config_n['experiments'].get(exp, {}).get('estimators', [])
        estimators_p = config_p['experiments'].get(exp, {}).get('estimators', [])
        estimators = list(dict.fromkeys(estimators_n + estimators_p)) # Preserve order, remove duplicates
        
        if estimators:
            generate_point_estimates_table(exp, estimators, config_n, config_p)
            generate_alt_point_estimates_table(exp, estimators, config_n, config_p)
            
            if exp in ["MainDML", "MainOLS", "MainGPS"]:
                generate_rpt_table(exp, estimators, fertilizer='N')
                generate_rpt_table(exp, estimators, fertilizer='P')
                generate_ptpo_table(exp, estimators, fertilizer='N')
                generate_ptpo_table(exp, estimators, fertilizer='P')
            
    # Main experiment plots
    plot_soil_time_estimates("MainDML", "plots/publication/SOIL_TIME_MainDML.pdf", plot_exhaustive=False)
    plot_legume_pfert_estimates("MainDML", "plots/publication/LEGUME_PFERT_MainDML.pdf", plot_exhaustive=False)
    plot_dose_response_curves("MainDML", "plots/publication/DOSE_RESPONSE_MainDML.pdf", plot_exhaustive=False)
    # Other experiments plots
    plot_other_dose_response_curves("plots/publication/DOSE_RESPONSE_Other.pdf")
    plot_soil_time_estimates("MainOLS", "plots/publication/SOIL_TIME_MainOLS.pdf", plot_exhaustive=False)
    plot_legume_pfert_estimates("MainOLS", "plots/publication/LEGUME_PFERT_MainOLS.pdf", plot_exhaustive=False)

    literature_comparison_plot("plots/publication/estimates_comparison.pdf")

    plot_dose_response_curves("MainDML", "plots/publication/DOSE_RESPONSE_ExhaustiveAdjDML.pdf", plot_exhaustive=True, plot_main=False)
    plot_soil_time_estimates("MainDML", "plots/publication/SOIL_TIME_ExhaustiveAdjDML.pdf", plot_exhaustive=True, plot_main=False)
    plot_legume_pfert_estimates("MainDML", "plots/publication/LEGUME_PFERT_ExhaustiveAdjDML.pdf", plot_exhaustive=True, plot_main=False)
    
    # Placebo results
    plot_placebo_dose_response_curves("MainGPS", "plots/publication/DOSE_RESPONSE_PLACEBO_MainGPS.pdf")
    plot_placebo_dose_response_curves("MainOLS", "plots/publication/DOSE_RESPONSE_PLACEBO_MainOLS.pdf")
    plot_ucc("MainDML", "DML_SOIL")
    plot_sensitivity_dose_curves("MainDML", "plots/publication/DOSE_RESPONSE_SensitivityDose_MainDML.pdf")
    plot_sensitivity_dose_curves("MainGPS", "plots/publication/DOSE_RESPONSE_SensitivityDose_MainGPS.pdf")
    # Descriptive statistics
    generate_descriptive_statistics()

if __name__ == "__main__":
    main()
