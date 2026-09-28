import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_gps_support(diagnostics, transformed_data, raw_data, treatment_var, fertilizer, save_path=None):
    """
    Plots the density of predicted treatment (T_pred) stratified by quantiles of actual treatment (T).
    
    Args:
        diagnostics (dict): Dictionary with 'T_pred', 'mask_support', and 'T'.
        transformed_data (pd.DataFrame): The transformed covariates DataFrame.
        raw_data (pd.DataFrame): The raw data before demeaning.
        treatment_var (str): The name of the treatment variable.
        fertilizer (str): "N" or "P".
        save_path (str): File path to save the plot.
    """
    if diagnostics is None:
        print("No diagnostics available for plotting.")
        return
        
    T_pred_anomaly = diagnostics.get('T_pred')
    mask_support = diagnostics.get('mask_support')
    T_demeaned = diagnostics.get('T')
    
    if T_pred_anomaly is None or mask_support is None or T_demeaned is None:
        print("Missing required diagnostics fields for GPS support plot.")
        return

    # Extract raw treatment and compute spatial expectation
    T_raw = raw_data.loc[transformed_data.index, treatment_var].values
    spatial_expectation = T_raw - T_demeaned[mask_support]
    T_pred_actual = T_pred_anomaly[mask_support] + spatial_expectation

    # Create the DataFrame for plotting using raw treatment for quantiles (or demeaned, depending on user intent)
    ps_df = pd.DataFrame({
        'T_pred': T_pred_actual, 
        'Strata': pd.qcut(T_raw, 5, labels=False, duplicates='drop')
    }, index=transformed_data.index)

    # Re-map Strata to categorical names
    strata_map = {0: "Q1 (Low)", 1: "Q2", 2: "Q3", 3: "Q4", 4: "Q5 (High)"}
    # In case there are fewer than 5 quantiles due to duplicates
    unique_strata = sorted(ps_df['Strata'].unique())
    current_map = {val: strata_map.get(i, f"Q{i+1}") for i, val in enumerate(unique_strata)}
    if len(unique_strata) == 5:
        current_map = strata_map
        
    # Generate Plot
    g = sns.FacetGrid(ps_df, row="Strata", hue="Strata", aspect=7, height=.5, palette="viridis")
    g.map(sns.kdeplot, "T_pred", clip_on=False, fill=True, alpha=0.7, lw=0)
    g.map(sns.kdeplot, "T_pred", clip_on=False, color="w", lw=2)

    g.fig.subplots_adjust(hspace=-.5)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    
    title_text = "Nitrogen" if fertilizer == "N" else "Phosphorus"
    g.axes.flat[0].set_title(title_text, fontweight='bold', y=.99) 
    g.axes.flat[-1].set_xlabel(r'$E[T|X,W]$'+' (kg/ha)')

    # Add text labels for each stratum
    for i, ax in enumerate(g.axes.flat):
        label = current_map.get(unique_strata[i], f"Q{i+1}")
        ax.text(0, 0.2, label, fontweight="bold", transform=ax.transAxes)
        ax.axhline(0, color='k', linestyle="-")
        ax.set_facecolor('none')
        ax.tick_params(length=0)

    g.axes.flat[0].text(
        0, 1, 'a', transform=g.axes.flat[0].transAxes, va='center', ha='center', fontsize=8,
        fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k')
    )
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(g.fig)
        print(f"Saved GPS support plot to {save_path}")
    else:
        # If no save path, just return the FacetGrid object
        return g

