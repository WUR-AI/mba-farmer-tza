from matplotlib import pyplot as plt 
import os 

plt.style.use('publication.mplstyle') # mpl style for publication-quality figures

nae_estimates = {
    'DML': {'mean': 12.5, 'se': 1.3},
    'OLS': {'mean': 10.2, 'se': 1.2},
    'Rurinda-TZ': {'mean': 13}, # 300 Sites, NOT, close to optimum (100-140) # Point-level between 0-40
    'Rurinda-TZ-SH': {'mean': 14.7}, # Nutrient omision trials, variable N-rates are an optimum region-based
    'Wortmann-TZ': {'mean': 16} # 50kg N-AE, 9 Sites. Used the overal TZ value, 
}

pae_estimates = {
    'DML': {'mean': 60, 'se': 7.6},
    'OLS': {'mean': 44, 'se': 7.8},
    'Rurinda-TZ': {'mean': 14.8}, # 300 Sites, NOT, close to optimum (100-140)
    'Rurinda-TZ-SH': {'mean': 24.1}, # Nutrient omision trials, variable N=P-rates close to optimum
    'Wortmann-TZ': {'mean': 0},# 10kg, 9 Sites. Used the overal TZ value 
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (4, 4))

# Plot NAE estimates
for method, data in nae_estimates.items():
    ax1.errorbar(
        data['mean'], method, xerr = 1.96*data.get('se', 0), 
        fmt = 'o' if method in ['OLS', 'DML'] else 's', 
        capsize = 3,elinewidth=2, ms=5, color='black'
    )
    ax1.set_xlabel('Average N-AE (kg/kg)')
    ax1.text(
        0, 1, 'a', transform=ax1.transAxes, va='center', ha='center', fontsize=8,
        fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k')
    )
    ax1.set_ylim(-.5, 4.5)
    ax1.set_title('Nitrogen')

# plot PAE estimates
for method, data in pae_estimates.items():
    ax2.errorbar(
        data['mean'], method, xerr = 1.96*data.get('se', 0), 
        fmt = 'o' if method in ['OLS', 'DML'] else 's',  
        capsize = 3,elinewidth=2, ms=5, color='black'
    )
    ax2.set_xlabel('Average P-AE (kg/kg)')
    ax2.text(
        0, 1, 'b', transform=ax2.transAxes, va='center', ha='center', fontsize=8,
        fontweight='bold', bbox=dict(boxstyle='circle', fc='w', ec='k')
    )
    ax2.set_ylim(-.5, 4.5)
    ax2.set_title('Phosphorus')
plt.tight_layout()
DATA_PATH = "/data/mba-tza"
plt.savefig(f'{DATA_PATH}/results/figures/compare_results.pdf')
    
