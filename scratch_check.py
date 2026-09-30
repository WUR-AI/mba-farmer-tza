import pickle
import numpy as np
from scipy.interpolate import interp1d

with open('outputs/robustness/N-MainDML/DML_SOIL_RATE_SPLINE_adj_1-RPT_DOSE.pkl', 'rb') as f:
    dfs = pickle.load(f)

def get_y(df, x_new):
    s = df.xs('Sandy', level='Soil type')
    f_interp = interp1d(s.index, s['Mean Lift'], bounds_error=False, fill_value=np.nan)
    return f_interp(x_new)

x_grid = np.linspace(50, 150, 100)
y0 = get_y(dfs[0], x_grid)
y1 = get_y(dfs[1], x_grid)
y2 = get_y(dfs[2], x_grid)
y3 = get_y(dfs[3], x_grid)

valid = ~np.isnan(y0) & ~np.isnan(y1)
print('Corr 0 and 1:', np.corrcoef(y0[valid], y1[valid])[0,1])
valid = ~np.isnan(y0) & ~np.isnan(y2)
print('Corr 0 and 2:', np.corrcoef(y0[valid], y2[valid])[0,1])
valid = ~np.isnan(y0) & ~np.isnan(y3)
print('Corr 0 and 3:', np.corrcoef(y0[valid], y3[valid])[0,1])
