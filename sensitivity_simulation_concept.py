import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from econml.dml import LinearDML
from sklearn.preprocessing import SplineTransformer
from sklearn.multioutput import MultiOutputRegressor

def generate_data(n_samples=1500, random_state=42):
    """
    Generates synthetic agricultural data following a Mitscherlich dose-response.
    """
    np.random.seed(random_state)
    
    # Confounders (X)
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    x4 = np.random.normal(0, 1, n_samples)
    x5 = np.random.normal(0, 1, n_samples)
    
    X = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})
    
    # Treatment T (N-rate)
    # x1, x2, x3 increase fertilizer usage Propensity
    mu_T = 100 + 30 * x1 + 30 * x5 + 30 * x3 
    mu_T = np.clip(mu_T, 10, 250)
    T = np.random.normal(mu_T, 20)
    T = np.clip(T, 0, 250)
    
    # Outcome parameters
    # a: Maximum yield. x1 increases 'a', x2 decreases 'a', x3 has no effect.
    a = 5000 + 400 * x1 - 800 * x2 + 0 * x3 + 200 * x4
    a = np.clip(a, 1000, 8000)  # Ensure maximum yield is strictly positive
    # a = 4000*np.ones_like(x1)
    
    # c: curvature
    c = 0.015 + x5 *0.006

    # b: soil inherent nutrient, y0: baseline fertility
    y0 = 2000 + 500 * x5
    y0 = np.clip(y0, 250, 5000)
    y0 = np.where(a > y0, y0, 0.3*a)
    # y0 = 2000*np.ones_like(x5)
    b = -1/c*np.log(1-y0/a)
       
    # True Outcome Y
    Y = a * (1 - np.exp(-c * (T + b))) + np.random.normal(0, 200, n_samples)
    Y = np.maximum(Y, 0)  # Ensure yield is never negative
    
    return X, T, Y, a, b, c

def true_dose_response(t_eval, a, b, c, t_ref=0):
    """
    Calculates the Marginal Average Dose-Response analytically.
    """
    y_t = np.mean(a * (1 - np.exp(-c * (t_eval + b))))
    y_ref = np.mean(a * (1 - np.exp(-c * (t_ref + b))))
    return y_t - y_ref

def fit_dml(X, T, Y):
    """
    Fits the DML_RATE estimator (mimicking DML_SOIL_RATE_SPLINE structure without soil).
    """
    spline = SplineTransformer(degree=3, n_knots=5, include_bias=False)
    
    model_y = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=5)
    model_t = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=5))
    
    est = LinearDML(
        model_y=model_y,
        model_t=model_t,
        treatment_featurizer=spline,
        discrete_treatment=False,
        cv=5,
        random_state=42,
        fit_cate_intercept=True
    )
    
    # We only care about ATE over T, so we use dummy X for heterogeneity
    dummy_X = np.ones((len(Y), 1))
    
    # EconML and SplineTransformer expect T to be 2D
    T_2d = T.reshape(-1, 1) if T.ndim == 1 else T
    Y_1d = Y.ravel()
    
    est.fit(Y=Y_1d, T=T_2d, W=X)
    return est

def get_partial_r2(X_full, T, Y, drop_var):
    """
    Calculates the partial R^2 of `drop_var` on T and Y (empirical UCC impact).
    """
    model_T = RandomForestRegressor(random_state=42, max_depth=5, n_jobs=-1)
    model_Y = RandomForestRegressor(random_state=42, max_depth=5, n_jobs=-1)
    
    # Full models
    model_T.fit(X_full, T)
    mse_T_full = np.mean((T - model_T.predict(X_full))**2)
    
    model_Y.fit(X_full, Y)
    mse_Y_full = np.mean((Y - model_Y.predict(X_full))**2)
    
    # Reduced models (dropping the variable of interest)
    X_red = X_full.drop(columns=[drop_var])
    
    model_T.fit(X_red, T)
    mse_T_red = np.mean((T - model_T.predict(X_red))**2)
    
    model_Y.fit(X_red, Y)
    mse_Y_red = np.mean((Y - model_Y.predict(X_red))**2)
    
    # Partial R^2 formula
    cd = (mse_T_red - mse_T_full) / mse_T_red if mse_T_red > 0 else 0
    cy = (mse_Y_red - mse_Y_full) / mse_Y_red if mse_Y_red > 0 else 0
    
    return max(0, cd), max(0, cy)

def main():
    try:
        plt.style.use('publication.mplstyle')
    except:
        pass

    print("Generating synthetic data...")
    X, T, Y, a, b, c = generate_data()
    print(pd.Series(T, name='Treatment').describe())
    print(pd.Series(Y, name='Outcome').describe())
    print(pd.Series(a, name='a').describe())
    print(pd.Series(a*(1-np.exp(-c*b)), name='y0').describe())
    print('c=',c)
    # exit()

    t_ref_val = np.percentile(T, 5)
    t_ref_val = 0
    t_eval = np.linspace(t_ref_val, np.percentile(T, 95), 50)
    t_ref = np.full_like(t_eval, t_ref_val)
    
    # Calculate ground truth curve
    true_dr = [true_dose_response(t, a, b, c, t_ref=t_ref_val) for t in t_eval]
    
    print("Fitting fully adjusted DML model...")
    est_full = fit_dml(X[['x1', 'x2', 'x4', 'x5']], T, Y)
    dr_full = np.ravel(est_full.effect(T0=t_ref.reshape(-1, 1), T1=t_eval.reshape(-1, 1)))
    
    scenarios = [
        ('x5', r'Increases baseline yield'),
        # ('x1', r'Decreases potential yield'),
        # ('x3', r'$\rho = 0$ (Instrumental Variable)')
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    # axes = axes.flatten()
    
    # Panel 1: Ground Truth vs Fully Adjusted
    # ax = axes[0]
    # ax.plot(t_eval, true_dr, '-', color="grey", linewidth=4, alpha=.5, label='Ground Truth')
    ax.plot(t_eval, dr_full, 'r--', linewidth=2, label='No UCC')
    ax.set_xlabel("N-rate (kg/ha)")
    ax.set_ylabel("Yield increase compared to\nreference nutrient rate (kg/ha)")
    ax.legend(frameon=False)
    
    for i, (var, title) in enumerate(scenarios):
        print(f"Testing omitted variable bias for {var}...")
        
        # Calculate empirical partial R2
        cd, cy = get_partial_r2(X, T, Y, var)
        
        # Fit DML without this confounder
        X_dropped = X.drop(columns=[var, 'x3'])
        est_dropped = fit_dml(X_dropped, T, Y)
        dr_dropped = np.ravel(est_dropped.effect(T0=t_ref.reshape(-1, 1), T1=t_eval.reshape(-1, 1)))
        
        ax.plot(
            t_eval, dr_dropped, color=['b', 'k'][i], linestyle=[":", "-."][i], linewidth=2, 
            label=f"UCC: $c_d={cd*100:.1f}$%, $c_y={cy*100:.1f}$%"
        )
        
        # ax.set_title(f"{title}\n$c_d={cd*100:.1f}$%, $c_y={cy*100:.1f}$%")
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig('plots/publication/sensitivity_simulation_concept.pdf')
    print("Plot successfully saved to sensitivity_simulation_concept.pdf")

if __name__ == "__main__":
    main()
