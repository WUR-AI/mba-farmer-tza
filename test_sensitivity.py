import pickle
import numpy as np

with open('outputs/estimators/DML_SOIL_RATE_SPLINE_main-rev_fertilizerAmountP_outcome_BACKDOOR_EFFICIENT_adj_1.pkl', 'rb') as f:
    model = pickle.load(f)

cy_mesh, cd_mesh = np.meshgrid(np.linspace(0, 0.5, 2), np.linspace(0, 0.5, 2))

lb_theta, ub_theta = model.est.sensitivity_interval(c_y=cy_mesh, c_t=cd_mesh, rho=1.0, alpha=0.05, interval_type='theta')
lb_ci, ub_ci = model.est.sensitivity_interval(c_y=cy_mesh, c_t=cd_mesh, rho=1.0, alpha=0.05, interval_type='ci')

print("Theta bounds:", lb_theta, ub_theta)
print("CI bounds:", lb_ci, ub_ci)
