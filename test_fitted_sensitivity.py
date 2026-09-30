import pickle
with open('outputs/estimators/DML_SOIL_main-rev_fertilizerAmountN_outcome_BACKDOOR_EFFICIENT_adj_1.pkl', 'rb') as f:
    model = pickle.load(f)

theta = 10.0
theta_stderr = 1.0
cf_y = 0.1
cf_d = 0.1
rho = 1.0

lb, ub, lci, uci = model.est.sensitivity_interval(theta, theta_stderr, cf_y, cf_d, rho)
print("LB:", lb)
print("UB:", ub)
