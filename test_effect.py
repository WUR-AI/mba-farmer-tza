import pickle
import numpy as np

with open("outputs/estimators/DML_SOIL_RATE_main-dag_fertilizerAmountN_outcome_BACKDOOR_EFFICIENT_adj_1.pkl", "rb") as f:
    model = pickle.load(f)

t25, t75 = np.quantile(model.T_absolute, [0.25, 0.75])
X = model.X

effect = model.est.effect(X, T0=t25, T1=t75)
print("ATE using .effect():", (effect / (t75 - t25)).mean())

X_sandy = X[X[:, 1] == 1]
print("Sandy CATE using .effect():", (model.est.effect(X_sandy, T0=t25, T1=t75) / (t75 - t25)).mean())

X_nonsandy = X[X[:, 0] == 1]
print("Non-sandy CATE using .effect():", (model.est.effect(X_nonsandy, T0=t25, T1=t75) / (t75 - t25)).mean())
