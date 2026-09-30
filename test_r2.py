import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

with open('outputs/cache/data_preprocessed_260116-transformed.pkl', 'rb') as f:
    data = pickle.load(f)

# P-users
data = data[data['fertP_total'] > 0]
Y = data['yield_kg_per_ha']
T = data['fertP_total']
cols = [c for c in data.columns if 'fieldHistory' in c or '0to100' in c] # Just take some cols
data = data.dropna(subset=['yield_kg_per_ha', 'fertP_total'] + cols)

X = data[cols]
rf = RandomForestRegressor(max_depth=3, random_state=43)
rf.fit(X, T)

mse_train = np.mean((T - rf.predict(X))**2)
r2_train = 1 - mse_train / np.var(T)

cv_scores = cross_val_score(rf, X, T, cv=5, scoring='r2')
print("Train R2:", r2_train)
print("CV R2:", np.mean(cv_scores))
