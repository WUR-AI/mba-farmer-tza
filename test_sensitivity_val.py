from econml.dml import LinearDML
import numpy as np

theta = 10.0
theta_stderr = 1.0
cf_y = 0.1
cf_d = 0.1
rho = 1.0

# LinearDML is a class, we need an instance
est = LinearDML()
# We don't even need to fit it to call sensitivity_interval
lb, ub, lci, uci = est.sensitivity_interval(theta, theta_stderr, cf_y, cf_d, rho)
print("Theta =", theta)
print("LB:", lb)
print("UB:", ub)
print("LCI:", lci)
print("UCI:", uci)

# For negative estimate
theta2 = -10.0
lb2, ub2, lci2, uci2 = est.sensitivity_interval(theta2, theta_stderr, cf_y, cf_d, rho)
print("\nTheta2 =", theta2)
print("LB2:", lb2)
print("UB2:", ub2)
