from econml.validate.sensitivity_analysis import sensitivity_interval
import numpy as np

theta = 10.0
theta_stderr = 1.0
cy = np.array([0.1, 0.2])
ct = np.array([0.1, 0.2])

lb_theta, ub_theta = sensitivity_interval(
    theta=theta, theta_stderr=theta_stderr, c_y=cy, c_t=ct, rho=1.0, interval_type='theta'
)
lb_ci, ub_ci = sensitivity_interval(
    theta=theta, theta_stderr=theta_stderr, c_y=cy, c_t=ct, rho=1.0, interval_type='ci'
)

print("theta:", lb_theta, ub_theta)
print("ci:", lb_ci, ub_ci)
