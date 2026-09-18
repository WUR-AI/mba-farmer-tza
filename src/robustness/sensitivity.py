def run_sensitivity_analysis(estimator_instance):
    """
    Runs sensitivity analysis on the fitted estimator.
    """
    if hasattr(estimator_instance, 'est') and hasattr(estimator_instance.est, 'sensitivity_summary'):
        summary_str = str(estimator_instance.est.sensitivity_summary())
        
        import re
        rv_theta = None
        rv_ci = None
        
        lines = summary_str.split('\n')
        for i, line in enumerate(lines):
            if "Robustness Value (Theta)" in line and "Robustness Value (CI)" in line:
                # Values are typically 2 lines down due to the separator line
                for j in range(1, 4):
                    if i + j < len(lines):
                        # Find all floating point numbers on this line
                        vals = re.findall(r'0\.\d+', lines[i+j])
                        if len(vals) >= 2:
                            rv_theta = float(vals[0])
                            rv_ci = float(vals[1])
                            break
                break
                
        return {'Robustness Value (Theta)': rv_theta, 'Robustness Value (CI)': rv_ci}
    else:
        return None
