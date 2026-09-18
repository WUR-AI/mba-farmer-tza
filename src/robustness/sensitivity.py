def run_sensitivity_analysis(estimator_instance):
    """
    Runs sensitivity analysis on the fitted estimator.
    """
    if hasattr(estimator_instance, 'est') and hasattr(estimator_instance.est, 'sensitivity_summary'):
        # Just return the summary or a string representation
        return estimator_instance.est.sensitivity_summary()
    else:
        return "Sensitivity analysis not available for this estimator."
