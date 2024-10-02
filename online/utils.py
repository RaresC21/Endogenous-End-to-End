def g_function(value_estimates, exploration):
    return [value_estimate*exploration_value for value_estimate, exploration_value in zip(value_estimates, exploration)]
