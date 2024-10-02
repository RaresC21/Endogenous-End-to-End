import numpy


def random_data(n_arms, mean='linear', var_coeff=1, n_data=100):
    ''' Randomly pick a price n_data times and observe the corresponding demand '''
    prices = numpy.linspace(0, 1, n_arms)
    demand_distributions = []
    # Generate Gaussian demand distributions for each price
    for price in prices:
        if mean == 'linear':
            mean_demand = (1 - price) # decreasing function of price
        else:
            mean_demand = (1 - price)**2
        std_demand = mean_demand * var_coeff
        demand_distribution = (mean_demand, std_demand)
        demand_distributions.append(demand_distribution)

    data_actions = []
    data_demands = []

    for _ in range(n_data):
        action = numpy.random.randint(0, n_arms)
        price = prices[action]
        demand = numpy.random.normal(demand_distributions[action][0], demand_distributions[action][1])
        data_actions.append(action)
        data_demands.append(demand)

    return data_actions, data_demands, prices, demand_distributions


def online_learning_data(agent):
    ''' Use the agent's actions and demands to generate data '''
    pass
