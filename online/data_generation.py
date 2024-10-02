import random
import numpy as np


def generate_data(n_arms, mean='linear', var_coeff=1):
    # Set the number of price points

    prices = []
    demand_distributions = []

    # Generate random prices
    # for _ in range(n_arms):
    #     price = round(random.uniform(0, 5), 2)  # Random price
    #     # normalize price
    #     price = price / 5
    #     prices.append(price)

    # generate n_arms equidistant prices between 0 and 1
    prices = np.linspace(0, 1, n_arms)

    # Generate Gaussian demand distributions for each price
    for price in prices:
        if mean == 'linear':
            mean_demand = (1 - price) # decreasing function of price
        else:
            mean_demand = (1 - price)**2
        std_demand = mean_demand * var_coeff
        demand_distribution = (mean_demand, std_demand)
        demand_distributions.append(demand_distribution)

    return prices, demand_distributions
