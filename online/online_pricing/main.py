import os
import sys
from agent import Agent, AgentOptimisticLearning, AgentSquareCB
from bandit import GaussianBandit
from environment import Environment
from policy import (EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy, UCBPolicy, AdaptedUCBPolicy, LinUCBPolicy, OptimisticLearningPolicy, SquareCB)
from data_generation import generate_data
from utils import g_function


n_arms = 50
var_coeff = 0.2
prices, parameters = generate_data(n_arms, mean='linear', var_coeff=var_coeff)

actions = prices
mu, sigma = zip(*parameters)
print("Prices: {}".format(actions))
print("Means: {}".format(mu))
print("Standard deviations: {}".format(sigma))
bandit = GaussianBandit(n_arms, mu, sigma, actions)
n_trials = 3000
n_experiments = 100


# print the optimal reward (by selecting the best price)
print('Optimal expected reward: {}'.format(bandit.optimal_reward()))

agents = [
    Agent(bandit, EpsilonGreedyPolicy(0.1, actions)),
    Agent(bandit, UCBPolicy(actions)),
    #Agent(bandit, RandomPolicy(actions)),
    Agent(bandit, AdaptedUCBPolicy(actions, g_function)),
    AgentOptimisticLearning(bandit, OptimisticLearningPolicy(actions), actions),
    # LinUCBPolicy(actions, n_arms, mu, sigma),
    AgentSquareCB(bandit, SquareCB(actions, n_arms, mu=n_arms))
]
env = Environment(bandit, agents, f'SquareCB vs other methods - n_arms={n_arms}, var_coeff={var_coeff}')
scores, optimal, distance_to_optimal = env.run(actions, n_trials, n_experiments)
env.plot_results(scores, optimal, distance_to_optimal)
