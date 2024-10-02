import os
import sys
from agent import Agent, AgentOptimisticLearning
from bandit import GaussianBandit
from environment import Environment
from policy import (EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy, UCBPolicy, AdaptedUCBPolicy, LinUCBPolicy, OptimisticLearningPolicy)
from data_generation import generate_data
from utils import g_function


n_arms = 10

prices, parameters = generate_data(n_arms)

actions = prices
mu, sigma = zip(*parameters)
print("Prices: {}".format(actions))
print("Means: {}".format(mu))
print("Standard deviations: {}".format(sigma))
bandit = GaussianBandit(n_arms, mu, sigma, actions)
n_trials = 2000
n_experiments = 500


# print the optimal reward (by selecting the best price)
print('Optimal expected reward: {}'.format(bandit.optimal_reward()))

agents = [
    Agent(bandit, EpsilonGreedyPolicy(0.1, actions)),
    Agent(bandit, UCBPolicy(actions)),
    #Agent(bandit, RandomPolicy(actions)),
    Agent(bandit, AdaptedUCBPolicy(actions, g_function)),
    # AgentOptimisticLearning(bandit, OptimisticLearningPolicy(actions)),
    # LinUCBPolicy(actions, n_arms, mu, sigma)
]
env = Environment(bandit, agents, 'Adapted UCB vs other methods')
scores, optimal = env.run(actions, n_trials, n_experiments)
env.plot_results(scores, optimal)
