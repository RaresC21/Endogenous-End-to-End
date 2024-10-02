import numpy as np
import random
from tqdm import tqdm
from time import time
from get_data import random_data


class Environment(object):
    def __init__(self, bandit, agents, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.agents = agents
        self.label = label

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def run(self, actions, trials=100, experiments=1):
        scores = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(scores)

        for _ in tqdm(range(experiments)):
            self.reset()
            data_actions, data_demands, prices, _ = random_data(len(actions))
            # first step: the agent behaves according to the randomly generated data
            for agent in self.agents:
                if str(agent) == 'Robust Learning':
                    for i, action in enumerate(data_actions):
                        price = prices[action]
                        demand = data_demands[i]
                        action, price = agent.initialize(action, price)
                        agent.observe(demand)

                if 'Naive' in str(agent):
                    data_actions_values = [prices[action] for action in data_actions]
                    alpha_0, alpha_1 = agent.initialize(data_actions_values, data_demands)

                if str(agent) == 'Robust Learning NN':
                    for i, action in enumerate(data_actions):
                        price = prices[action]
                        demand = data_demands[i]
                        action, price = agent.initialize(action, price)
                        agent.observe(demand)
                    data_actions_values = [prices[action] for action in data_actions]
                    trained_net = agent.train_initial_nn(data_actions_values, data_demands)

            # second step: the agent behaves according to its policy
            for t in range(1, trials):
                for i, agent in enumerate(self.agents):
                    if str(agent) == 'Robust Learning':
                        self.bandit.reset()
                        action, price = agent.choose()  # index and corresponding action (price) 
                        reward_demand, reward, is_optimal = self.bandit.pull(action, price)
                        #agent.observe(reward_demand) # different for UCB and adapted UCB (different observed reward)

                    if str(agent) == 'Naive Agent':
                        self.bandit.reset()
                        action, price = agent.choose(alpha_0, alpha_1)
                        demand = data_demands[i]
                        reward_demand, reward, is_optimal = self.bandit.pull(action, price)

                    if str(agent) == 'Robust Learning NN':
                        self.bandit.reset()
                        action, price = agent.choose(trained_net)
                        demand = data_demands[i]
                        reward_demand, reward, is_optimal = self.bandit.pull(action, price)
                        #agent.observe(demand)

                    

                    scores[t, i] += reward 
                    if is_optimal:
                        optimal[t, i] += 1


        return scores / experiments, optimal / experiments