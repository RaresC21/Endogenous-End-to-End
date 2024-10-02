import numpy as np


class Agent(object):
    """
    An Agent is able to take one of a set of actions at each time step. The
    action is chosen using a strategy based on the history of prior actions
    and outcome observations.
    """
    def __init__(self, bandit, policy, prior=0, gamma=None):
        self.policy = policy
        self.k = bandit.k
        self.prior = prior
        self.gamma = gamma
        self._value_estimates = prior*np.ones(self.k)
        self.action_attempts = np.zeros(self.k) 
        self.t = 0
        self.last_action = None

    def __str__(self):
        return str(self.policy)

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self._value_estimates[:] = self.prior
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0

    def choose(self):
        action, price = self.policy.choose(self)   # index and price
        self.last_action = action
        return action, price
    
    def initialize(self, action, price):
        self.last_action = action
        return action, price

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1

        N = self.action_attempts[self.last_action]

        q = self._value_estimates[self.last_action]  # this is different for AdaptedUCB vs the usual methods

        self._value_estimates[self.last_action] += (1/N)*(reward - q) # update average: old <- old + (new - old)/n = (n-1)*old/n + new/n
        self.t += 1


    @property
    def value_estimates(self):
        return self._value_estimates

    

class AgentOptimisticLearning(object):
    """Adaptation for OptimisticLearning Policy"""
    def __init__(self, bandit, policy, action_values):
        self.policy = policy
        self.k = bandit.k
        self.actions_list = []
        self.demands_list = []
        self.sum_a = 0
        self.sum_a_squared = 0
        self.sum_d = 0
        self.sum_d_squared = 0
        self.sum_ad = 0
        self.t = 0
        self.last_action = None
        self.action_values = action_values
    
    def __str__(self):
        return str(self.policy)

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self.actions_list = []
        self.demands_list = []
        self.sum_a = 0
        self.sum_a_squared = 0
        self.sum_d = 0
        self.sum_d_squared = 0
        self.sum_ad = 0
        self.last_action = None
        self.t = 0

    def choose(self):
        action, price = self.policy.choose(self)   # index and price
        self.last_action = action
        return action, price
    
    def initialize(self, action, price):
        self.last_action = action
        return action, price

    def observe(self, demand):
        '''Store the sum of past demands and store every action'''
        last_action = self.last_action
        self.actions_list.append(last_action)
        self.demands_list.append(demand)
        last_action_value = self.action_values[last_action]
        self.sum_a_squared += last_action_value**2
        self.sum_a += last_action_value
        self.sum_d += demand
        self.sum_d_squared += demand**2
        self.sum_ad += demand*last_action_value
        self.t += 1


