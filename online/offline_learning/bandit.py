import numpy as np


class MultiArmedBandit(object):
    """
    A Multi-armed Bandit
    """
    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal = 0

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action):
        return 0, True


class GaussianBandit(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution with
    provided mean and standard deviation.
    """
    def __init__(self, k, mu, sigma, actions): 
        super(GaussianBandit, self).__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.actions = actions
        self.optimal = np.argmax([self.mu[i] * self.actions[i] for i in range(self.k)])
        self.reset()

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma, self.k)
        return self.action_values

    def pull(self, action, price):
        return (self.action_values[action], price*self.action_values[action],   # different methods don't OBSERVE the same thing
                action == self.optimal)
    
    def optimal_reward(self):
        # index of optimal expected reward (vector mu)
        return np.max([self.mu[i] * self.actions[i] for i in range(self.k)])

