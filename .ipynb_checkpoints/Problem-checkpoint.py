import numpy as np 
import scipy as sp
import random

from scipy.special import softmax

import torch
import torch.nn.functional as F

class ProblemGenerator: 
    def __init__(self, n_locations, h, b, DEVICE): 
        self.n_locations = n_locations  
        self.h = h 
        self.b = b 
        self.DEVICE = DEVICE
        
        self.means = np.random.rand(self.n_locations) + 1
        self.vars  = softmax(np.random.rand(self.n_locations) * 5)
        
        self.actions = [i for i in range(self.n_locations)]
        
    def get_actions(self, n):
        o = np.random.choice(self.actions, n)
        return torch.tensor(o).to(self.DEVICE).long()

    def get_demand(self, n):
        v = np.expand_dims(self.vars, axis=0)
        m = np.expand_dims(self.means, axis=0)
        o = np.random.randn(n, self.n_locations) * np.repeat(v, n, axis=0) + np.repeat(m, n, axis=0)
        return torch.tensor(o).to(self.DEVICE).float()

    def get_objective(self, v, z): 
        return torch.mean(self.h * F.relu(v - z) + self.b * F.relu(z - v), dim=1)
    