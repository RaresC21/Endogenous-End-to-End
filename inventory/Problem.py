import numpy as np 
import scipy as sp
import random

from torch.distributions.multivariate_normal import MultivariateNormal

from scipy.special import softmax

import torch
import torch.nn.functional as F

def is_psd(mat):
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all())

class ProblemGenerator: 
    def __init__(self, n_locations, h, b, DEVICE): 
        self.n_locations = n_locations  
        self.h = h 
        self.b = b 
        self.DEVICE = DEVICE
        
        self.means = torch.rand(self.n_locations) + 1
        self.vars  = F.softmax(torch.rand(self.n_locations) * 5)
        
        self.actions = torch.tensor([i for i in range(self.n_locations)], device=DEVICE)
        self.actions_ = self.actions.cpu().detach().numpy()
        
    def get_actions(self, n):
        o = np.random.choice(self.actions_, n)
        return torch.tensor(o).to(self.DEVICE).long()

    def get_demand(self, n, corr = 1):
        x = torch.randn(self.n_locations) * 0.1
        x[0] = corr
        x = x.unsqueeze(1)
        
        covar = x @ x.T + torch.eye(self.n_locations) * 0.1
        print(covar)
        print("PSD:", is_psd(covar))
        
        print(self.means.shape, covar.shape)
        sampler = MultivariateNormal(self.means, covar) 
        return sampler.sample(torch.Size([n])).to(self.DEVICE) #* self.vars.unsqueeze(0).repeat(n, 1)
        

    def get_objective(self, v, z): 
        return torch.mean(self.h * F.relu(v - z) + self.b * F.relu(z - v), dim=1)
    