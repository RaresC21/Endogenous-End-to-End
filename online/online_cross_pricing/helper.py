import numpy as np 

class ProblemGenerator: 
    def __init__(self, n_items, prices): 
        self.n_items = n_items  
        self.alpha = np.random.rand(n_items) 
        self.beta = -np.random.rand(n_items) 
        self.gamma = 0.1 * np.random.randn(n_items, n_items) 

        self.prices = prices 
        self.n_prices = prices.shape[0]

    def get_actions(self, n): 
        return np.random.choice(self.prices, n * self.n_items).reshape(n, self.n_items)
    
    def get_nominal_demand(self, actions): 
        return ProblemGenerator.predict_objective(actions, self.alpha, self.beta, self.gamma)

    def get_demand(self, actions): 
        d = self.get_nominal_demand(actions)
        # return d**2
        return np.random.lognormal(d, 0.5)
        # return (d**2) * np.random.lognormal(1, 0.5, size=d.shape)
    
    def get_objective(self, actions): 
        demand = self.get_nominal_demand(actions) 
        return np.sum(demand * actions, axis=1)

    @staticmethod
    def predict_objective(actions, alpha, beta, gamma): 
        a = alpha + actions * beta
        outprod = np.einsum('ij,ik->ijk', actions, actions) * gamma
        return a + np.sum(outprod, axis=-1)
