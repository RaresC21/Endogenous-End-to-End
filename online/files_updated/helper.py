import numpy as np 
import scipy as sp

class ProblemGenerator: 
    def __init__(self, n_items, unit_cost = 0.3): 
        self.n_items = n_items  
        self.unit_cost = unit_cost
        
        self.base_demand = np.random.rand(n_items) 
        self.alpha = np.random.randn(n_items, n_items) * 0.5
        # self.alpha = np.random.choice([-1,0,1], size=(n_items, n_items)) * (1/n_items)

        # self.base_action = np.random.choice([0.4, 0.5, 0.6], self.n_items)
        self.base_action = np.ones(n_items)

        for i in range(n_items): self.alpha[i,i] = 0
        
    def get_actions(self, n): 
        # diff = sp.sparse.rand(n, self.n_items, density=0.1, format='csr') * 0.5
        # return np.array(diff - self.base_action)

        return np.random.choice([0.5, 0.6, 0.7, 0.8,0.9,1.0], n*self.n_items).reshape(n, self.n_items)
        # return np.random.rand(n * self.n_items).reshape(n, self.n_items)

    def get_nominal_demand(self, actions): 
        return ProblemGenerator.predict(actions, self.alpha, self.base_demand) 

    def get_demand(self, actions): 
        d = self.get_nominal_demand(actions)
        d = np.random.lognormal(d, 0.5)
        # d = np.random.normal(d, 0.5)
        return d
        # return d * self.capacity / np.mean(np.sum(d, axis=1))

    def get_objective(self, actions, quantile=None): 
        np.random.seed(0)
        actions = np.tile(actions, (1000,1))
        demand = self.get_demand(actions) 
        obj = ProblemGenerator.objective(actions, demand, self.unit_cost)
        # return [np.mean(obj)]
        if quantile is None: 
            return [np.mean(obj)]
        return [np.quantile(obj, quantile)] 
    

    # def evaluate(self, actions, revs, alpha, alpha_0): 
    #     predicted_demand = ProblemGenerator.predict(actions, alpha, alpha_0)
    #     predicted_objective = ProblemGenerator.objective(actions, predicted_demand, self.unit_cost)
    #     # true_objective = self.get_objective(actions) 
    #     return np.mean(np.abs(predicted_objective - revs))

    # @staticmethod
    # def evaluate(actions, demand, alpha, alpha_0): 
    #     predicted_demand = ProblemGenerator.predict(actions, alpha, alpha_0)
    #     predicted_objective = ProblemGenerator.objective(actions, predicted_demand, self.unit_cost)
    #     true_objective = 

    @staticmethod
    def objective(actions, demand, unit_cost): 
        return np.sum(np.maximum(demand - actions, 0) + unit_cost * actions, axis=1)

    @staticmethod
    def predict(actions, alpha, alpha_0): 
        a = (actions @ alpha.T) 
        return (alpha_0 + a) 
    
    @staticmethod
    def predict_linear(actions, alpha, alpha_0): 
        a = (actions @ alpha.T)
        return (alpha_0 + a) 
    