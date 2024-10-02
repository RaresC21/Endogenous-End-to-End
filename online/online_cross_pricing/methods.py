
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import product

from problem import *

# def make_features(prices): 
    # outprod = np.einsum('ij,ik->ijk', prices, prices)
    # for i in range(len(outprod)): 
    #     outprod[i,:,:] = np.fill_diagonal(outprod[i,:,:], 0)
    # print(outprod)
    # return np.concatenate((prices, outprod.reshape(len(prices), -1)), axis=1)

class LinearPredictor: 
    def __init__(self, prices, price_data, demand_data): 
        self.prices = prices
        self.price_data = price_data
        self.demand_data = demand_data
        
        self.n_prices = prices.shape[0] 
        self.n_items = price_data.shape[1]

        self.reg = LinearRegression().fit(price_data, demand_data)

    def predict(self, prices): 
        return self.reg.predict(prices)

    def get_params(self): 
        alpha = self.reg.intercept_
        beta = self.reg.coef_.diagonal()
        gamma = np.copy(self.reg.coef_)
        np.fill_diagonal(gamma, 0)
        return alpha, beta, gamma

    def decision(self): 
        alpha, beta, gamma = self.get_params()
        return pricing_opt(alpha, beta, gamma, self.prices)


class RobustE2E: 
    def __init__(self, prices, price_data, demand_data): 
        self.prices = prices
        self.price_data = price_data
        self.demand_data = demand_data
        
        self.n_prices = prices.shape[0] 
        self.n_items = price_data.shape[1]
        self.n_data = price_data.shape[0]

        self.revenues = self.demand_data * self.price_data
        self.eps0 = self.bound()

    def bound(self): 
        model = gp.Model("upperbound")
        model.setParam('OutputFlag', 0)
        alpha = model.addVars(self.n_items, vtype=gp.GRB.CONTINUOUS, name="alpha", lb=-200)
        beta  = model.addVars(self.n_items, vtype=gp.GRB.CONTINUOUS, name="beta", lb=-200)
        gamma = model.addVars(self.n_items, self.n_items, vtype=gp.GRB.CONTINUOUS, name="gamma", lb=-200)

        model.setObjective(sum(sum( (self.price_data[n,i] * (alpha[i] + self.price_data[n,i]*beta[i] + sum(gamma[i,j] * self.price_data[n,j] for j in range(self.n_items))) - self.revenues[n,i])**2 for i in range(self.n_items) ) for n in range(self.n_data)))
        model.optimize() 
        return model.objVal

    def decision(self, eps=0.1): 
        def cutting_plane(action):
            outer_problem = self.setup_outer()

            # last_upper = -1e4

            best_action = None 
            best_val = -1e2
            last_upper = 1e2
            for _ in range(40):
                lower_bound, alpha_, beta_, gamma_ = self.g(action, eps)
                upper_bound, action = outer_problem((alpha_, beta_, gamma_))

                if lower_bound > best_val: 
                    best_val = lower_bound 
                    best_action = action 

            return best_val, best_action

        best_val = -1e5
        for n, initial_action in enumerate(self.price_data[:1]): 
            val, a = cutting_plane(initial_action)
            if val > best_val: 
                best_val = val 
                decision = a
        return decision 

    def g(self, action, eps): 
        model = gp.Model("upperbound")
        model.setParam('OutputFlag', 0)
        alpha = model.addVars(self.n_items, vtype=gp.GRB.CONTINUOUS, name="alpha", lb=-200)
        beta  = model.addVars(self.n_items, vtype=gp.GRB.CONTINUOUS, name="beta", lb=-200)
        gamma = model.addVars(self.n_items, self.n_items, vtype=gp.GRB.CONTINUOUS, name="gamma", lb=-200)
        for i in range(self.n_items): model.addConstr(gamma[i,i] == 0)

        model.setObjective(sum(action[i] * (alpha[i] + beta[i] * action[i] + sum(gamma[i,j] * action[j] for j in range(self.n_items))) for i in range(self.n_items)), gp.GRB.MINIMIZE)

        model.addConstr(sum(sum( (self.price_data[n,i] * (alpha[i] + self.price_data[n,i]*beta[i] + sum(gamma[i,j] * self.price_data[n,j] for j in range(self.n_items))) - self.revenues[n,i])**2 for i in range(self.n_items) ) for n in range(self.n_data)) <= self.eps0 * (1 + eps))
        model.optimize() 
        # print("STATUS:", model.status)

        return model.objVal, \
                np.array([alpha[i].x for i in range(self.n_items)]), \
                np.array([beta[i].x for i in range(self.n_items)]), \
                np.array([[gamma[i,j].x for j in range(self.n_items)] for i in range(self.n_items)])

    def setup_outer(self): 
        model = gp.Model("upperbound")
        model.setParam('OutputFlag', 0)
        x = model.addVars(self.n_items, self.n_prices, vtype=gp.GRB.BINARY)
        y = model.addVars(self.n_items, self.n_items, self.n_prices, self.n_prices, vtype=gp.GRB.BINARY)
        ob = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-10000)

        for i in range(self.n_items):
            model.addConstr(sum(x[i,t] for t in range(self.n_prices)) == 1)
        
        for i in range(self.n_items): 
            for j in range(self.n_items): 
                for t in range(self.n_prices): 
                    model.addConstr(sum(y[i,j,t,tt] for tt in range(self.n_prices)) == x[i,t])
                    model.addConstr(sum(y[i,j,tt,t] for tt in range(self.n_prices)) == x[j,t])
        model.setObjective(ob, gp.GRB.MAXIMIZE)

        # add u = alpha, beta, gamma as new cut and re-solve
        def update_outer(u):
            alpha, beta, gamma = u
            model.addConstr(ob <= sum(alpha[i] * sum(self.prices[t] * x[i,t] for t in range(self.n_prices)) + beta[i] * sum((self.prices[t]**2) * x[i,t] for t in range(self.n_prices)) for i in range(self.n_items))
                + sum(sum(sum(sum(gamma[i,j] * self.prices[t1] * self.prices[t2] * y[i,j,t1,t2] for t2 in range(self.n_prices)) for t1 in range(self.n_prices)) for j in range(self.n_items)) for i in range(self.n_items)))

            model.update()
            model.optimize() 
            # print("STATUS:", model.status)
            x_sol = np.array([[x[i,j].x for j in range(self.n_prices)] for i in range(self.n_items)])
            return model.objVal, np.array([self.prices[np.argmax(x_sol[i,:])] for i in range(self.n_items)])

        return update_outer

class Robust2Stage: 
    def __init__(self, prices, price_data, demand_data): 
        self.prices = prices
        self.price_data = price_data
        self.demand_data = demand_data
        
        self.n_prices = prices.shape[0] 
        self.n_items = price_data.shape[1]
        self.n_data = price_data.shape[0]

        linear_predictor = LinearPredictor(self.prices, self.price_data, self.demand_data)
        self.alpha_star, self.beta_star, self.gamma_star = linear_predictor.get_params()
        self.eps0 = self.bound()

    def bound(self): 
        model = gp.Model("upperbound")
        model.setParam('OutputFlag', 0)
        alpha = model.addVars(self.n_items, vtype=gp.GRB.CONTINUOUS, name="alpha", lb=-200)
        beta  = model.addVars(self.n_items, vtype=gp.GRB.CONTINUOUS, name="beta", lb=-200)
        gamma = model.addVars(self.n_items, self.n_items, vtype=gp.GRB.CONTINUOUS, name="gamma", lb=-200)

        alpha_star, beta_star, gamma_star = self.alpha_star, self.beta_star, self.gamma_star

        model.setObjective(sum( (alpha_star[i] - alpha[i])**2 + (beta_star[i] - beta[i])**2 + sum((gamma_star[i,j] - gamma[i,j])**2 for j in range(self.n_items)) for i in range(self.n_items)))
        model.optimize() 
        return model.objVal

    def decision(self, eps=0.1): 
        u_set = [(self.alpha_star, self.beta_star, self.gamma_star)]

        outer_problem = self.setup_outer()

        best_action = None 
        best_val = -1e2
        last_upper = 1e2
        for _ in range(40):
            upper_bound, action = outer_problem(u_set[-1])
            lower_bound, alpha_, beta_, gamma_ = self.g(action, eps)
            u_set.append((alpha_, beta_, gamma_))
            # print("bound:", upper_bound, lower_bound, upper_bound - lower_bound)

            if lower_bound > best_val: 
                best_val = lower_bound 
                best_action = action 
            if upper_bound > last_upper - 1e-3: break
            last_upper = upper_bound

        return best_action

    def g(self, action, eps): 
        alpha_star, beta_star, gamma_star = self.alpha_star, self.beta_star, self.gamma_star

        model = gp.Model("upperbound")
        model.setParam('OutputFlag', 0)
        alpha = model.addVars(self.n_items, vtype=gp.GRB.CONTINUOUS, name="alpha", lb=-200)
        beta  = model.addVars(self.n_items, vtype=gp.GRB.CONTINUOUS, name="beta", lb=-200)
        gamma = model.addVars(self.n_items, self.n_items, vtype=gp.GRB.CONTINUOUS, name="gamma", lb=-200)
        for i in range(self.n_items): model.addConstr(gamma[i,i] == 0)

        model.setObjective(sum(action[i] * (alpha[i] + beta[i] * action[i] + sum(gamma[i,j] * action[j] for j in range(self.n_items))) for i in range(self.n_items)), gp.GRB.MINIMIZE)
        
        model.addConstr(sum( (alpha_star[i] - alpha[i])**2 for i in range(self.n_items)) + sum((beta_star[i] - beta[i])**2 for i in range(self.n_items)) + sum(sum((gamma_star[i,j] - gamma[i,j])**2 for j in range(self.n_items)) for i in range(self.n_items)) <= eps)
        # model.addConstr(sum(sum( ((alpha[i] + self.price_data[n,i]*beta[i] + sum(gamma[i,j] * self.price_data[n,j] for j in range(self.n_items))) - self.demand_data[n,i])**2 for i in range(self.n_items) ) for n in range(self.n_data)) <= self.eps0 * (1 + eps))
        model.optimize() 
        # print("STATUS:", model.status)
        return model.objVal, \
                np.array([alpha[i].x for i in range(self.n_items)]), \
                np.array([beta[i].x for i in range(self.n_items)]), \
                np.array([[gamma[i,j].x for j in range(self.n_items)] for i in range(self.n_items)])

    def setup_outer(self): 
        model = gp.Model("upperbound")
        model.setParam('OutputFlag', 0)
        x = model.addVars(self.n_items, self.n_prices, vtype=gp.GRB.BINARY)
        y = model.addVars(self.n_items, self.n_items, self.n_prices, self.n_prices, vtype=gp.GRB.BINARY)
        ob = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-10000)

        for i in range(self.n_items):
            model.addConstr(sum(x[i,t] for t in range(self.n_prices)) == 1)
        
        for i in range(self.n_items): 
            for j in range(self.n_items): 
                for t in range(self.n_prices): 
                    model.addConstr(sum(y[i,j,t,tt] for tt in range(self.n_prices)) == x[i,t])
                    model.addConstr(sum(y[i,j,tt,t] for tt in range(self.n_prices)) == x[j,t])
        model.setObjective(ob, gp.GRB.MAXIMIZE)

        # add u = alpha, beta, gamma as new cut and re-solve
        def update_outer(u):
            alpha, beta, gamma = u
            model.addConstr(ob <= sum(alpha[i] * sum(self.prices[t] * x[i,t] for t in range(self.n_prices)) + beta[i] * sum((self.prices[t]**2) * x[i,t] for t in range(self.n_prices)) for i in range(self.n_items))
                + sum(sum(sum(sum(gamma[i,j] * self.prices[t1] * self.prices[t2] * y[i,j,t1,t2] for t2 in range(self.n_prices)) for t1 in range(self.n_prices)) for j in range(self.n_items)) for i in range(self.n_items)))

            model.update()
            model.optimize() 

            x_sol = np.array([[x[i,j].x for j in range(self.n_prices)] for i in range(self.n_items)])
            return model.objVal, np.array([self.prices[np.argmax(x_sol[i,:])] for i in range(self.n_items)])

        return update_outer
    




class SquareCB:
    ''' SquareCB first runs a linear regression for every action, then sample an action from a created probability distribution'''

    def __init__(self, actions, demands, possible_prices, t, mu=20, gamma_scale=10, gamma_exp = 0.7):
        self.actions = actions
        self.demands = demands
        self.gamma_scale = gamma_scale
        self.gamma_exp = gamma_exp
        self.possible_prices = possible_prices
        self.t = t

    def __str__(self):
        return 'SquareCB'
    
    def choose(self):
        # linear regression for every action
        n_items = self.actions.shape[1]
        n_actions = self.possible_prices.shape[0]
        past_actions  = self.actions
        past_demands = self.demands
        # maximize the g function (demand * price)
        # features are prices and prices squared
        self.mu = n_items ** n_actions
        # one model for each item 
        # to_predict: for each item, n_actions possible prices. So total of n_item ^ n_actions possible samples to predict
        to_predict = [list(x) for x in product(self.possible_prices, repeat=n_items)]
        models = []
        for i in range(n_items):
            actions_train, demands_train = past_actions, past_demands[:, i]
            reg = LinearRegression().fit(actions_train, demands_train)
            models.append(reg)
        # predict total demand for every combination of prices
        predictions = []
        for i, prices in enumerate(to_predict):
            demand = 0
            for j in range(n_items):
                demand += prices[j] * models[j].predict(np.array(prices).reshape(1, -1))
            predictions.append(demand[0])
        # pick action that maximizes the predictions
        max_index = np.argmax(predictions)
        b = np.max(predictions)
        # define vector of probabilities for every a action different from the max
        p = len(predictions)
        probs = np.zeros(len(predictions))
        gamma = self.gamma_scale * (self.t**self.gamma_exp) # increasing rate with time
        for i in range(p):
            if i != max_index:
                probs[i] = 1 / (self.mu + gamma * (b-predictions[i]))
        probs[max_index] = 1 - np.sum(probs)
        # softmax 
        # probs = np.exp(probs) / np.sum(np.exp(probs))
        # sample action
        rng = np.random.default_rng()
        price = rng.choice(to_predict, p=probs)
        action = np.where(to_predict == price)[0][0]
        # return action and corresponding index after the sampling
        return action, price