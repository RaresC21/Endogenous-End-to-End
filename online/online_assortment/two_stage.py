import numpy as np
import gurobipy as gp
import torch
from problem import *
from helper import ProblemGenerator
from sklearn.linear_model import LinearRegression
from itertools import product
import nn_model

class TwoStage:
    def __init__(self, actions, demand, unit_cost):
        self.actions = actions
        self.demand = demand
        self.unit_cost = unit_cost

    def predict_estimator(self):
        ''' minimize mean squared error between predicted demand and actual demand '''
        model = gp.Model("estimator")
        model.setParam('OutputFlag', 0)
        n = len(self.actions)
        p = len(self.actions[0])
        theta = model.addVars(p, p, name="theta", lb=-1e-5)
        bias = model.addVars(p, name="bias", lb=1e-5)
        model.setObjective(sum(sum(self.demand[i, k] - sum(theta[k, j] * self.actions[i, j] for j in range(p)) - bias[k] for k in range(p)) ** 2 for i in range(n)) / n, gp.GRB.MINIMIZE)
        
        model.optimize()

        return np.array([[theta[i, j].x for j in range(p)] for i in range(p)]), np.array([bias[i].x for i in range(p)])
    
    def two_stage(self):
        ''' use estimator to get the optimal actions by minimizing the objective function'''
        theta, bias = self.predict_estimator()
        p = len(self.actions[0])
        model = gp.Model("two_stage")
        model.setParam('OutputFlag', 0)
        w = model.addVars(p, name="w")
        # variable to model the max
        y = model.addVars(p, name="y")

        model.addConstrs(y[i] >= sum(theta[i, j] * w[j] for j in range(p)) + bias[i] - w[i] for i in range(p))
        model.addConstrs(y[i] >= 0 for i in range(p))

        model.setObjective(sum(y[i] + self.unit_cost * w[i] for i in range(p)), gp.GRB.MINIMIZE)

        model.optimize()

        return np.array([w[i].x for i in range(p)]), model.objVal
    
    def decision(self):
        return self.two_stage()
        




class JointOptimization:
    def __init__(self, actions, demand, unit_cost):
        self.actions = actions
        self.demand = demand
        self.unit_cost = unit_cost


    def predict_estimator(self, warm_start=None):
        model = gp.Model("estimator")
        model.setParam('OutputFlag', 0)
        # stop after 60 seconds or gap of 1%
        model.setParam('TimeLimit', 60)
        model.setParam("MIPGap", 1e-2)


        n = len(self.actions)
        p = len(self.actions[0])
        theta = model.addVars(p, p, name="theta", lb=-1e5)
        bias = model.addVars(p, name="bias", lb=-1e5)
        # warm start based on the previous solution
        if warm_start is not None:
            for i in range(p):
                bias[i].Start = warm_start[1][i]
                for j in range(p):
                    theta[i, j].Start = warm_start[0][i][j]

        # max1 models the max(0, ...) quantity
        # bin1 is a binary variable to control which element is equal to the maximum
        max1 = model.addVars(n, p, name="max1")
        bin1 = model.addVars(n, p, name="bin1", vtype=gp.GRB.BINARY)

        # contraints to model the max(0, ...)
        model.addConstrs(max1[i, k] >= sum(theta[k, j] * self.actions[i, j] for j in range(p)) + bias[k] - self.actions[i, k] for i in range(n) for k in range(p))
        model.addConstrs(max1[i, k] >= 0 for i in range(n) for k in range(p))
        model.addConstrs(max1[i, k] <= sum(theta[k, j] * self.actions[i, j] for j in range(p)) + bias[k] - self.actions[i, k] + 100 * (1 - bin1[i, k]) for i in range(n) for k in range(p))
        model.addConstrs(max1[i, k] <= 100 * bin1[i, k] for i in range(n) for k in range(p))


        model.setObjective(
            sum(
                (sum(max1[i, k] for k in range(p)) - 
                 sum(max(0, self.demand[i, k] - self.actions[i, k]) for k in range(p))
                 ) ** 2
                for i in range(n)
            ),
            gp.GRB.MINIMIZE
        )

        model.optimize()
        # print(n, p)
        # print('OBJECTIVE', model.objVal)

        return np.array([[theta[i, j].x for j in range(p)] for i in range(p)]), np.array([bias[i].x for i in range(p)])


    def joint_optimization(self, eps=10**-2, warm_start=None):
        ''' join optimization, add the MSE in the constraint '''
        theta_0, bias_0 = self.predict_estimator(warm_start=warm_start)
        model = gp.Model("joint_opt")
        model.setParam('OutputFlag', 0)
        model.setParam('NonConvex', 2)

        p = len(self.actions[0])
        w = model.addVars(p, name="w")
        theta = model.addVars(p, p, name="theta", lb=-1e5)
        bias = model.addVars(p, name="bias", lb=-1e5)
        y = model.addVars(p, name="y") # for the maximum

        # action should be positive
        model.addConstrs(w[i] >= 0 for i in range(p))
        # theta should be close to the one we got from the estimamtor prediction
        model.addConstr(sum(theta_0[i, j] - theta[i, j] for i in range(p) for j in range(p))**2 
                         + sum(bias_0[i] - bias[i] for i in range(p))**2 <= eps)
        model.addConstrs(y[i] >= sum(theta[i, j] * w[j] for j in range(p)) + bias[i] - w[i] for i in range(p))
        model.addConstrs(y[i] >= 0 for i in range(p))
        
        
        model.setObjective(sum(y[i] + self.unit_cost * w[i] for i in range(p)), gp.GRB.MINIMIZE)
        model.optimize()

        return np.array([w[i].x for i in range(p)]), model.objVal, [[theta[i, j].x for i in range(p)] for j in range(p)], [bias[i].x for i in range(p)]





class KernelPredictor: 
    def __init__(self, actions, demand, unit_cost, capacity, lambd=0.1): 
        self.actions = actions
        self.demand = demand 
        self.unit_cost = unit_cost
        self.capacity = capacity
        self.lambd = lambd
        self.n = len(actions)
        self.K_hat = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.K_hat[i][j] = np.dot(actions[i], actions[j])
        self.c = np.zeros(self.n)
        # define c for the asortment problem
        for i in range(self.n):
            self.c[i] = np.sum(np.maximum(demand[i] - actions[i], 0) + unit_cost * actions[i])

    def predict(self, action):
        K_sample = np.zeros(self.n)
        for i in range(self.n):
            K_sample[i] = np.dot(action, self.actions[i])
        h = K_sample.T @ np.linalg.inv(self.K_hat + self.n * np.eye(self.n) * self.lambd) @ self.c
        return h

    def decision(self): 
        h = [self.predict(action) for action in self.actions]
        best_index = np.argmin(h)
        return self.actions[best_index]




class SampleGD:
    def __init__(self, actions, demand, unit_cost, n_samples = 1): 
        self.actions = torch.tensor(actions) 
        self.demand = torch.tensor(demand)
        self.unit_cost = unit_cost
        self.revs = torch.tensor(ProblemGenerator.objective(actions, demand, unit_cost))

        self.N = len(actions)
        n_items = len(actions[0])

        best_loss = 1e5
        for n in range(n_samples): 
            alpha_init = np.random.randn(n_items, n_items)
            alpha_0_init = np.random.randn(n_items)

            loss, alpha, alpha_0 = self.solve_gd(alpha_init, alpha_0_init)
            
            if loss < best_loss: 
                best_loss = loss 
                best_alpha = np.copy(alpha) 
                best_alpha_0 = np.copy(alpha_0)

        self.alpha = best_alpha 
        self.alpha_0 = best_alpha_0

    def loss(self, actions, revs, alpha, alpha_0): 
        pred_demand = alpha_0 + (actions @ alpha.T)
        cost = torch.sum(torch.nn.functional.relu(pred_demand - actions) + self.unit_cost * actions, dim=1)
        return torch.mean(torch.abs(cost - revs))

    def solve_gd(self, alpha_init, alpha_0_init):
        alpha = torch.nn.Parameter(torch.tensor(alpha_init))
        alpha_0 = torch.nn.Parameter(torch.tensor(alpha_0_init))

        optimizer = torch.optim.SGD([alpha, alpha_0], lr=0.0001, momentum = 0.9)

        epochs = 100
        batch = 10
        
        cur_losses = []
        for e in range(epochs): 
            optimizer.zero_grad() 

            for i in range(0, self.N, batch):
                actions = self.actions[i:i+batch,:]
                revs = self.revs[i:i+batch]

                loss = self.loss(actions, revs, alpha, alpha_0)
                loss.backward() 

                optimizer.step()
                cur_losses.append(loss.item())

                # print("epoch:", e, " ", i, " loss:", np.mean(cur_losses))
        return np.mean(cur_losses[-100:]), alpha.detach().numpy(), alpha_0.detach().numpy()

    def predict(self, action):
        return ProblemGenerator.predict_linear(action, self.alpha, self.alpha_0)

    def decision(self, capacity): 
        return assortment_opt(self.alpha, self.alpha_0, self.unit_cost, capacity)
    


class SampleGD_NN:
    def __init__(self, actions, demand, unit_cost, trained_net, n_samples = 1): 
        self.actions = torch.tensor(actions) 
        self.demand = torch.tensor(demand)
        self.unit_cost = unit_cost
        self.revs = torch.tensor(ProblemGenerator.objective(actions, demand, unit_cost))

        self.N = len(actions)
        n_items = len(actions[0])

        self.first_layers_output = nn_model.get_output_first_layers(trained_net, self.actions.float())  # output for every past action
        parameters, bias = nn_model.get_last_hidden_layer_params(trained_net)

        best_loss = 1e5
        for n in range(n_samples): 
            # alpha_init = np.random.randn(n_items, n_items)
            # alpha_0_init = np.random.randn(n_items)
            alpha_init = parameters.detach().numpy()
            alpha_0_init = bias.detach().numpy()

            loss, alpha, alpha_0 = self.solve_gd(alpha_init, alpha_0_init)
            
            if loss < best_loss: 
                best_loss = loss 
                best_alpha = np.copy(alpha) 
                best_alpha_0 = np.copy(alpha_0)

        self.alpha = best_alpha 
        self.alpha_0 = best_alpha_0

    def loss(self, actions, revs, alpha, alpha_0, last_output): 
        pred_demand = alpha_0 + (last_output @ alpha.T)
        cost = torch.sum(torch.nn.functional.relu(pred_demand - actions) + self.unit_cost * actions, dim=1)
        return torch.mean(torch.abs(cost - revs))

    def solve_gd(self, alpha_init, alpha_0_init):
        alpha = torch.nn.Parameter(torch.tensor(alpha_init))
        alpha_0 = torch.nn.Parameter(torch.tensor(alpha_0_init))

        optimizer = torch.optim.SGD([alpha, alpha_0], lr=0.0001, momentum = 0.9)

        epochs = 1000
        batch = 10
        
        cur_losses = []
        for e in range(epochs): 
            optimizer.zero_grad() 

            for i in range(0, self.N, batch):
                actions = self.actions[i:i+batch,:]
                last_output = self.first_layers_output[i:i+batch]
                revs = self.revs[i:i+batch]

                loss = self.loss(actions, revs, alpha, alpha_0, last_output)
                loss.backward() 

                optimizer.step()
                cur_losses.append(loss.item())

                # print("epoch:", e, " ", i, " loss:", np.mean(cur_losses))
        return np.mean(cur_losses[-100:]), alpha.detach().numpy(), alpha_0.detach().numpy()

    def predict(self, action):
        return ProblemGenerator.predict_linear(action, self.alpha, self.alpha_0)

    def decision(self, capacity): 
        return assortment_opt(self.alpha, self.alpha_0, self.unit_cost, capacity)
    




class SquareCB:
    ''' SquareCB first runs a linear regression for every action, then sample an action from a created probability distribution'''

    def __init__(self, actions, demands, possible_prices, t, mu=20, gamma_scale=20, gamma_exp = 0.7, unit_cost=0.7):
        self.actions = actions
        self.demands = demands
        self.gamma_scale = gamma_scale
        self.gamma_exp = gamma_exp
        self.possible_prices = possible_prices
        self.t = t
        self.unit_cost = unit_cost

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
        self.mu = n_items ** n_actions if n_items != 1 else n_actions
        # one model for each item 
        # to_predict: for each item, n_actions possible prices. So total of n_item ^ n_actions possible samples to predict
        to_predict = [list(x) for x in product(self.possible_prices, repeat=n_items)]
        models = []
        for i in range(n_items):
            actions_train, demands_train = past_actions, past_demands[:, i]
            reg = LinearRegression().fit(actions_train, demands_train)
            models.append(reg)
        # predict total demand for every combination of prices / items
        predictions = []
        for i, prices in enumerate(to_predict):
            demand = 0
            for j in range(n_items):
                demand += max(models[j].predict(np.array(prices).reshape(1, -1))[0] - prices[j], 0) + self.unit_cost * prices[j]
            predictions.append(demand)
        # pick action that maximizes the predictions
        min_index = np.argmin(predictions)
        b = np.min(predictions)
        # define vector of probabilities for every a action different from the max
        p = len(predictions)
        probs = np.zeros(len(predictions))
        gamma = self.gamma_scale #* (self.t**self.gamma_exp) # increasing rate with time
        for i in range(p):
            if i != min_index:
                probs[i] = 1 / (self.mu - gamma * (b-predictions[i]))
        probs[min_index] = 1 - np.sum(probs)
        # softmax 
        # probs = np.exp(probs) / np.sum(np.exp(probs))
        # sample action
        rng = np.random.default_rng()
        price = rng.choice(to_predict, p=probs)
        action = np.where(to_predict == price)[0][0]
        # return action and corresponding index after the sampling
        return action, price