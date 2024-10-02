import numpy as np 
from helper import * 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import gurobipy as gp

from problem import *
from helper import ProblemGenerator

import torch

class Predictor: 
    def __init__(self, alpha, alpha_0): 
        self.alpha = alpha 
        self.alpha_0 = alpha_0

class LinearPredictor: 
    def __init__(self, actions, demand, unit_cost, capacity): 
        self.actions = actions
        self.demand = demand 
        self.unit_cost = unit_cost
        self.capacity = capacity
        
        self.reg = LinearRegression().fit(actions, demand)
        self.alpha, self.alpha_0 = self.get_params()

    def predict(self, actions): 
        return self.reg.predict(actions)

    def get_params(self): 
        alpha_0 = self.reg.intercept_
        alpha = np.copy(self.reg.coef_)
        return alpha, alpha_0

    def decision(self): 
        return assortment_opt(self.alpha, self.alpha_0, self.unit_cost, self.capacity)

class ObjectivePredictor:
    def __init__(self, n_items, unit_cost, capacity, warm_start = None, REG = 0): 
        self.n_items = n_items
        self.unit_cost = unit_cost
        self.capacity = capacity

        model = gp.Model()
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 0.1*60)
        model.setParam('MIPGap', 0.0)
        self.alpha_0_var = model.addVars(n_items, vtype=gp.GRB.CONTINUOUS, name="alpha_0", lb=-5, ub=5)
        self.alpha_var = model.addVars(n_items, n_items, vtype=gp.GRB.CONTINUOUS, name="alpha", lb=-5, ub=5)
        for i in range(n_items): model.addConstr(self.alpha_var[i,i] == 0)

        if warm_start is not None:
            for i in range(n_items): 
                self.alpha_0_var[i].Start = warm_start.alpha_0[i]
                for k in range(n_items): 
                    self.alpha_var[i,k].Start = warm_start.alpha[i,k]
                self.alpha_var[i,i].Start = 0
        self.model = model 

        self.vs = []

    def add_data(self, actions, demand): 
        N = len(actions)
        n_items = len(actions[0])

        y = self.model.addVars(N, n_items, vtype=gp.GRB.BINARY)
        p = self.model.addVars(N, n_items, vtype=gp.GRB.CONTINUOUS, lb=-1e5)
        v = self.model.addVars(N, vtype=gp.GRB.CONTINUOUS)
        pred = self.model.addVars(N, n_items, vtype=gp.GRB.CONTINUOUS, lb=-1e5)

        M = 10

        revs = ProblemGenerator.objective(actions, demand, self.unit_cost)

        for n, (s, d, r) in enumerate(zip(actions, demand, revs)):
            # print(s)
            for k in range(n_items):
                self.model.addConstr(pred[n,k] <= M)
                self.model.addConstr(pred[n,k] == self.alpha_0_var[k] + sum(self.alpha_var[k,j] * s[j] for j in range(n_items)))
                self.model.addConstr(p[n,k] >= pred[n,k] - s[k])
                self.model.addConstr(p[n,k] >= 0)
                self.model.addConstr(p[n,k] <= 0 + (1 - y[n,k]) * M)
                self.model.addConstr(p[n,k] <= pred[n,k] - s[k] + y[n,k] * M)

            self.model.addConstr(v[n] >= (sum(p[n,k] + self.unit_cost * s[k] for k in range(n_items))) - r)
            self.model.addConstr(v[n] >= r - (sum(p[n,k] + self.unit_cost * s[k] for k in range(n_items))))
            self.vs.append(v[n])

        T = len(self.vs)
        self.model.setObjective(sum(self.vs[n] for n in range(T))/T, gp.GRB.MINIMIZE)

        self.model.update()
        self.model.optimize() 

        self.alpha_0 = np.array([self.alpha_0_var[i].x for i in range(n_items)])
        self.alpha = np.array([[self.alpha_var[i,k].x for k in range(n_items)] for i in range(n_items)])
        return Predictor(self.alpha, self.alpha_0)

    def predict(self, action): 
        return ProblemGenerator.predict_linear(action, self.alpha, self.alpha_0)

    def decision(self, capacity): 
        return assortment_opt(self.alpha, self.alpha_0, self.unit_cost, capacity)

    def evaluate(self, alpha, alpha_0): 
        p = ProblemGenerator.predict_linear(self.actions, alpha, alpha_0)
        predicted_objective = ProblemGenerator.objective(self.actions, p, self.unit_cost)
        loss = np.mean(np.abs(predicted_objective - self.revs))
        return loss


class RobustPredictor: 
    def __init__(self, actions, demand, unit_cost, capacity, base_predictor, e2e = True): 
        self.actions = actions
        self.demand = demand 
        self.unit_cost = unit_cost
        self.capacity = capacity
        self.e2e = e2e

        self.base_predictions = base_predictor.predict(actions)
        self.base_alpha = base_predictor.alpha
        self.base_alpha_0 = base_predictor.alpha_0

        print(self.base_alpha)
        print(self.base_alpha_0)

        self.N = len(actions)
        self.n_items = len(actions[0])

    def decision(self, eps=0.1): 
        def cutting_plane(action): 

            outer_problem = self.setup_outer()

            best_action = None 
            best_val = 1e5
            for i in range(5):
                upper_bound, alpha_, alpha_0_ = self.g(action, eps)
                lower_bound, action = outer_problem((alpha_, alpha_0_))

                if upper_bound < best_val: 
                    best_val = lower_bound 
                    best_action = action 
                last_upper = upper_bound

            return best_val, best_action
        
        best_val = 1e5
        for n, initial_action in enumerate(self.actions[:20]): 
            val, a = cutting_plane(initial_action)
            if val < best_val: 
                best_val = val 
                decision = a
        return decision 

    def g(self, action, eps): 
        n_items = self.n_items
        model = gp.Model("upperbound")
        model.setParam('OutputFlag', 0)
        alpha_0 = model.addVars(n_items, vtype=gp.GRB.CONTINUOUS, name="alpha_0", lb=-20, ub=20)
        alpha = model.addVars(n_items, n_items, vtype=gp.GRB.CONTINUOUS, name="alpha", lb=-50, ub=50)
        if self.e2e:
            for i in range(n_items): model.addConstr(alpha[i,i] == 0)
        
        y = model.addVars(n_items, vtype=gp.GRB.BINARY)
        p = model.addVars(n_items, vtype=gp.GRB.CONTINUOUS)
        pred = model.addVars(n_items, vtype=gp.GRB.CONTINUOUS, lb=-1e5)

        M = 10
    
        for k in range(n_items):
            model.addConstr(pred[k] <= M)
            model.addConstr(pred[k] == alpha_0[k] + sum(alpha[k,j] * (0 if j == k else action[j]) for j in range(n_items)))
            model.addConstr(p[k] >= pred[k] - action[k])
            # model.addConstr(p[k] >= 0) implicit
            model.addConstr(p[k] <= pred[k] - action[k] + (1 - y[k]) * M)
            model.addConstr(p[k] <= 0 + y[k] * M)

        model.setObjective(sum(p[k] + action[k] * self.unit_cost for k in range(n_items)), gp.GRB.MAXIMIZE)
        model.addConstr(sum(sum((alpha[j,k] - self.base_alpha[j,k])**2 for j in range(n_items)) for k in range(n_items)) <= eps * n_items**2)
        model.addConstr(sum((alpha_0[i] - self.base_alpha_0[i])**2 for i in range(n_items)) <= eps * n_items)

        model.optimize() 

        alpha_0 = np.array([alpha_0[i].x for i in range(n_items)])
        alpha = np.array([[alpha[i,k].x for k in range(n_items)] for i in range(n_items)])

        return model.objVal, alpha, alpha_0

    def setup_outer(self): 
        n_items = self.n_items 

        model = gp.Model("upperbound")
        model.setParam('OutputFlag', 0)
        w = model.addVars(n_items, vtype=gp.GRB.CONTINUOUS, lb=0)
        ob = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-10000)

        model.addConstr(sum(w[i] for i in range(n_items)) <= self.capacity)
        model.setObjective(ob, gp.GRB.MINIMIZE)

        # add u = alpha, beta, gamma as new cut and re-solve
        def update_outer(u):
            alpha, alpha_0 = u
            v = model.addVars(n_items, vtype=gp.GRB.CONTINUOUS, lb=0)
            for i in range(n_items):
                # model.addConstr(v[i] >= 0) implicit
                model.addConstr(v[i] >= sum(alpha[i,k] * w[k] for k in range(n_items)) + alpha_0[i] - w[i])

            model.addConstr(ob >= sum(v[i] for i in range(n_items)) + self.unit_cost * sum(w[i] for i in range(n_items)))

            model.update()
            model.optimize() 
            # print("STATUS:", model.status)
            return model.objVal, np.array([w[i].x for i in range(self.n_items)])

        return update_outer

class RobustDemand: 
    def __init__(self, actions, demand, unit_cost, capacity, base_predictor, e2e = True): 
        self.actions = actions
        self.demand = demand 
        self.unit_cost = unit_cost
        self.capacity = capacity
        self.e2e = e2e

        self.base_alpha = base_predictor.alpha
        self.base_alpha_0 = base_predictor.alpha_0

        self.N = len(actions)
        self.n_items = len(actions[0])

    def predict(self, action): 
        return ProblemGenerator.predict_linear(action, self.base_alpha, self.base_alpha_0)

    def decision(self, eps=0): 
        return assortment_opt(self.base_alpha, self.base_alpha_0 + eps, self.unit_cost, self.capacity)


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

        epochs = 1000
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


class ApproximateGD:
    def __init__(self, actions, demand, unit_cost):
        self.actions = torch.tensor(actions)
        self.demand = torch.tensor(demand)
        self.unit_cost = unit_cost
        self.revs = torch.tensor(ProblemGenerator.objective(actions, demand, unit_cost))

    def subgradient(self, action, demand, unit_cost):
        ''' subgradient of the loss function '''
        return unit_cost + torch.where(action <= demand, torch.tensor(-1), torch.tensor(0))

    
    def compute_weights(self, a, actions):
        ''' kNN weights based on proximity of action,demand=a,d to other actions '''
        n = actions.shape[0]
        dists = np.zeros(n)
        for i in range(n):
            dists[i] = np.linalg.norm(a - actions[i])
        # inverse of distances
        # dists = 1 / (dists+1)
        # weights = dists / np.sum(dists)

        # knn weihts: 1 if k nearest, 0 otherwise
        k = 5
        weights = np.zeros(n)
        indices = np.argsort(dists)[:k]
        weights[indices] = 1 / k
        return weights
    
    def approximate_gradient(self, a, actions, demand, unit_cost):
        ''' Compute the approximate gradient as a weighted sum of subgradients '''
        n = actions.shape[0]
        # subgrads is a 2D array with shape (n, <number_of_features>)
        # subgrads = np.array([self.subgradient(a, d, unit_cost) for a, d in zip(actions, demand)])
        weights = self.compute_weights(a, actions)

        approx_grad = np.zeros(a.shape[0])  # Initialize with the correct shape
        for i in range(n):
            # Update approx_grad with the weighted sum
            approx_grad += weights[i] * np.array(self.subgradient(a, demand[i], unit_cost))

        return approx_grad


    def solve(self, actions, demand, unit_cost, lr=10**-2, epochs=200):
        ''' Run gradient descent on samples one by one using the approximate gradient to find optimal w '''
        # vector of random weights with specified seed
        w = np.random.rand(actions.shape[1])
        w = torch.tensor(w)
        for epoch in range(epochs):
            if epoch % 40 == 0:
                lr *= 0.3
            # for _ in range(len(actions)):
            # Compute the gradient for the current sample
            grad = self.approximate_gradient(w, actions, demand, unit_cost)
            for j in range(len(grad)):
                # Update the weight for the current sample
                w[j] = max(w[j] - lr * grad[j], 0)
        return w

    
    def decision(self):
        ''' Solve the assortment optimization problem using the approximate gradient '''
        w = self.solve(self.actions, self.demand, self.unit_cost)
        return np.array([w[i] for i in range(len(w))])
    


class ApproximateObjective:
    ''' Use the Objective approximation with KNN weights, solving a NMIP '''
    def __init__(self, actions, demand, unit_cost):
        self.actions = actions
        self.demand = demand
        self.unit_cost = unit_cost
        self.revs = torch.tensor(ProblemGenerator.objective(actions, demand, unit_cost))

    def setup_model(self, actions, demand, unit_cost, k=10):
        M = 1000
        model = gp.Model("approximate_objective")
        model.setParam('OutputFlag', 0)
        n = actions.shape[0]
        n_items = actions.shape[1]
        w = model.addVars(n_items, vtype=gp.GRB.CONTINUOUS, lb=0)
        x = model.addVars(n, vtype=gp.GRB.BINARY)
        y = model.addVars(n, n_items, vtype=gp.GRB.CONTINUOUS, lb=0)
        for i in range(n):
            for j in range(n_items):
                model.addConstr(y[i, j] >= - w[j] + demand[i, j])
        model.addConstr(sum(x[i] for i in range(n)) == k)
        for i in range(n):
            for j in range(n):
                if i != j:
                    model.addConstr(sum((w[k] - actions[i][k])**2 for k in range(n_items)) - 
                                    sum((w[k] - actions[j][k])**2 for k in range(n_items)) <= M * (x[j] - x[i] + 1))
        model.setObjective(sum(x[i] * sum(y[i, j] + unit_cost * w[j] for j in range(n_items)) for i in range(n)), gp.GRB.MINIMIZE)

        # run model
        model.optimize()
        return model, w
    
    def decision(self):
        ''' Solve the assortment optimization problem using the approximate objective '''
        model, w = self.setup_model(self.actions, self.demand, self.unit_cost)
        return np.array([w[i].x for i in range(len(w))])



                                    


# create new class by adapting the following method for the asortment optimization problem
# def kernel_action(actions_train, demands_train, actions_set, lambd=0.1):
#     n = len(actions_train)
#     K_hat = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             K_hat[i][j] = actions_train[i] * actions_train[j] + actions_train[i]**2 * actions_train[j]**2
#     c = np.zeros(n)
#     for i in range(n):
#         c[i] = actions_train[i] * demands_train[i]
#     values = []
#     for action in actions_set:
#         K_sample = np.zeros(n)
#         for i in range(n):
#             K_sample[i] = action * actions_train[i] + action**2 * actions_train[i]**2
#         h = K_sample.T @ np.linalg.inv(K_hat + n * np.eye(n) * lambd) @ c
#         values.append(h)
#     best_index = np.argmax(values)
#     return best_index, values[best_index]

class KernelPredictor: 
    def __init__(self, actions, demand, unit_cost, capacity, lambd=0.1): 
        self.actions = actions
        self.demand = demand 
        self.unit_cost = unit_cost
        self.capacity = capacity

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
        h = K_sample.T @ np.linalg.inv(self.K_hat + self.n * np.eye(self.n) * 0.1) @ self.c
        return h

    def decision(self): 
        h = [self.predict(action) for action in self.actions]
        best_index = np.argmin(h)
        return self.actions[best_index], h[best_index]
