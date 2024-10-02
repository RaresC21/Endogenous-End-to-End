import numpy as np 
import gurobipy as gp

def assortment_opt(alpha, beta, unit_cost=0.3, capacity=5): 
    n_items = len(alpha) 

    model = gp.Model("upperbound")
    model.setParam('OutputFlag', 0)
    w = model.addVars(n_items, vtype=gp.GRB.CONTINUOUS, lb=0)
    v = model.addVars(n_items, vtype=gp.GRB.CONTINUOUS)


    for i in range(n_items):
        model.addConstr(v[i] >= sum((0 if i == k else alpha[i,k]) * w[k] for k in range(n_items)) + beta[i] - w[i])
        model.addConstr(v[i] >= 0)

    model.addConstr(sum(w[i] for i in range(n_items)) <= capacity)
    model.setObjective(sum(v[i] for i in range(n_items)) + unit_cost * sum(w[i] for i in range(n_items)), gp.GRB.MINIMIZE)

    model.optimize() 
    return np.array([w[i].x for i in range(n_items)])