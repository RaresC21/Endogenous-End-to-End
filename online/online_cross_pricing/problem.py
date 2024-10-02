import numpy as np 
import gurobipy as gp

def pricing_opt(alpha, beta, gamma, prices): 
    n_items = len(alpha) 
    n_prices = len(prices)
    for i in range(n_items): 
        gamma[i,i] = 0

    model = gp.Model("upperbound")
    model.setParam('OutputFlag', 0)
    x = model.addVars(n_items, n_prices, vtype=gp.GRB.BINARY)
    y = model.addVars(n_items, n_items, n_prices, n_prices, vtype=gp.GRB.BINARY)

    for i in range(n_items):
        model.addConstr(sum(x[i,t] for t in range(n_prices)) == 1)
    
    for i in range(n_items): 
        for j in range(n_items): 
            for t in range(n_prices): 
                model.addConstr(sum(y[i,j,t,tt] for tt in range(n_prices)) == x[i,t])
                model.addConstr(sum(y[i,j,tt,t] for tt in range(n_prices)) == x[j,t])

    model.setObjective(sum( alpha[i] * sum(prices[t] * x[i,t] for t in range(n_prices)) + beta[i] * sum(prices[t]*prices[t] * x[i,t] for t in range(n_prices)) for i in range(n_items))
                        + sum(sum(sum(sum(gamma[i,j] * prices[t1] * prices[t2] * y[i,j,t1,t2] for t2 in range(n_prices)) for t1 in range(n_prices)) for j in range(n_items)) for i in range(n_items)), gp.GRB.MAXIMIZE)
    model.optimize() 
    x_sol = np.array([[x[i,j].x for j in range(n_prices)] for i in range(n_items)])
    return np.array([prices[np.argmax(x_sol[i,:])] for i in range(n_items)])

def random_decision(n_items): 
    return np.random.rand(n_items)