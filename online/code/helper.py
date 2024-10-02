import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import gurobipy as gp
import torch
import nn_model

import matplotlib.pyplot as plt 

def get_reward(action, demand):
    return np.sum(action * demand)

def set_actions(n_actions): 
    return np.linspace(0, 1, n_actions)

class Demand(): 
    def __init__(self, num_features = 0, power=2): 
        self.num_features = num_features 
        self.A = np.random.randn(self.num_features)
        self.power = power
    def get_demand(self, action, features): 
        mean_demand = self.get_mean_demand(action, features) 

        # variance = mean_demand
        variance = 0.5 * action
        # print(mean_demand)
        return np.random.normal(mean_demand, variance, *mean_demand.shape)
        # return mean_demand + variance * np.random.randn(*action.shape)

    def get_mean_demand(self, action, features): 
        nominal_price = (features @ self.A) ** 2 
        action_change = (1 - action) ** self.power

        # print(action_change[0], nominal_price[0])
        # print("nominal_price:", nominal_price)
        # print("average nominal price:", np.mean(nominal_price), np.mean(action_change)) 
        # print(nominal_price)
        # print(nominal_price * 1 * np.mean(action_change) / np.mean(nominal_price))
        return action_change + nominal_price * 1 * np.mean(action_change) / np.mean(nominal_price)

    def get_features(self, n): 
        return np.random.rand(n, self.num_features)

# def get_mean_demand(action): 
#     return (1 - action) ** 3

# def get_demand(action, num_features): 
#     mean = (1 - action) ** 3
#     variance = 0.5
#     return np.random.normal(mean, variance, *action.shape)
#     # return mean + variance * np.random.randn(*action.shape)
#     # return np.maximum(0, mean + mean * variance * np.random.randn(*action.shape))

def generate_data(action_set, n_samples, generator = Demand()):   
    n_ac = len(action_set)
    # print(action_set)
    # actions = np.append(np.random.choice(action_set[:3], n_samples//10*9), np.random.choice(action_set, n_samples//10))
    actions = np.random.choice(action_set, n_samples)
    features = generator.get_features(n_samples)
    demands = generator.get_demand(actions, features)
    if generator.num_features == 0:
        return np.expand_dims(actions, axis=1), demands
    
    actions = np.expand_dims(actions, axis=1)
    return np.concatenate((actions, features), axis=1), demands 

def load_data_online(agent):
    data = np.load('data/' + str(agent) + '.npy')
    actions = data[:, 0]
    demands = data[:, 1]
    return actions, demands

def predict_mean(x_train, y_train): 
    reg = LinearRegression().fit(x_train, y_train)
    return reg

def two_stage_action(actions_train, demands_train, action_set, feature): 
    reg = predict_mean(actions_train, demands_train)
    err = mean_squared_error(reg.predict(actions_train), demands_train)
    # print("LS err", err)
    features = np.tile(feature, (len(action_set), 1))
    inp = np.concatenate((np.expand_dims(action_set,axis=1), features), axis=1)
    predictions = reg.predict(inp)

    rewards = action_set * predictions 
    return np.argmax(rewards)

def robust_action(actions_train, demands_train, action_set, feature, robustness = 0.1): 
    optimal_values = []
    actions_list = action_set

    past_history = list(zip(actions_train, demands_train))
    time_steps = actions_train.shape[0]
    dim = actions_train.shape[1]

    ### first minimization problem to get constraint upperbound
    mod_ub = gp.Model("upperbound")
    mod_ub.setParam('OutputFlag', 0)
    alpha_0 = mod_ub.addVar(vtype=gp.GRB.CONTINUOUS, name="alpha_0", lb=-200)
    alpha_1 = mod_ub.addVars(dim, vtype=gp.GRB.CONTINUOUS, name="alpha_1", lb=-200)

    # print(sum(alpha_1 * actions_train[0]))
    mod_ub.setObjective(sum( ((alpha_0 + sum(alpha_1[k] * actions_train[i][k] for k in range(dim))) - demands_train[i])**2 for i in range(time_steps)) / time_steps)

    # print(past_history)

    # sum_w_squared = sum([a**2 for a, _ in past_history])
    # sum_w = sum([a for a, _ in past_history])
    # sum_d = sum([d for _, d in past_history])
    # sum_d_sqared = sum([d**2 for _, d in past_history])
    # sum_wd = sum([a*d for a, d in past_history])
    # mod_ub.setObjective(alpha_0**2 + alpha_1**2 * sum_w_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_w/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_wd/time_steps, gp.GRB.MINIMIZE)
    mod_ub.optimize()
    # print("optimal alpha: ", alpha_0.x, alpha_1.x)
    upperbound = mod_ub.objVal
    # print("upperbound:", upperbound)

    # print('upperbound:', upperbound)

    for w in action_set:
        ### maximization problem to get optimal value of alpha
        mod = gp.Model("max_alpha")
        mod.setParam('OutputFlag', 0)
        alpha_0 = mod.addVar(vtype=gp.GRB.CONTINUOUS, name="alpha_0", lb=-200)
        alpha_1 = mod.addVars(dim, vtype=gp.GRB.CONTINUOUS, name="alpha_1", lb=-200)

        f = np.concatenate(([[w]], feature), axis=1)[0]
        mod.setObjective(w*(alpha_0 + sum(alpha_1[k] * f[k] for k in range(dim))), gp.GRB.MINIMIZE)
        
        # sum_w_squared = sum([a**2 for a, _ in past_history])
        # sum_w = sum([a for a, _ in past_history])
        # sum_d = sum([d for _, d in past_history])
        # sum_d_sqared = sum([d**2 for _, d in past_history])
        # sum_wd = sum([a*d for a, d in past_history])
        # mod.addConstr(alpha_0**2 + alpha_1**2 * sum_w_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_w/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_wd/time_steps <= upperbound + upperbound*(1/np.sqrt(time_steps)), "c0")
        # mod.addConstr(alpha_0**2 + alpha_1**2 * sum_w_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_w/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_wd/time_steps <= upperbound + upperbound*robustness, "c0")
        mod.addConstr(sum( ((alpha_0 + sum(alpha_1[k] * actions_train[i][k] for k in range(dim))) - demands_train[i])**2 for i in range(time_steps)) / time_steps <= upperbound + upperbound * robustness, "c0")
        mod.optimize()
        optimal_values.append(mod.objVal)
    #     print(mod.objVal)
    # print()
    # print("ACTIONS":, self.actions)
    # print(optimal_values)
    best_index = np.argmax(optimal_values)  # corresponds to the best action
    best_w = action_set[best_index]  # best action
    # print("ACTION:", best_w)
    return best_index, optimal_values[best_index]


def robust_action_nn(actions_train, demands_train, action_set, trained_net, robustness = 0.1):
    past_actions = torch.tensor(actions_train).view(-1, 1).float()
    first_layers_output = nn_model.get_output_first_layers(trained_net, past_actions)  # output for every past action
    parameters, bias = nn_model.get_last_hidden_layer_params(trained_net)
    n_params = parameters.shape[1]
    optimal_values = []
    # list of pairs (action, demand) for past time steps
    time_steps = len(actions_train)
    ### first minimization problem to get constraint upperbound
    mod_ub = gp.Model("upperbound")
    mod_ub.setParam('OutputFlag', 0)
    # alpha_0 = mod_ub.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_0")
    # alpha_1 = mod_ub.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_1")
    # alpha variable (vector of size 5)
    alpha = mod_ub.addVars(n_params, lb=-200, name="alpha")
    alpha_0 = mod_ub.addVar(vtype=gp.GRB.CONTINUOUS, lb=-200, name="bias")
    # initial value for alpha
    for i in range(n_params):
        alpha[i].start = parameters[0][i].item()
    alpha_0.start = bias[0].item()
    # add a constraint to say that each alpha must not go too far from the initial value
    for i in range(n_params):
        mod_ub.addConstr(alpha[i]-parameters[0][i].item() <= abs(parameters[0][i].item())/2)
        mod_ub.addConstr(parameters[0][i].item()-alpha[i] <= abs(parameters[0][i].item())/2)
    mod_ub.addConstr(alpha_0-bias[0].item() <= abs(bias[0].item())/2)
    mod_ub.addConstr(bias[0].item()-alpha_0 <= abs(bias[0].item())/2)
    print("BIAS LAST LAYER (before optimization): ", bias[0].item())
    print("PARAMETERS LAST LAYER (before optimization): ", [parameters[0][i].item() for i in range(n_params)])
    #mod_ub.setObjective(alpha_0**2 + alpha_1**2 * sum_a_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_a/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_ad/time_steps, gp.GRB.MINIMIZE)
    mod_ub.setObjective(1/time_steps * sum([(a*(d-sum([output[i]*alpha[i] for i in range(n_params)]) - alpha_0))**2 for (a, d, output) in zip(actions_train, demands_train, first_layers_output)]), gp.GRB.MINIMIZE)
    mod_ub.optimize()
    upperbound = mod_ub.objVal
    # print optimal alpha values
    print("OPTIMAL PARAMETERS LAST LAYER (ALPHA): ")
    print(" - BIAS: ", alpha_0.x)
    print(" - PARAMETERS:", [alpha[i].x for i in range(n_params)])
    output_test = sum([first_layers_output[0][i]*alpha[i].x for i in range(n_params)]) + alpha_0.x
    print('output test: ', output_test)
    # manually compute objective function
    #print('obj: ', 1/time_steps * sum([(a*(d-sum([output[i]*alpha[i].x for i in range(n_params)]) - alpha_0.x))**2 for (a, d, output) in zip(actions_train, demands_train, first_layers_output)]))
    print('upperbound NN:', upperbound)

    for w in action_set:
        print(f"NN prediction for action {w}: ", trained_net(torch.tensor([w]).float().view(-1, 1)).squeeze().item())
        w_tensor = torch.tensor([w]).float()
        features_w = nn_model.get_output_first_layers(trained_net, w_tensor)
        ### minimization (robust) problem to get optimal value of alpha
        mod = gp.Model("max_alpha")
        mod.setParam('OutputFlag', 0)
        alpha = mod.addVars(n_params, lb=-200, name="alpha")
        alpha_0 = mod.addVar(vtype=gp.GRB.CONTINUOUS, lb=-200, name="bias")
        # initial value for alpha
        for i in range(n_params):
            alpha[i].start = parameters[0][i].item()
        alpha_0.start = bias[0].item()
        for i in range(n_params):
            mod.addConstr(alpha[i]-parameters[0][i].item() <= abs(parameters[0][i].item())/2)
            mod.addConstr(parameters[0][i].item()-alpha[i] <= abs(parameters[0][i].item())/2)
        mod.addConstr(alpha_0-bias[0].item() <= abs(bias[0].item())/2)
        mod.addConstr(bias[0].item()-alpha_0 <= abs(bias[0].item())/2)
        mod.addConstr(1/time_steps * sum([(a*(d-sum([output[i]*alpha[i] for i in range(n_params)]) - alpha_0))**2 for (a, d, output) in zip(actions_train, demands_train, first_layers_output)]) <= upperbound + upperbound*robustness, "c0")
        mod.setObjective(w*(sum([features_w[i]*alpha[i] for i in range(n_params)]) + alpha_0), gp.GRB.MINIMIZE)
        mod.optimize()
        if mod.status == gp.GRB.Status.INFEASIBLE:
            print('FEATURES W', features_w)
        # get optimal alpha
        optimal_values.append(mod.objVal)
        # print(alpha_0.x, alpha[0].x, alpha[1].x, alpha[2].x, alpha[3].x, alpha[4].x)
        # print(mod.objVal)
        # print('------------------')

    # # best prediction from the neural net
    # preds = trained_net(torch.tensor(action_set).float().view(-1, 1)).squeeze()
    # values = [action_set[i] * preds[i].detach().numpy() for i in range(len(action_set))]
    # best_index = np.argmax(values)
    # best_w = action_set[best_index]
    # return best_index, values[best_index].item()
    
    best_index = np.argmax(optimal_values)  # index corresponding to the best action
    return best_index, optimal_values[best_index]


def kernel_action(actions_train, demands_train, actions_set, feature, lambd=0.1):
    n = len(actions_train)
    K_hat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_hat[i][j] = (actions_train[i] @ actions_train[j] + 1) ** 3  #actions_train[i] * actions_train[j] + actions_train[i]**2 * actions_train[j]**2
    c = np.zeros(n)
    for i in range(n):
        c[i] = actions_train[i][0] * demands_train[i]
    values = []
    for action in actions_set:
        K_sample = np.zeros(n)
        for i in range(n):
            cur = np.concatenate(([[action]], feature), axis=1)
            K_sample[i] = (cur @ actions_train[i] + 1) ** 3  #action * actions_train[i] + action**2 * actions_train[i]**2
        h = K_sample.T @ np.linalg.inv(K_hat + n * np.eye(n) * lambd) @ c
        values.append(h)
    best_index = np.argmax(values)
    return best_index, values[best_index]

def kernel_action_nn(actions_train, demands_train, actions_set, trained_net, lambd=0.1):
    ''' same, but this time, features come from the neural net instead of price and price squared '''
    n = len(actions_train)
    K_hat = np.zeros((n, n))
    features = nn_model.get_output_first_layers(trained_net, torch.tensor(actions_train).float().view(-1, 1))
    features = np.array([features[i].detach().numpy() for i in range(n)])
    for i in range(n):
        for j in range(n):
            K_hat[i][j] = features[i] @ features[j]
    c = np.zeros(n)
    for i in range(n):
        c[i] = actions_train[i] * demands_train[i]
    values = []
    for action in actions_set:
        features_sample = nn_model.get_output_first_layers(trained_net, torch.tensor([action]).float().view(-1, 1))
        features_sample = features_sample.detach().numpy()
        # inner product between features_sample and each feature in features
        K_sample = np.zeros(n)
        for i in range(n):
            K_sample[i] = features_sample @ features[i]
        h = K_sample.T @ np.linalg.inv(K_hat + n * np.eye(n) * lambd) @ c
        values.append(h)
    best_index = np.argmax(values)
    return best_index, values[best_index]

