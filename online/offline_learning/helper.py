import numpy as np
from sklearn.linear_model import LinearRegression
import gurobipy as gp
import torch
import nn_model

import matplotlib.pyplot as plt 

def get_reward(action, demand):
    return np.sum(action * demand)

def set_actions(n_actions): 
    return np.linspace(0, 1, n_actions)

def get_mean_demand(action): 
    return (1 - action)**2

def get_demand(action): 
    mean = (1 - action)**2
    variance = 0.1
    return np.random.lognormal(mean, variance, *action.shape)
    # return np.maximum(0, mean + mean * variance * np.random.randn(*action.shape))

def generate_data(action_set, n_samples):   
    actions = np.random.choice(action_set, n_samples)
    demands = get_demand(actions)
    return actions, demands

def load_data_online(agent):
    data = np.load('data/' + str(agent) + '.npy')
    actions = data[:, 0]
    demands = data[:, 1]
    return actions, demands

def predict_mean(x_train, y_train): 
    reg = LinearRegression().fit(np.expand_dims(x_train, axis=1), y_train)
    return reg

def two_stage_action(actions_train, demands_train, action_set): 
    reg = predict_mean(actions_train, demands_train)
    predictions = reg.predict(np.expand_dims(action_set, axis=1))

    rewards = action_set * predictions 
    return np.argmax(rewards)

def robust_action(actions_train, demands_train, action_set, robustness = 0.1): 
    optimal_values = []
    actions_list = action_set

    past_history = list(zip(actions_train, demands_train))
    time_steps = len(actions_train)

    ### first minimization problem to get constraint upperbound
    mod_ub = gp.Model("upperbound")
    mod_ub.setParam('OutputFlag', 0)
    alpha_0 = mod_ub.addVar(vtype=gp.GRB.CONTINUOUS, name="alpha_0", lb=-200)
    alpha_1 = mod_ub.addVar(vtype=gp.GRB.CONTINUOUS, name="alpha_1", lb=-200)

    # print(past_history)

    sum_w_squared = sum([a**2 for a, _ in past_history])
    sum_w = sum([a for a, _ in past_history])
    sum_d = sum([d for _, d in past_history])
    sum_d_sqared = sum([d**2 for _, d in past_history])
    sum_wd = sum([a*d for a, d in past_history])
    mod_ub.setObjective(alpha_0**2 + alpha_1**2 * sum_w_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_w/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_wd/time_steps, gp.GRB.MINIMIZE)
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
        alpha_1 = mod.addVar(vtype=gp.GRB.CONTINUOUS, name="alpha_1", lb=-200)
        mod.setObjective(w*(alpha_0+alpha_1*w), gp.GRB.MINIMIZE)
        
        sum_w_squared = sum([a**2 for a, _ in past_history])
        sum_w = sum([a for a, _ in past_history])
        sum_d = sum([d for _, d in past_history])
        sum_d_sqared = sum([d**2 for _, d in past_history])
        sum_wd = sum([a*d for a, d in past_history])
        # mod.addConstr(alpha_0**2 + alpha_1**2 * sum_w_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_w/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_wd/time_steps <= upperbound + upperbound*(1/np.sqrt(time_steps)), "c0")
        mod.addConstr(alpha_0**2 + alpha_1**2 * sum_w_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_w/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_wd/time_steps <= upperbound + upperbound*robustness, "c0")
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


def kernel_action(actions_train, demands_train, actions_set, lambd=0.1):
    n = len(actions_train)
    K_hat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_hat[i][j] = actions_train[i] * actions_train[j] + actions_train[i]**2 * actions_train[j]**2
    c = np.zeros(n)
    for i in range(n):
        c[i] = actions_train[i] * demands_train[i]
    values = []
    for action in actions_set:
        K_sample = np.zeros(n)
        for i in range(n):
            K_sample[i] = action * actions_train[i] + action**2 * actions_train[i]**2
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

