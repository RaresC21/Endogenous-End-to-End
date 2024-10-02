import numpy as np
import gurobipy as gp
import nn_model
import torch

class Policy(object):
    """
    A policy prescribes an action to be taken based on the memory of an agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent):
        return 0, 0
    

class NaivePolicy(Policy):

    def __init__(self, actions):
        self.actions = actions

    def __str__(self):
        return 'Naive Agent'
    
    def initialize(self, actions, demands):
        mod_naive = gp.Model("naive")
        mod_naive.setParam('OutputFlag', 0)
        alpha_0 = mod_naive.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_0")
        alpha_1 = mod_naive.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_1")
        # sum_a = sum(actions)
        # sum_d = sum(demands)
        # sum_a_squared = sum([a**2 for a in actions])
        # sum_d_squared = sum([d**2 for d in demands])
        # sum_ad = sum([a*d for a, d in zip(actions, demands)])
        #mod_naive.setObjective(alpha_0**2 + alpha_1**2 * sum_a_squared + sum_d_squared + 2*alpha_0*alpha_1*sum_a - 2*alpha_0*sum_d - 2*alpha_1*sum_ad, gp.GRB.MINIMIZE)
        mod_naive.setObjective(sum([(d - alpha_0 - alpha_1*a)**2 for d, a in zip(demands, actions)]), gp.GRB.MINIMIZE)
        mod_naive.optimize()
        return alpha_0.x, alpha_1.x

    def choose(self, alpha_0, alpha_1):
        # use the best values alpha_0 and alpha_1 to choose the best action
        optimal_values = []
        for w in self.actions:
            optimal_values.append(w*(alpha_0 + alpha_1*w))
        best_index = np.argmax(optimal_values)  # index corresponding to the best action
        best_w = self.actions[best_index]  # best action
        return best_index, best_w


class RobustLearningPolicy(Policy):
    """Robust Learning policy
    """
    def __init__(self, actions):
        self.actions = actions

    def __str__(self):
        return 'Robust Learning'
    
    def choose(self, agent):
        optimal_values = []
        # list of pairs (action, demand) for past time steps
        time_steps = len(agent.actions_list)
        print(time_steps)
        ### first minimization problem to get constraint upperbound
        mod_ub = gp.Model("upperbound")
        mod_ub.setParam('OutputFlag', 0)
        alpha_0 = mod_ub.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_0")
        alpha_1 = mod_ub.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_1")
        sum_a_squared = agent.sum_a_squared
        sum_a = agent.sum_a
        sum_d = agent.sum_d
        sum_d_sqared = agent.sum_d_squared
        sum_ad = agent.sum_ad
        mod_ub.setObjective(alpha_0**2 + alpha_1**2 * sum_a_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_a/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_ad/time_steps, gp.GRB.MINIMIZE)
        mod_ub.optimize()
        upperbound = mod_ub.objVal

        for w in self.actions:
            ### minimization (robust) problem to get optimal value of alpha
            mod = gp.Model("max_alpha")
            mod.setParam('OutputFlag', 0)
            alpha_0 = mod.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_0")
            alpha_1 = mod.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_1")
            mod.addConstr(alpha_0**2 + alpha_1**2 * sum_a_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_a/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_ad/time_steps <= upperbound + upperbound*(1/np.sqrt(time_steps)), "c0")
            mod.setObjective(w*(alpha_0+alpha_1*w), gp.GRB.MINIMIZE)
            mod.optimize()
            optimal_values.append(mod.objVal)
        
        best_index = np.argmax(optimal_values)  # index corresponding to the best action
        best_w = self.actions[best_index]  # best action
        return best_index, best_w



class RobustLearningPolicyNN(Policy):
    """Robust Learning policy
    """
    def __init__(self, actions):
        self.actions = actions

    def __str__(self):
        return 'Robust Learning NN'
    
    def initialize(self, actions, demands):
        '''Train initial neural network on the observed actions/demands'''
        net = nn_model.Net()
        x_train = torch.tensor(actions).float()
        x_train = x_train.view(-1, 1)
        y_train = torch.tensor(demands).float()
        trained_net = nn_model.train(net, x_train, y_train)
        return trained_net
    
    def choose(self, agent, trained_net):
        past_actions = [agent.action_values[k] for k in agent.actions_list]
        past_actions = torch.tensor(past_actions).view(-1, 1).float()
        first_layers_output = nn_model.get_output_first_layers(trained_net, past_actions)  # output for every past action
        parameters, bias = nn_model.get_last_hidden_layer_params(trained_net)
        n_params = parameters.shape[0]
        optimal_values = []
        # list of pairs (action, demand) for past time steps
        time_steps = len(agent.actions_list)
        ### first minimization problem to get constraint upperbound
        mod_ub = gp.Model("upperbound")
        mod_ub.setParam('OutputFlag', 0)
        # alpha_0 = mod_ub.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_0")
        # alpha_1 = mod_ub.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_1")
        # alpha variable (vector of size 5)
        alpha = mod_ub.addVars(n_params, lb=-gp.GRB.INFINITY, name="alpha")
        alpha_0 = mod_ub.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="bias")
        # initial value for alpha
        for i in range(n_params):
            alpha[i].start = parameters[0][i]
        alpha_0.start = bias[0]
        # sum_a_squared = agent.sum_a_squared
        # sum_a = agent.sum_a
        # sum_d = agent.sum_d
        # sum_d_sqared = agent.sum_d_squared
        # sum_ad = agent.sum_ad
        #mod_ub.setObjective(alpha_0**2 + alpha_1**2 * sum_a_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_a/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_ad/time_steps, gp.GRB.MINIMIZE)
        mod_ub.setObjective(1/time_steps * sum([(a*(d-sum([output[i]*alpha[i] for i in range(n_params)]) - alpha_0))**2 for (a, d, output) in zip(past_actions, agent.demands_list, first_layers_output)]), gp.GRB.MINIMIZE)
        mod_ub.optimize()
        upperbound = mod_ub.objVal

        for w in self.actions:
            w_tensor = torch.tensor([w]).float()
            features_w = nn_model.get_output_first_layers(trained_net, w_tensor)
            ### minimization (robust) problem to get optimal value of alpha
            mod = gp.Model("max_alpha")
            mod.setParam('OutputFlag', 0)
            # alpha_0 = mod.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_0")
            # alpha_1 = mod.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_1")
            alpha = mod.addVars(n_params, lb=-gp.GRB.INFINITY, name="alpha")
            alpha_0 = mod.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="bias")
            # initial value for alpha
            for i in range(n_params):
                alpha[i].start = parameters[0][i]
            alpha_0.start = bias[0]
            mod.addConstr(1/time_steps * sum([(a*(d-sum([output[i]*alpha[i] for i in range(n_params)]) - alpha_0))**2 for (a, d, output) in zip(past_actions, agent.demands_list, first_layers_output)]) <= upperbound + upperbound*(1/np.sqrt(time_steps)), "c0")
            mod.setObjective(w*(sum([features_w[i]*alpha[i] for i in range(n_params)]) + alpha_0), gp.GRB.MINIMIZE)
            mod.optimize()
            optimal_values.append(mod.objVal)
        
        best_index = np.argmax(optimal_values)  # index corresponding to the best action
        best_w = self.actions[best_index]  # best action
        return best_index, best_w