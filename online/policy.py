import numpy as np
import gurobipy as gp
import torch
import nn_model

class Policy(object):
    """
    A policy prescribes an action to be taken based on the memory of an agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent):
        return 0, 0
    

class EpsilonGreedyPolicy(Policy):
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """
    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            rand = np.random.choice(len(agent.value_estimates))
            return rand, self.actions[rand]
        else:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
            if len(check) == 1:
                return action, self.actions[action]
            else:
                return np.random.choice(check), self.actions[action]


class GreedyPolicy(EpsilonGreedyPolicy):
    """
    The Greedy policy only takes the best apparent action, with ties broken by
    random selection. This can be seen as a special case of EpsilonGreedy where
    epsilon = 0 i.e. always exploit.
    """
    def __init__(self, actions):
        super(GreedyPolicy, self).__init__(0, actions)

    def __str__(self):
        return 'greedy'


class RandomPolicy(EpsilonGreedyPolicy):
    """
    The Random policy randomly selects from all available actions with no
    consideration to which is apparently best. This can be seen as a special
    case of EpsilonGreedy where epsilon = 1 i.e. always explore.
    """
    def __init__(self, actions):
        super(RandomPolicy, self).__init__(1, actions)

    def __str__(self):
        return 'random'


class UCBPolicy(Policy):
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """
    def __init__(self, actions, eps = 1e-6):
        self.eps = eps
        self.actions = actions

    def __str__(self):
        return 'UCB'

    def choose(self, agent):
        exploration = np.log10(agent.t+1) / (agent.action_attempts) # pull everything once
        exploration[np.isnan(exploration)] = 0
        exploration = np.sqrt(exploration)
        q = agent.value_estimates + exploration  # different "value_estimates" from AdaptedUCB /!\ Here, it is an estimate of g
        action = np.argmax(q)
        check = np.where(q == q[action])[0]
        if len(check) == 1:
            return action, self.actions[action]
        else:
            return np.random.choice(check), self.actions[action]


class AdaptedUCBPolicy(Policy):
    """
    Adapter Upper Confidence Bound (AdaptedUCB) policy. It optimistically chooses
    the price with the highest objective, using the upper confidence bound as
    the deterministic realization of D(w) with an increasing convex function g(w, UCB).
    """
    def __init__(self,actions, g_function, eps = 1e-6):
        self.g_function = g_function
        self.actions = actions
        self.eps = eps

    def __str__(self):
        return 'Adapted UCB'

    def choose(self, agent):
        exploration = np.log10(agent.t + 1) / (agent.action_attempts)
        exploration[np.isnan(exploration)] = 0
        exploration = np.sqrt(exploration)
        q = self.g_function(self.actions, agent.value_estimates + exploration) # different "value_estimates" from UCB /!\ Here, it is an estimate of the demand
        action = np.argmax(q)
        check = np.where(q == q[action])[0]
        if len(check) == 1:
            return action, self.actions[action]
        else:
            return np.random.choice(check), self.actions[action]
        


# class LinUCBPolicy(Policy):
#     """
#     Linear UCB policy
#     """

#     def __init__(self, actions, k, mu, sigma, alpha=1):
#         self.actions = actions
#         # features contain 1, price and price ** 2 for evert price (action)
#         self.features = np.zeros((len(actions), 3))
#         for i, price in enumerate(actions):
#             self.features[i] = np.array([1, price, price ** 2])
#         self.A = np.identity(3)
#         self.b = np.zeros(3)
#         self.alpha = alpha
#         self.k = k
#         self.mu = mu
#         self.sigma = sigma

#     def update(self):
#         self.A_inv = np.linalg.inv(self.A)
#         self.theta = np.dot(self.A_inv, self.b)
#         features = self.features
#         p = np.zeros(len(self.actions))
#         for i in range(len(self.actions)):
#             p[i] = np.dot(self.theta, features[i]) + self.alpha * np.sqrt(np.dot(np.dot(features[i], self.A_inv), features[i]))
#         return p
    
#     def choose(self, p):
#         return np.argmax(p), self.actions[np.argmax(p)]
    
#     def reset(self):
#         self.action_values = np.random.normal(self.mu, self.sigma, self.k)

#     def observe(self, action):
#         print(action)
#         print(self.action_values[action])
#         print(self.actions[action])
#         self.optimal = np.argmax([self.mu[i] * self.actions[i] for i in range(self.k)])
#         reward = self.action_values[action] * self.actions[action] # g function, product of demand and price
#         self.A += np.outer(self.features[action], self.features[action])
#         self.b += reward * self.features[action]
#         return reward, action == self.optimal
    
#     def __str__(self):
#         return 'LinUCB'


# all the UCB policy process in a single function, with updates at each step
class LinUCBPolicy(Policy):
    """
    Linear UCB policy
    """
    def __init__(self, actions, k, mu, sigma, version='1', alpha=1):
        self.actions = actions
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.version = version

    def __str__(self):
        if self.version=='1':
            return 'LinUCB1'
        elif self.version=='2':
            return 'LinUCB2'

    def reset(self, action_values):
        self.action_values = action_values

    def run(self, time_steps):
        list_picked_actions = []
        list_observed_demands = []
        if self.version=='1':
            # features contain 1, price and price ** 2 for evert price (action)
            d = 3
            features = np.zeros((len(self.actions), d))
            for i, price in enumerate(self.actions):
                features[i] = np.array([1, price, price**2])
            A = np.identity(d)
            b = np.zeros(d)
            theta = np.zeros(d)
            optimal = np.argmax([self.mu[i] * self.actions[i] for i in range(self.k)])
            rewards = np.zeros(time_steps)
            optimal_actions = np.zeros(time_steps)
            for t in range(time_steps):
                self.reset(np.random.normal(self.mu, self.sigma, self.k))
                A_inv = np.linalg.inv(A)
                theta = np.dot(A_inv, b)
                p = np.zeros(len(self.actions))
                for i in range(len(self.actions)):
                    p[i] = np.dot(theta, features[i]) + self.alpha * np.sqrt(np.dot(np.dot(features[i], A_inv), features[i]))
                max_score = np.argmax(p)
                check = np.where(p == p[max_score])[0]
                action = np.random.choice(check) if len(check) > 1 else max_score
                reward = self.action_values[action] * self.actions[action] # g function, product of demand and price
                A += np.outer(features[action], features[action])
                b += reward * features[action]
                rewards[t] = reward
                if action == optimal:
                    optimal_actions[t] = 1
                list_picked_actions.append(self.actions[action])
                list_observed_demands.append(self.action_values[action])
            return rewards, optimal_actions, list_picked_actions, list_observed_demands
        elif self.version=='2':
            # Almost the same as version 1, but this time, different matrices and features for each action
            d = 3
            features = np.zeros((len(self.actions), d))
            for i, price in enumerate(self.actions):
                features[i] = np.array([1, price, price**2])
            A = np.asarray([np.identity(d) for i in range(self.k)])
            b = np.zeros((self.k, d))
            theta = np.zeros((self.k, d))
            optimal = np.argmax([self.mu[i] * self.actions[i] for i in range(self.k)])
            rewards = np.zeros(time_steps)
            optimal_actions = np.zeros(time_steps)
            for t in range(time_steps):
                self.reset(np.random.normal(self.mu, self.sigma, self.k))
                A_inv = np.asarray([np.linalg.inv(A[i]) for i in range(self.k)])
                theta = np.asarray([np.dot(A_inv[i], b[i]) for i in range(self.k)])
                p = np.zeros(len(self.actions))
                for i in range(len(self.actions)):
                    p[i] = np.dot(theta[i], features[i]) + self.alpha * np.sqrt(np.dot(np.dot(features[i], A_inv[i]), features[i]))
                max_score = np.argmax(p)
                check = np.where(p == p[max_score])[0]
                action = np.random.choice(check) if len(check) > 1 else max_score
                reward = self.action_values[action] * self.actions[action]
                A[action] += np.outer(features[action], features[action])
                b[action] += reward * features[action]
                rewards[t] = reward
                if action == optimal:
                    optimal_actions[t] = 1
                list_picked_actions.append(self.actions[action])
                list_observed_demands.append(self.action_values[action])
            return rewards, optimal_actions, list_picked_actions, list_observed_demands
        


# Structure for optimistic learning: 
#  - choose according to the max problem over alpha, and then discrete max problem over the actions w (prices)
#  - for the price chose, observe the demand (like in AdaptedUCB) (and not the reward)


class OptimisticLearningPolicy(Policy):
    """Optimistic Learning policy
    """
    def __init__(self, actions):
        self.actions = actions

    def __str__(self):
        return 'Optimistic Learning'
    
    def choose(self, agent):
        optimal_values = []
        # list of pairs (action, demand) for past time steps
        time_steps = len(agent.actions_list)
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
            ### maximization problem to get optimal value of alpha
            mod = gp.Model("max_alpha")
            mod.setParam('OutputFlag', 0)
            alpha_0 = mod.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_0")
            alpha_1 = mod.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="alpha_1")
            mod.addConstr(alpha_0**2 + alpha_1**2 * sum_a_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_a/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_ad/time_steps <= upperbound + upperbound*(1/np.sqrt(time_steps)), "c0")
            mod.setObjective(w*(alpha_0+alpha_1*w), gp.GRB.MAXIMIZE)
            mod.optimize()
            optimal_values.append(mod.objVal)
        
        best_index = np.argmax(optimal_values)  # index corresponding to the best action
        best_w = self.actions[best_index]  # best action
        return best_index, best_w



class OptimisticLearningPolicyNN(Policy):
    """Optimistic Learning Policy with NN
    """
    def __init__(self, actions):
        self.actions = actions
        self.trained_net = None

    def __str__(self):
        return 'Optimistic Learning NN'
    
    def choose(self, agent, robustness=0.5):
        actions_train = agent.actions_list
        demands_train = agent.demands_list
        past_actions = torch.tensor(actions_train).view(-1, 1).float()
        # scale actions
        past_actions = (past_actions - past_actions.mean()) / past_actions.std()
        if (len(actions_train) == len(self.actions)) or (len(actions_train) % (len(self.actions)*3)) == 0:
            trained_net = nn_model.train(nn_model.Net(), past_actions, torch.tensor(demands_train).float())
            self.trained_net = trained_net
        else:
            trained_net = self.trained_net
        first_layers_output = nn_model.get_output_first_layers(trained_net, past_actions)  # output for every past action
        # back to original scale
        first_layers_output = first_layers_output * np.std(demands_train) + np.mean(demands_train)
        parameters, bias = nn_model.get_last_hidden_layer_params(trained_net)
        n_params = parameters.shape[1]
        optimal_values = []
        # list of pairs (action, demand) for past time steps
        time_steps = len(actions_train)
        ### first minimization problem to get constraint upperbound
        mod_ub = gp.Model("upperbound")
        mod_ub.setParam('OutputFlag', 0)
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
        #mod_ub.setObjective(alpha_0**2 + alpha_1**2 * sum_a_squared/time_steps + sum_d_sqared/time_steps + 2*alpha_0*alpha_1*sum_a/time_steps - 2*alpha_0*sum_d/time_steps - 2*alpha_1*sum_ad/time_steps, gp.GRB.MINIMIZE)
        mod_ub.setObjective(1/time_steps * sum([(a*(d-sum([output[i]*alpha[i] for i in range(n_params)]) - alpha_0))**2 for (a, d, output) in zip(actions_train, demands_train, first_layers_output)]), gp.GRB.MINIMIZE)
        mod_ub.optimize()
        upperbound = mod_ub.objVal
        # print optimal alpha values
        # output_test = sum([first_layers_output[0][i]*alpha[i].x for i in range(n_params)]) + alpha_0.x
        # manually compute objective function
        #print('obj: ', 1/time_steps * sum([(a*(d-sum([output[i]*alpha[i].x for i in range(n_params)]) - alpha_0.x))**2 for (a, d, output) in zip(actions_train, demands_train, first_layers_output)]))

        for w in self.actions:
            w_tensor = torch.tensor([w]).float()
            # normalize w
            w_tensor = (w_tensor - w_tensor.mean()) / w_tensor.std()
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
            mod.addConstr(1/time_steps * sum([(a*(d-sum([output[i]*alpha[i] for i in range(n_params)]) - alpha_0))**2 for (a, d, output) in zip(actions_train, demands_train, first_layers_output)]) <= abs(upperbound) + 0.1, "c0")
            mod.setObjective(w*(sum([features_w[i]*alpha[i] for i in range(n_params)]) + alpha_0), gp.GRB.MAXIMIZE)
            mod.optimize()
            if mod.status == gp.GRB.Status.INFEASIBLE:
                print('FEATURES W', features_w)
            # get optimal alpha
            optimal_values.append(mod.objVal)
        print('optimal values: ', optimal_values)
        best_index = np.argmax(optimal_values)  # index corresponding to the best action
        print('best index: ', best_index)
        return best_index, optimal_values[best_index]