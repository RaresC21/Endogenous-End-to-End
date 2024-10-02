import numpy as np
from sklearn.linear_model import LinearRegression
import gurobipy as gp
from tqdm import tqdm

import matplotlib.pyplot as plt 

from helper import * 


if __name__ == '__main__':

    n_samples = 1000
    n_actions = 50

    action_set = set_actions(n_actions)

    actions_train, demands_train = generate_data(action_set, n_samples)
    # actions_train, demands_train = load_data_online('Adapted UCB')  
    # net = nn_model.Net()
    # trained_net = nn_model.train(net, torch.tensor(actions_train).float().view(-1, 1), torch.tensor(demands_train).float())


    best_action_index = np.argmax(action_set * get_mean_demand(action_set))
    best_action = action_set[best_action_index]

    baseline_action_index = two_stage_action(actions_train, demands_train, action_set)
    baseline_action = action_set[baseline_action_index]
    
    robust_actions = [] 
    robust_actions_nn = []
    robustness_levels = np.arange(0,1,0.025)
    for robustness in tqdm(robustness_levels):
        robust_action_index, _ = robust_action(actions_train, demands_train, action_set, robustness)
        robust_actions.append(action_set[robust_action_index]) 
        # robust_action_index_nn, _ = robust_action_nn(actions_train, demands_train, action_set, trained_net, robustness)
        # robust_actions_nn.append(action_set[robust_action_index_nn])
    
    plt.plot(robustness_levels, [action * get_mean_demand(action) for action in robust_actions], label = 'robust', color = 'blue')
    # plt.plot(robustness_levels, [action * get_mean_demand(action) for action in robust_actions_nn], label = 'robust NN', color = 'red')
    
    print("best     action:", best_action,     " best     reward:", best_action * get_mean_demand(best_action))
    print("baseline action:", baseline_action, " baseline reward:", baseline_action * get_mean_demand(baseline_action))

    for robust_action in robust_actions: 
        print("robust   action:", robust_action, " robust   reward:", robust_action * get_mean_demand(robust_action))

    plt.plot(robustness_levels, [baseline_action * get_mean_demand(baseline_action) for _ in robustness_levels], label = 'baseline', linestyle='--',color = 'orange')
    plt.plot(robustness_levels, [best_action * get_mean_demand(best_action) for _ in robustness_levels], label = 'optimal', linestyle='-.', color = 'green')
    plt.legend()
    plt.xlabel('epsilon (robustness level)')
    plt.ylabel('Reward')
    plt.title('Reward as a function of robustness')
    plt.show()
