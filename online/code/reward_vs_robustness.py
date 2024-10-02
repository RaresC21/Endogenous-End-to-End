import numpy as np
from sklearn.linear_model import LinearRegression
import gurobipy as gp
from tqdm import tqdm

import matplotlib.pyplot as plt 

from helper import * 

def evaluate_(action_set, test_feature, actions_train, demands_train, robustness_levels):
    # print("action set", action_set)
    # print(demand_generator.get_mean_demand(action_set, test_feature))
    # print()
    # best_action_index = np.argmax(action_set * demand_generator.get_mean_demand(action_set, test_feature))
    # best_action = action_set[best_action_index]

    baseline_action_index = two_stage_action(actions_train, demands_train, action_set, test_feature)
    baseline_action = action_set[baseline_action_index]
    
    kernel_action_index, _ = kernel_action(actions_train, demands_train, action_set, test_feature, 0.001)
    kernel_action_ = action_set[kernel_action_index]
    kernel_reward = kernel_action_ * demand_generator.get_mean_demand(kernel_action_, test_feature)

    robust_actions = [] 
    for robustness in robustness_levels:
        robust_action_index, _ = robust_action(actions_train, demands_train, action_set, test_feature, robustness)
        robust_actions.append(action_set[robust_action_index]) 

    best_reward = np.max(action_set * demand_generator.get_mean_demand(action_set, test_feature))
    baseline_reward = baseline_action * demand_generator.get_mean_demand(baseline_action, test_feature)
    robust_rewards = [r * demand_generator.get_mean_demand(r, test_feature) for r in robust_actions]
    return best_reward, baseline_reward, kernel_reward, robust_rewards

if __name__ == '__main__':

    n_samples = 200
    n_actions = 25
    n_features = 5
    n_test = 10

    robustness_levels = np.arange(0,1,0.1)

    action_set = set_actions(n_actions)

    demand_generator = Demand(n_features, 2)
    actions_train, demands_train = generate_data(action_set, n_samples, demand_generator)

    best_reward_all = [] 
    baseline_reward_all = []
    robust_rewards_all = []
    kernel_rewards_all = []

    for _ in tqdm(range(n_test)):
        test_feature = demand_generator.get_features(1)
        best_reward, baseline_reward, kernel_reward, robust_rewards = evaluate_(action_set, test_feature, actions_train, demands_train, robustness_levels)
        best_reward_all.append(best_reward)
        baseline_reward_all.append(baseline_reward)
        robust_rewards_all.append(robust_rewards)
        kernel_rewards_all.append(kernel_reward)
    
    robust_rewards_all = np.array(robust_rewards_all)
    best_ = np.mean(best_reward_all)
    baseline_ = np.mean(baseline_reward_all)
    kernel_ = np.mean(kernel_rewards_all)

    plt.plot(robustness_levels, [best_ for _ in range(len(robustness_levels))], label = 'optimal', linestyle='-.', color = 'green')
    plt.plot(robustness_levels, np.mean(robust_rewards_all, axis=0), label = 'robust', color = 'blue')
    plt.plot(robustness_levels, [kernel_ for _ in range(len(robustness_levels))], label = 'kernel', linestyle=':', color = 'purple')
    plt.plot(robustness_levels, [baseline_ for _ in range(len(robustness_levels))], label = 'baseline', linestyle='--',color = 'orange')

    plt.legend()
    plt.xlabel('epsilon (robustness level)')
    plt.ylabel('Reward')
    plt.title('Reward as a function of robustness')
    plt.show()
