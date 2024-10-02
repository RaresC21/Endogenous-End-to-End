import numpy as np
from sklearn.linear_model import LinearRegression
import gurobipy as gp
import nn_model
import matplotlib.pyplot as plt 
from tqdm import tqdm

from helper import * 


if __name__ == '__main__':

    n_trials = 10

    sample_counts = [10 * i for i in range(1,10)] + [50 * i for i in range(2, 20)] #+ [50 * i for i in range(3, 5, 1)]  
    n_actions = 25
    action_set = set_actions(n_actions)

    robust_actions = []

    robust_frac = []
    baseline_frac = []
    kernel_frac = []

    robust_rewards_all = []
    robust_rewards_nn_all = []
    baseline_rewards_all = []
    best_rewards_all = []
    kernel_rewards_all = []

    robust_vars = [] 
    robust_vars_nn = []
    baseline_vars = [] 
    best_vars = []
    kernel_vars = []

    # actions_train, demands_train = generate_data(action_set, 1000)
    # net = nn_model.Net()
    # trained_net = nn_model.train(net, torch.tensor(actions_train).float().view(-1, 1), torch.tensor(demands_train).float())

    n_features = 5

    demand_generator = Demand(n_features)

    for n_samples in sample_counts:
        robust_rewards = []
        robust_rewards_nn = []
        kernel_rewards = []
        baseline_rewards = []
        best_rewards = []

        actions_train, demands_train = generate_data(action_set, n_samples, demand_generator)

        for _ in tqdm(range(n_trials)): 
            test_feature = demand_generator.get_features(1)
            # print("TEST FEATURE :", test_feature)
            # actions_train, demands_train = load_data_online('Adapted UCB')
            # actions_train, demands_train = actions_train[:n_samples], demands_train[:n_samples]

            # train nn model
            # net = nn_model.Net()
            # trained_net = nn_model.train(net, torch.tensor(actions_train).float().view(-1, 1), torch.tensor(demands_train).float())

            best_rewards.append(np.max(action_set * demand_generator.get_mean_demand(action_set, test_feature)))

            baseline_action_index = two_stage_action(actions_train, demands_train, action_set, test_feature)
            baseline_action = action_set[baseline_action_index]
            baseline_rewards.append(baseline_action * demand_generator.get_mean_demand(baseline_action, test_feature))

            robust_action_index, _ = robust_action(actions_train, demands_train, action_set, test_feature, .1)
            robust_action_ = action_set[robust_action_index]
            robust_rewards.append(robust_action_ * demand_generator.get_mean_demand(robust_action_, test_feature))

            # kernel_action_index, _ = kernel_action_nn(actions_train, demands_train, action_set, trained_net, test_feature, 0.001)
            kernel_action_index, _ = kernel_action(actions_train, demands_train, action_set, test_feature, 0.001)
            kernel_action_ = action_set[kernel_action_index]
            kernel_rewards.append(kernel_action_ * demand_generator.get_mean_demand(kernel_action_, test_feature))

            # robust_action_index_nn, _ = robust_action_nn(actions_train, demands_train, action_set, trained_net, 1)
            # robust_action_nn_ = action_set[robust_action_index_nn]
            # robust_rewards_nn.append(robust_action_nn_ * get_mean_demand(robust_action_nn_))

            # print("robust action: ", robust_action_)
            # print("robust action nn: ", robust_action_nn_)

        # print("robust rewards:", robust_rewards)

        # print("wins", [r1 > r2 and r1 > r3 for r1, r2, r3 in zip(robust_rewards, kernel_rewards, baseline_rewards)])
        # robust_frac.append(np.sum([r1 > r2 and r1 > r3 for r1, r2, r3 in zip(robust_rewards, kernel_rewards, baseline_rewards)]) * 1.0 / n_trials )
        # baseline_frac.append(np.sum([r1 > r2 and r1 > r3 for r1, r2, r3 in zip(baseline_rewards, kernel_rewards, robust_rewards)]) * 1.0 / n_trials)
        # kernel_frac.append(np.sum([r1 > r2 and r1 > r3 for r1, r2, r3 in zip(kernel_rewards, robust_rewards, baseline_rewards)]) * 1.0 / n_trials)

        robust_rewards_all.append(np.mean(robust_rewards))
        # robust_rewards_nn_all.append(np.mean(robust_rewards_nn))
        best_rewards_all.append(np.mean(best_rewards))
        baseline_rewards_all.append(np.mean(baseline_rewards))
        kernel_rewards_all.append(np.mean(kernel_rewards))

        robust_vars.append(np.std(robust_rewards))
        # print('robust var:', robust_vars)
        # robust_vars_nn = np.std(robust_rewards_nn)
        best_vars.append(np.std(best_rewards))
        baseline_vars.append(np.std(baseline_rewards))
        # print('baseline var:', baseline_vars)
        # kernel_vars.append(np.std(kernel_rewards))
    
        print("robust:", robust_rewards[-1])
        print("best:", best_rewards_all[-1]) 
        print("baseline:", baseline_rewards_all[-1]) 
        print("kernel:", kernel_rewards_all[-1])

    # plt.errorbar(sample_counts, robust_rewards_nn_all, robust_vars_nn, label = 'robust nn', color = 'red')
    # plt.errorbar(sample_counts, best_rewards_all, best_vars, label = 'optimal', linestyle='--', color = 'green')
    # plt.errorbar(sample_counts, baseline_rewards_all, baseline_vars, label = 'baseline', linestyle='-.', color = 'orange')
    # plt.errorbar(sample_counts, kernel_rewards_all, kernel_vars, label = 'kernel', linestyle=':', color = 'purple')
    # plt.errorbar(sample_counts, robust_rewards_all, robust_vars, label = 'robust', color = 'blue')
    
    plt.plot(sample_counts, best_rewards_all, label = 'optimal', linestyle='--', color = 'green')
    plt.plot(sample_counts, baseline_rewards_all, label = 'baseline', linestyle='-.', color = 'orange')
    plt.plot(sample_counts, kernel_rewards_all, label = 'kernel', linestyle=':', color = 'purple')
    plt.plot(sample_counts, robust_rewards_all, label = 'robust', color = 'blue')

    plt.legend()
    plt.xlabel('data size')
    plt.ylabel('Average reward')
    plt.title('Reward as a function of data size')
    plt.show()

    # plt.plot(sample_counts, baseline_frac, label = 'baseline', linestyle='-.', color = 'orange')
    # plt.plot(sample_counts, kernel_frac, label = 'kernel', linestyle=':', color = 'purple')
    # plt.plot(sample_counts, robust_frac, label = 'robust', color = 'blue')

    # plt.legend()
    # plt.xlabel('data size')
    # plt.ylabel('Percent win')
    # plt.title('Percent win as a function of data size')

    # plt.show()
