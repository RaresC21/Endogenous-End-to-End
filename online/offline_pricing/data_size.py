import numpy as np
from sklearn.linear_model import LinearRegression
import gurobipy as gp
import nn_model
import matplotlib.pyplot as plt 
from tqdm import tqdm

from helper import * 


if __name__ == '__main__':

    sample_counts = [1 * i for i in range(2, 20)] + [10 * i for i in range(3, 10, 2)]   # [1 * i for i in range(2, 20)] + 
    # n_actions = 50
    for n_actions in [50, 100, 500]:
        action_set = set_actions(n_actions)

        robust_actions = []

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

        for n_samples in sample_counts:
            robust_rewards = []
            robust_rewards_nn = []
            kernel_rewards = []
            baseline_rewards = []
            best_rewards = []

            for _ in tqdm(range(30)): 
                actions_train, demands_train = generate_data(action_set, n_samples)
                # actions_train, demands_train = load_data_online('Adapted UCB')
                # actions_train, demands_train = actions_train[:n_samples], demands_train[:n_samples]

                # train nn model
                # net = nn_model.Net()
                # trained_net = nn_model.train(net, torch.tensor(actions_train).float().view(-1, 1), torch.tensor(demands_train).float())

                best_action_index = np.argmax(action_set * get_mean_demand(action_set))
                best_action = action_set[best_action_index]
                best_rewards.append(best_action * get_mean_demand(best_action))

                baseline_action_index = two_stage_action(actions_train, demands_train, action_set)
                baseline_action = action_set[baseline_action_index]
                baseline_rewards.append(baseline_action * get_mean_demand(baseline_action))

                robust_action_index, _ = robust_action(actions_train, demands_train, action_set, 0.8)
                robust_action_ = action_set[robust_action_index]
                robust_rewards.append(robust_action_ * get_mean_demand(robust_action_))

                #kernel_action_index, _ = kernel_action_nn(actions_train, demands_train, action_set, trained_net, 0.001)
                kernel_action_index, _ = kernel_action(actions_train, demands_train, action_set, 0.001)
                kernel_action_ = action_set[kernel_action_index]
                kernel_rewards.append(kernel_action_ * get_mean_demand(kernel_action_))

                # robust_action_index_nn, _ = robust_action_nn(actions_train, demands_train, action_set, trained_net, 1)
                # robust_action_nn_ = action_set[robust_action_index_nn]
                # robust_rewards_nn.append(robust_action_nn_ * get_mean_demand(robust_action_nn_))

                # print("robust action: ", robust_action_)
                # print("robust action nn: ", robust_action_nn_)

            robust_rewards_all.append(np.mean(robust_rewards))
            # robust_rewards_nn_all.append(np.mean(robust_rewards_nn))
            best_rewards_all.append(np.mean(best_rewards))
            baseline_rewards_all.append(np.mean(baseline_rewards))
            kernel_rewards_all.append(np.mean(kernel_rewards))

            robust_vars = np.std(robust_rewards)
            # robust_vars_nn = np.std(robust_rewards_nn)
            best_vars = np.std(best_rewards)
            baseline_vars = np.std(baseline_rewards)
            kernel_vars = np.std(kernel_rewards)
        
        plt.errorbar(sample_counts, robust_rewards_all, robust_vars, label = 'robust', color = 'blue')
        # plt.errorbar(sample_counts, robust_rewards_nn_all, robust_vars_nn, label = 'robust nn', color = 'red')
        plt.errorbar(sample_counts, best_rewards_all, best_vars, label = 'optimal', linestyle='--', color = 'green')
        plt.errorbar(sample_counts, baseline_rewards_all, baseline_vars, label = 'baseline', linestyle='-.', color = 'orange')
        plt.errorbar(sample_counts, kernel_rewards_all, kernel_vars, label = 'kernel', linestyle=':', color = 'purple')
        
        plt.legend()
        plt.xlabel('data size')
        plt.ylabel('Average reward')
        plt.title('Reward as a function of data size')
        plt.savefig(f'offline_learning/images/kernel_lognormal_n_actions={n_actions}_quadratic_var0.1.png')
        plt.show()
