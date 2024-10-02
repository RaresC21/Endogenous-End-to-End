import numpy as np
from sklearn.linear_model import LinearRegression
import gurobipy as gp
import nn_model
import matplotlib.pyplot as plt 
from tqdm import tqdm
import seaborn as sns
import pandas as pd

from helper import * 


if __name__ == '__main__':

    actions, demands = load_data_online('Adapted UCB')
    # plot the evolution of the reward with the number of samples
    # plt.plot([a*d for a, d in zip(actions, demands)])
    # plt.show()
    # plt.hist(actions)
    # plt.show()
    # plt.plot(actions)
    # plt.show()

    n_actions = 15
    action_set = set_actions(n_actions)
    n_samples = 500

    robust_rewards = []
    robust_rewards_nn = []
    baseline_rewards = []
    best_rewards = []

    robust_rewards_online = []
    robust_rewards_nn_online = []
    baseline_rewards_online = []
    best_rewards_online = []

    robust_list_actions = []
    robust_nn_list_actions = []
    baseline_list_actions = []

    robust_online_list_actions = []
    robust_nn_online_list_actions = []
    baseline_online_list_actions = []


    for _ in tqdm(range(100)): 
        actions_train, demands_train = generate_data(action_set, n_samples)
        actions_train_online, demands_train_online = load_data_online('Adapted UCB')
        actions_train_online, demands_train_online = actions_train_online[:n_samples], demands_train_online[:n_samples]

        # train nn model
        # net = nn_model.Net()
        # trained_net = nn_model.train(net, torch.tensor(actions_train).float().view(-1, 1), torch.tensor(demands_train).float())

        best_action_index = np.argmax(action_set * get_mean_demand(action_set))
        best_action = action_set[best_action_index]
        best_rewards.append(best_action * get_mean_demand(best_action))

        baseline_action_index = two_stage_action(actions_train, demands_train, action_set)
        baseline_action = action_set[baseline_action_index]
        baseline_rewards.append(baseline_action * get_mean_demand(baseline_action))

        robust_action_index, _ = robust_action(actions_train, demands_train, action_set, 0.5)
        robust_action_ = action_set[robust_action_index]
        robust_rewards.append(robust_action_ * get_mean_demand(robust_action_))

        # robust_action_index_nn, _ = robust_action_nn(actions_train, demands_train, action_set, trained_net, 1)
        # robust_action_nn_ = action_set[robust_action_index_nn]
        # robust_rewards_nn.append(robust_action_nn_ * get_mean_demand(robust_action_nn_))

        # same thing for online
        # net = nn_model.Net()
        # trained_net = nn_model.train(net, torch.tensor(actions_train_online).float().view(-1, 1), torch.tensor(demands_train_online).float())

        best_action_index_online = np.argmax(action_set * get_mean_demand(action_set))
        best_action_online = action_set[best_action_index_online]
        best_rewards_online.append(best_action_online * get_mean_demand(best_action_online))

        baseline_action_index_online = two_stage_action(actions_train_online, demands_train_online, action_set)
        baseline_action_online = action_set[baseline_action_index_online]
        baseline_rewards_online.append(baseline_action_online * get_mean_demand(baseline_action_online))

        robust_action_index_online, _ = robust_action(actions_train_online, demands_train_online, action_set, 0.5)
        robust_action_online_ = action_set[robust_action_index_online]
        robust_rewards_online.append(robust_action_online_ * get_mean_demand(robust_action_online_))

        # robust_action_index_nn_online, _ = robust_action_nn(actions_train_online, demands_train_online, action_set, trained_net, 1)
        # robust_action_nn_online_ = action_set[robust_action_index_nn_online]
        # robust_rewards_nn_online.append(robust_action_nn_online_ * get_mean_demand(robust_action_nn_online_))

        print('Robust action: ', robust_action_index, robust_action_)
        print('Robust action online: ', robust_action_index_online, robust_action_online_)
        print('Baseline action: ', baseline_action_index, baseline_action)
        print('Best action: ', best_action_index, best_action)
        print('Best action online: ', best_action_index_online, best_action_online) 

        # print('robust', robust_action_ * get_demand(robust_action_))
        # print('robust online', robust_action_online_ * get_demand(robust_action_online_))
        # print('best', best_action_online * get_demand(best_action_online))


        robust_list_actions.append(robust_action_index == best_action_index)
        # robust_nn_list_actions.append(robust_action_index_nn == best_action_index)
        baseline_list_actions.append(baseline_action_index == best_action_index)

        robust_online_list_actions.append(robust_action_index_online == best_action_index_online)
        # robust_nn_online_list_actions.append(robust_action_index_nn_online == best_action_index_online)
        baseline_online_list_actions.append(baseline_action_index_online == best_action_index_online)

    robust_reward = np.mean(robust_rewards)
    # robust_reward_nn = np.mean(robust_rewards_nn)
    best_reward = np.mean(best_rewards)
    baseline_reward = np.mean(baseline_rewards)

    robust_var = np.std(robust_rewards)
    # robust_var_nn = np.std(robust_rewards_nn)
    best_var = np.std(best_rewards)
    baseline_var = np.std(baseline_rewards)

    robust_reward_online = np.mean(robust_rewards_online)
    # robust_reward_nn_online = np.mean(robust_rewards_nn_online)
    best_reward_online = np.mean(best_rewards_online)
    baseline_reward_online = np.mean(baseline_rewards_online)

    robust_var_online = np.std(robust_rewards_online)
    # robust_var_nn_online = np.std(robust_rewards_nn_online)
    best_var_online = np.std(best_rewards_online)
    baseline_var_online = np.std(baseline_rewards_online)

    proportion_robust = np.mean(robust_list_actions)
    # proportion_robust_nn = np.mean(robust_nn_list_actions)
    proportion_baseline = np.mean(baseline_list_actions)

    proportion_robust_online = np.mean(robust_online_list_actions)
    # proportion_robust_nn_online = np.mean(robust_nn_online_list_actions)
    proportion_baseline_online = np.mean(baseline_online_list_actions)
    
        
    # 2 histogram plot with 4 bars for mean and variance

    # now for each data generation (online and random), plot 2 bars for each method (one blue bar and one red bar). Use seaborn

    reward_data = {
    'Agent': ['robust', 
            #   'robust_nn', 
              'baseline'] * 2,
    'Average Reward': [robust_reward, 
                    #    robust_reward_nn, 
                       baseline_reward,
                       robust_reward_online, 
                    #    robust_reward_nn_online, 
                       baseline_reward_online],
    'Type': ['Random'] * 2 + ['Online'] * 2
    }
    reward_df = pd.DataFrame(reward_data)

    # Create a dataframe for the average standard deviation data
    variance_data = {
        'Agent': ['robust', 
                #   'robust_nn', 
                  'baseline'] * 2,
        'Average Standard Deviation': [robust_var, 
                                    #    robust_var_nn, 
                                       baseline_var,
                                       robust_var_online, 
                                    #    robust_var_nn_online, 
                                       baseline_var_online],
        'Type': ['Random'] * 2 + ['Online'] * 2
    }
    variance_df = pd.DataFrame(variance_data)

    proportion_data = {
        'Agent': ['robust', 
                #   'robust_nn', 
                  'baseline'] * 2,
        'Proportion': [proportion_robust, 
                    #    proportion_robust_nn, 
                       proportion_baseline,
                       proportion_robust_online, 
                    #    proportion_robust_nn, 
                       proportion_baseline_online],
        'Type': ['Random'] * 2 + ['Online'] * 2
    }

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('n_samples = {}'.format(n_samples))

    ax1 = fig.add_subplot(3, 1, 1)
    sns.barplot(x='Agent', y='Average Reward', hue='Type', data=reward_df, palette=['blue', 'red'], ax=ax1)
    ax1.axhline(y=best_reward, color='r', linestyle='-')
    ax1.set_ylabel('Average Reward')
    ax1.set_xlabel('Agent')

    ax2 = fig.add_subplot(3, 1, 2)
    sns.barplot(x='Agent', y='Average Standard Deviation', hue='Type', data=variance_df, palette=['blue', 'red'], ax=ax2)
    ax2.axhline(y=best_var, color='r', linestyle='-')
    ax2.set_ylabel('Average Standard Deviation')
    ax2.set_xlabel('Agent')

    ax3 = fig.add_subplot(3, 1, 3)
    sns.barplot(x='Agent', y='Proportion', hue='Type', data=proportion_data, palette=['blue', 'red'], ax=ax3)
    ax3.set_ylabel('Proportion')
    ax3.set_xlabel('Agent')

    print(proportion_robust_online)
    print(proportion_robust)
    print(robust_reward_online)
    print(robust_reward)


    plt.savefig('offline_learning/images/seaborn_random_data_online_' + 'n_samples = {}'.format(n_samples) + '.png', bbox_inches='tight')