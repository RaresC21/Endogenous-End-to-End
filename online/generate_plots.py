import os
import sys
from agent import Agent, AgentOptimisticLearning
from bandit import GaussianBandit
from environment import Environment
from policy import (EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy, UCBPolicy, AdaptedUCBPolicy, LinUCBPolicy, OptimisticLearningPolicy, OptimisticLearningPolicyNN)
from data_generation import generate_data
from utils import g_function
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import random



def plot_results(scores, optimal, n_arms, mean, var_coeff, n_trials, n_experiments, agents_names, optimal_reward):
    # create title based on outputs 
    mean_str = '(1-p)' if mean == 'linear' else '(1-p)^2'
    var_str = mean_str + '*{}'.format(var_coeff)
    title = 'n_arms={}, mean=${}$, var=${}$, n_trials={}, n_experiments={}'.format(n_arms, mean_str, var_str, n_trials, n_experiments)
    optimal = optimal * 100
    # Create a figure with subplots and set the figure size
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    plt.suptitle(title)

    axs[0].plot(scores)
    axs[0].plot([0, len(scores)], [optimal_reward] * 2, 'k--')
    axs[0].set_ylabel('Average Reward')
    axs[0].legend(agents_names, loc=4)

    axs[1].set_title("Running Average Reward")
    for i in range(scores.shape[1]):
        axs[1].plot(np.convolve(scores[:, i], np.ones(100)/100, mode='valid'), lw=2)
    axs[1].plot([0, len(scores)], [optimal_reward] * 2, 'k--')
    axs[1].set_ylabel('Running Average Reward')
    axs[1].legend(agents_names, loc=4)

    axs[2].plot(optimal)
    axs[2].set_ylim(0, 100)
    axs[2].set_ylabel('% Optimal Action')
    axs[2].set_xlabel('Time Step')
    axs[2].legend(agents_names, loc=4)

    plt.savefig('images/curves/' + title + '.png', bbox_inches='tight')

def plot_hist(scores, optimal, n_arms, mean, var_coeff, n_trials, n_experiments, agents_names, optimal_reward):
    '''Plot a single histogram with the averaged achieved reward at time step n_trials for every of the agents '''
    mean_str = '(1-p)' if mean == 'linear' else '(1-p)^2'
    var_str = mean_str + '*{}'.format(var_coeff)
    title = 'n_arms={}, mean=${}$, var=${}$, n_trials={}, n_experiments={}'.format(n_arms, mean_str, var_str, n_trials, n_experiments)
    fig = plt.figure(figsize=(10, 10))

    fig.suptitle(title)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_ylabel('Average Reward at time step {}'.format(n_trials))
    ax1.set_xlabel('Agent')
    # give a different to the highest bar
    ax1.bar(np.arange(len(agents_names)), scores[n_trials - 1, :], color=['b' if score != np.max(scores[n_trials - 1, :]) else 'r' for score in scores[n_trials - 1, :]])
    ax1.set_xticks(np.arange(len(agents_names)))
    ax1.set_xticklabels(agents_names)
    ax1.axhline(y=optimal_reward, color='r', linestyle='-')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_ylabel('% Optimal Action at time step {}'.format(n_trials))
    ax2.set_xlabel('Agent')
    # give a different to the highest bar
    ax2.bar(np.arange(len(agents_names)), optimal[n_trials - 1, :]*100, color=['b' if score != np.max(optimal[n_trials - 1, :]) else 'r' for score in optimal[n_trials - 1, :]])
    ax2.set_xticks(np.arange(len(agents_names)))
    ax2.set_xticklabels(agents_names)
    ax2.axhline(y=100, color='r', linestyle='-')

    plt.savefig('images/hist/' + title + '.png', bbox_inches='tight')


def plot_var(results, var_coeff_list, n_arms=10, mean='linear', n_trials=2000, n_experiments=500):
    ''' plot how the average reward and % optimal action change with the variance coefficient for the different agents '''
    mean_str = '(1-p)' if mean == 'linear' else '(1-p)^2'
    title = 'n_arms={}, mean=${}$, n_trials={}, n_experiments={}'.format(n_arms, mean_str, n_trials, n_experiments)
    # for fixed mean and n_arms, store the results for each variance coefficient in a list
    results_var = [results[(n_arms, mean, var_coeff)] for var_coeff in var_coeff_list]
    # 2 subplots for the evolution of the average reward and % optimal action with the variance coefficient
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))
    plt.suptitle(title)
    # plot average reward evolution with variance coefficient, at time step n_trials
    axs[0].set_title("Average Reward")
    agent_names = results_var[0][3]
    for i in range(len(agent_names)):
        score = [results_var[j][0][n_trials - 1][i] for j in range(len(results_var))]
        axs[0].plot(var_coeff_list, score)
    axs[0].legend(agent_names, loc=4)
                       
    axs[0].set_ylabel('Average Reward')
    axs[0].set_xlabel('Variance Coefficient')

    # plot % optimal action evolution with variance coefficient
    axs[1].set_title("% Optimal Action")
    for i in range(len(agent_names)):
        actions = [results_var[j][1][n_trials - 1][i] for j in range(len(results_var))]
        axs[1].plot(var_coeff_list, actions)
    axs[1].legend(agent_names, loc=4)
        
    axs[1].set_ylabel('% Optimal Action')
    axs[1].set_xlabel('Variance Coefficient')

    plt.savefig('images/variance/' + title + '.png', bbox_inches='tight')




def generate_plots(n_arms=10, mean='linear', var_coeff=1, n_trials=2000, n_experiments=500):
    actions, parameters = generate_data(n_arms)
    mu, sigma = zip(*parameters)
    bandit = GaussianBandit(n_arms, mu, sigma, actions)
    agents = [
    Agent(bandit, EpsilonGreedyPolicy(0.1, actions)),
    Agent(bandit, UCBPolicy(actions)),
    #Agent(bandit, RandomPolicy(actions)),
    Agent(bandit, AdaptedUCBPolicy(actions, g_function)),
    AgentOptimisticLearning(bandit, OptimisticLearningPolicy(actions), actions),
    AgentOptimisticLearning(bandit, OptimisticLearningPolicyNN(actions), actions),
    LinUCBPolicy(actions, n_arms, mu, sigma, version='1'),
    LinUCBPolicy(actions, n_arms, mu, sigma, version='2'),
    ]
    env = Environment(bandit, agents)
    scores, optimal = env.run(actions, n_trials, n_experiments)
    optimal_reward = bandit.optimal_reward()
    agents_names = [str(agent) for agent in agents]
    plot_results(scores, optimal, n_arms, mean, var_coeff, n_trials, n_experiments, agents_names, optimal_reward)
    plot_hist(scores, optimal, n_arms, mean, var_coeff, n_trials, n_experiments, agents_names, optimal_reward)
    return scores, optimal, optimal_reward, agents_names



if __name__ == '__main__':
    #generate_plots(10, 'linear', 1, 2000, 500)

    # generate plots for different parameters
    n_arms_list = [10]
    mean_list = ['linear']
    var_coeff_list = [0.1]
    # list with every combination of parameters
    params = [(n_arms, mean, var_coeff) for n_arms in n_arms_list for mean in mean_list for var_coeff in var_coeff_list]
    dict_var = {}
    for (n_arms, mean, var_coeff) in tqdm(params):
        score, optimal, optimal_reward, agent_names = generate_plots(n_arms, mean, var_coeff, 300, 2)
        dict_var[(n_arms, mean, var_coeff)] = (score, optimal, optimal_reward, agent_names)
        # save the actions and observed demands for the agents

    # plot how the average reward and % optimal action change with the variance coefficient for the different agents
    # plot_var(dict_var, var_coeff_list, n_arms=5, mean='linear', n_trials=2000, n_experiments=500)

