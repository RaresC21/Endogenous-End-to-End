import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import random
from tqdm import tqdm
from time import time


def save_data(actions, demands, agent):
    ''' Save 2 columns array of actions and observed demands for the agent '''
    data = np.column_stack((actions, demands))
    np.save('data/' + str(agent) + '.npy', data)


# pick random number to define the seed for this run
n = random.randint(0, 1000000)

class Environment(object):
    def __init__(self, bandit, agents, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.agents = agents
        self.label = label

    def reset(self):
        action_values = self.bandit.reset()
        for agent in self.agents:
            if 'LinUCB' in str(agent):
                agent.reset(action_values)
            else:
                agent.reset()

    def run(self, actions, trials=100, experiments=1):
        scores = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(scores)

        dic_data = {}
        for i, agent in enumerate(self.agents):
            dic_data[str(agent)] = []

        for k in tqdm(range(experiments)):
            self.reset()
            # first step: choose each action once to initialize the counts at 1
            for i, agent in enumerate(self.agents):
                if 'LinUCB' not in str(agent): # LinUCB doesn't need initialization
                    for action, price in enumerate(actions):
                        action, price = agent.initialize(action, price)
                        reward_demand, reward, is_optimal = self.bandit.pull(action, price, method=str(agent))
                        if 'Adapted' in str(agent) or 'Optimistic' in str(agent):
                            agent.observe(reward_demand) # different for UCB and adapted UCB (different observed reward)
                        else:
                            agent.observe(reward)
                        scores[0, i] += reward / len(actions)
                        if is_optimal:
                            optimal[0, i] += 1 / len(actions)

            for t in range(1, trials):
                random.seed(n)   # IMPORTANT STEP
                self.bandit.reset()
                for i, agent in enumerate(self.agents):
                    # if 'LinUCB' in str(agent):
                    #     p = agent.update()
                    #     action, price = agent.choose(p), actions[action]
                    #     score, is_optimal = agent.observe(action)
                    #     scores[t, i] += score
                    #     if is_optimal:
                    #         optimal[t, i] += 1

                    if 'LinUCB' not in str(agent):
                        action, price = agent.choose()  # index and corresponding action (price) 
                        reward_demand, reward, is_optimal = self.bandit.pull(action, price, method=str(agent))
                        if 'Adapted' in str(agent) or 'Optimistic' in str(agent):
                            agent.observe(reward_demand) # different for UCB and adapted UCB (different observed reward)
                            if 'NN' in str(agent):
                                print(reward_demand)
                        else:
                            agent.observe(reward)
                        
                        if k == 0:
                            dic_data[str(agent)].append((actions[action], reward_demand))

                        scores[t, i] += reward 
                        if is_optimal:
                            optimal[t, i] += 1
                
            # run the process for LinUCB
            for i, agent in enumerate(self.agents):
                if 'LinUCB' in str(agent):
                    random.seed(n)    # TO PICK SAME RANDOM VALUES AS FOR THE OTHER AGENTS
                    rewards, optimal_actions, list_actions, list_demands = agent.run(trials)
                    scores[:, i] += rewards
                    optimal[:, i] += optimal_actions
                    if k == 0:
                        for j in range(len(list_actions)):
                            dic_data[str(agent)].append((list_actions[j], list_demands[j]))

            # save data for each agent
            if k == 0:
                for agent in self.agents:
                    save_data(np.array(dic_data[str(agent)])[:, 0], np.array(dic_data[str(agent)])[:, 1], agent)
        

        return scores / experiments, optimal / experiments

    def plot_results(self, scores, optimal):
        optimal = optimal * 100
        sns.set_style('white')
        sns.set_context('talk')
        plt.subplot(3, 1, 1)
        plt.title(self.label)
        plt.plot(scores)
        #running average for each agent
        # for i in range(scores.shape[1]):
        #     plt.plot(np.convolve(scores[:, i], np.ones(100)/100, mode='valid'), lw=2)
        plt.plot([0, len(scores)], [self.bandit.optimal_reward()] * 2, 'k--')
        plt.ylabel('Average Reward')
        plt.legend(self.agents, loc=4)
        # add a new subplot with running average of rewards for each agent
        plt.subplot(3, 1, 2)
        for i in range(scores.shape[1]):
            plt.plot(np.convolve(scores[:, i], np.ones(100)/100, mode='valid'), lw=2)
        plt.plot([0, len(scores)], [self.bandit.optimal_reward()] * 2, 'k--')
        plt.ylabel('Running Average Reward')
        plt.legend(self.agents, loc=4)
        plt.subplot(3, 1, 3)
        # for i in range (optimal.shape[1]):
        #     plt.plot(optimal[:, i])
        plt.plot(optimal[:, 0])
        plt.plot(optimal[:, 1])
        plt.plot(optimal[:, 2])
        plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)
        sns.despine()
        plt.show()

    def plot_beliefs(self):
        sns.set_context('talk')
        pal = sns.color_palette("cubehelix", n_colors=len(self.agents))
        plt.title(self.label + ' - Agent Beliefs')

        rows = 2
        cols = int(self.bandit.k / 2)

        axes = [plt.subplot(rows, cols, i+1) for i in range(self.bandit.k)]
        for i, val in enumerate(self.bandit.action_values):
            color = 'r' if i == self.bandit.optimal else 'k'
            axes[i].vlines(val, 0, 1, colors=color)

        for i, agent in enumerate(self.agents):
            for j, val in enumerate(agent.value_estimates):
                axes[j].vlines(val, 0, 0.75, colors=pal[i], alpha=0.8)

        min_p = np.argmin(self.bandit.action_values)
        for i, ax in enumerate(axes):
            ax.set_xlim(0, 1)
            if i % cols != 0:
                ax.set_yticklabels([])
            if i < cols:
                ax.set_xticklabels([])
            else:
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0', '', '0.5', '', '1'])
            if i == int(cols/2):
                title = '{}-arm Bandit - Agent Estimators'.format(self.bandit.k)
                ax.set_title(title)
            if i == min_p:
                ax.legend(self.agents)

        sns.despine()
        plt.show()