class NaiveAgent(object):
    def __init__(self, n_arms, policy):
        self.policy = policy
        self.k = n_arms
        self.t = 0
        self.last_action = None

    def __str__(self):
        return str(self.policy)
    
    def reset(self):
        self.t = 0
        self.last_action = None

    def choose(self, alpha_0, alpha_1):
        action, price = self.policy.choose(alpha_0, alpha_1)
        self.last_action = action
        return action, price
    
    def initialize(self, actions, demands):
        return self.policy.initialize(actions, demands)


class AgentRobustLearning(object):
    """Adaptation for RobustLearning Policy"""
    def __init__(self, n_arms, policy, action_values):
        self.policy = policy
        self.k = n_arms
        self.actions_list = []
        self.demands_list = []
        self.sum_a = 0
        self.sum_a_squared = 0
        self.sum_d = 0
        self.sum_d_squared = 0
        self.sum_ad = 0
        self.t = 0
        self.last_action = None
        self.action_values = action_values
    
    def __str__(self):
        return str(self.policy)

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self.actions_list = []
        self.demands_list = []
        self.sum_a = 0
        self.sum_a_squared = 0
        self.sum_d = 0
        self.sum_d_squared = 0
        self.sum_ad = 0
        self.last_action = None
        self.t = 0

    def choose(self):
        action, price = self.policy.choose(self)   # index and price
        self.last_action = action
        return action, price
    
    def initialize(self, action, price):
        self.last_action = action
        return action, price

    def observe(self, demand):
        '''Store the sum of past demands and store every action'''
        last_action = self.last_action
        self.actions_list.append(last_action)
        self.demands_list.append(demand)
        last_action_value = self.action_values[last_action]
        self.sum_a_squared += last_action_value**2
        self.sum_a += last_action_value
        self.sum_d += demand
        self.sum_d_squared += demand**2
        self.sum_ad += demand*last_action_value
        self.t += 1




class AgentRobustLearningNN(object):
    """Adaptation for RobustLearning Policy"""
    def __init__(self, n_arms, policy, action_values):
        self.policy = policy
        self.k = n_arms
        self.actions_list = []
        self.demands_list = []
        self.t = 0
        self.last_action = None
        self.action_values = action_values
    
    def __str__(self):
        return str(self.policy)

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self.actions_list = []
        self.demands_list = []
        self.last_action = None
        self.t = 0

    def choose(self, trained_net):
        action, price = self.policy.choose(self, trained_net)   # index and price
        self.last_action = action
        return action, price
    
    def initialize(self, action, price):
        self.last_action = action
        return action, price
    
    def train_initial_nn(self, actions, demands):
        return self.policy.initialize(actions, demands)

    def observe(self, demand):
        '''Store the sum of past demands and store every action'''
        last_action = self.last_action
        self.actions_list.append(last_action)
        self.demands_list.append(demand)
        self.t += 1
