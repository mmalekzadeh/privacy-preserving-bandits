import numpy as np

###____________  BasePolicy Interface ____________###
class BasePolicy:
    """
    num_actions: (int) Number of arms [indexed by 0 ... num_actions-1]
    num_contexts: (int) Number of features [indexed by 0 ... num_contexts-1].
    """
    def __init__(self, num_actions, context_combinations_size):
        self.num_actions = num_actions
        self.context_combinations_size = context_combinations_size
        self.total_rewards = np.zeros((context_combinations_size, num_actions), dtype = np.longdouble)
        self.total_counts = np.zeros((context_combinations_size, num_actions), dtype = np.longdouble)
    
    def act(self, context):
        pass
        
    def feedback(self, action, reward, context):
        pass

###____________  Random  ____________###
class Random(BasePolicy):
    def __init__(self, num_actions):
        BasePolicy.__init__(self, num_actions, 1)
        self.name = "Random"

    def act(self, context=None):
        next_action = np.random.choice(self.num_actions)
        return next_action
    
    def feedback(self, action, reward, context=None): 
        self.total_rewards[action] += reward
        self.total_counts[action] += 1

###____________  Greedy ____________###
class Greedy(BasePolicy):
    
    def __init__(self, num_actions, context_combinations_size=1, optimistic=0.0):
        BasePolicy.__init__(self, num_actions, context_combinations_size)
        self.name = "Greedy"
        self.optimistic = optimistic
    
    def act(self, context=None, context_code=0):
        current_averages = np.divide(self.total_rewards[context_code],
                                     self.total_counts[context_code], 
                                     where = self.total_counts[context_code] > 0)
        current_averages[self.total_counts[context_code] <= 0] = self.optimistic
        current_action = np.argmax(current_averages)
        return current_action
        
    def feedback(self, action, reward, context=None):
        if context is not None:
            context_code = int("".join(str(x) for x in context), 2) 
        else:
            context_code = 0
        self.total_rewards[context_code, action] += reward
        self.total_counts[context_code, action] += 1

###____________  EpsilonGreedy ____________###
class EpsilonGreedy(Greedy):
    def __init__(self, num_actions, context_combinations_size=1, optimistic=0., epsilon=0.):
        Greedy.__init__(self, num_actions, context_combinations_size, optimistic=optimistic)
        if (epsilon is None or epsilon < 0 or epsilon > 1):
            print("EpsilonGreedy: Invalid value of epsilon", flush = True)
        self.name = "Epsilon Greedy"
        self.epsilon = epsilon
    
    def act(self, context=None, context_code=0):
        choice = None
        if self.epsilon == 0:
            choice = 0
        elif self.epsilon == 1:
            choice = 1
        else:
            choice = np.random.binomial(1, self.epsilon)
            
        if choice == 1:
            current_action =  np.random.choice(self.num_actions)
        else:
            current_averages = np.divide(self.total_rewards[context_code],
                                     self.total_counts[context_code], 
                                     where = self.total_counts[context_code] > 0)
            current_averages[self.total_counts[context_code] <= 0] = self.optimistic
            current_action = np.argmax(current_averages)
        return current_action

###____________  UCB ____________###
class UCB(Greedy):
    def __init__(self, num_actions, context_combinations_size=1):
        Greedy.__init__(self, num_actions, context_combinations_size)
        self.name = "UCB"
        self.round = np.zeros(context_combinations_size).astype(int)
        
    def act(self, context=None, context_code=0):
        current_action = None
        if self.round[context_code] < self.num_actions:
            current_action = self.round[context_code]
        else:
            current_averages = np.divide(self.total_rewards[context_code],
                                         self.total_counts[context_code])
            current_delta = np.sqrt(np.divide(2 * np.log(self.round[context_code]+1),
                                              self.total_counts[context_code]))
            current_averages_delta = current_averages + current_delta
            current_action = np.argmax(current_averages_delta)
        
        self.round[context_code] += 1
        return current_action