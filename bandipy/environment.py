import numpy as np
from bandipy import prior

class ContextSpace:
    def __init__(self, contexts,
                context_type=None, contexts_dist=None,
                contexts_dist_params=None, contexts_data=None):
        self.contexts = contexts
        self.size = len(contexts)
        self.context_type = context_type
        self.combinations = 0
        if context_type == "binary":
            self.combinations_size = 2**(len(contexts))
        elif context_type == "categorical":
            self.combinations_size = -1 # Need to be implemented 
        elif context_type == "real" or context_type is None:
            self.combinations_size = float("inf")
        else:
            print("\"context_type\" is invalid!!!")
        
        self.contexts_dist = contexts_dist
        self.context_parameters = None
        if contexts_dist == "binomial" or contexts_dist is None:
            self.context_parameters = contexts_dist_params
        else:
            print("The context distribution is unknown!")
        
        self.contexts_data = contexts_data
        
        
class ActionSpace:
    def __init__(self, actions):
        self.actions = actions
        self.size = len(actions)
        
class RewardSpace:
    def __init__(self, rewards, reward_type=None,
                reward_dist=None, reward_dist_params=None,
                reward_data=None):
        self.rewards = rewards
        self.size = 0
        if reward_type in ["binary","categorical"]:
            self.size = len(rewards)
        elif reward_type == "real" or reward_type is None:
            self.size = float("inf")
        else:
            print("\"reward_type\" is invalid!!!")
            
        self.reward_dist = reward_dist
        self.reward_parameters = None
        self.optimal_arm = None
        if reward_dist in ["bernoulli","given"]:
            self.reward_parameters = reward_dist_params
            self.optimal_arm = np.argmax(self.reward_parameters)
        elif reward_dist is None:
            self.reward_parameters = reward_dist_params
        else:
            print("The reward distribution is unknown!")

        self.reward_data= reward_data

###____________  ContextualBanditEnv ____________###
class ContextualBanditEnv:
    def __init__(self, context_s=None, context_ns=None, rewards=None,
                 rnd_seed=0, rand_gen = np.random):
        
        #self.random = rand_gen
        #self.random.seed(rnd_seed)

        context_data = None
        if context_ns is None:
            context_data = context_s
        else:
            context_data = np.concatenate((context_s,context_ns), axis=1)

        num_contexts = context_data.shape[1]
        self.context_space = ContextSpace(range(num_contexts), contexts_data=context_data)
        
        if len(rewards.shape) == 1:
            num_actions = context_data.shape[1]
        elif len(rewards.shape) == 2:
            num_actions = rewards.shape[1]
        
        self.action_space = ActionSpace(range(num_actions))
        
        num_rewards = 2
        self.reward_space = RewardSpace(range(num_rewards), reward_data=rewards)

    def get_context(self, trial=0):
        return self.context_space.contexts_data[trial]
    
    def step(self, action, context, trial=0):
        
        valid_action = True
        if (action is None or action < 0 or action >= self.action_space.size):
            print("Algorithm chose an invalid action; reset reward to -inf", flush = True)
            reward = 0
            valid_action = False

        reward = -1
        if valid_action:
            if len(self.reward_space.reward_data.shape) == 1:
                reward = self.reward_space.reward_data[trial]
            elif len(self.reward_space.reward_data.shape) == 2:
                reward = self.reward_space.reward_data[trial, action]

        return reward

class RandomBanditEnv:
    def __init__(self,
                 num_contexts = 1,        # Contexts
                 context_type="binary", contexts_dist="binomial", contexts_dist_params=None,
                 num_actions = 1,         # Actions
                 num_rewards = 2,         # Rewards
                 reward_type = "binary", reward_dist = "bernoulli", reward_dist_params=None,
                 rnd_seed=0, rand_gen = np.random):
        #self.random = rand_gen
        #self.random.seed(rnd_seed)
        
        self.context_space = ContextSpace(range(num_contexts), context_type,
                                          contexts_dist, contexts_dist_params)
        self.action_space = ActionSpace(range(num_actions))
        self.reward_space = RewardSpace(range(num_rewards), reward_type,
                                        reward_dist, reward_dist_params)
    
    def binomial_reward(self, prob):
        return self.random.binomial(1, prob)
    
    def get_context(self, size=1):
        if self.context_space.contexts_dist == 'binomial':
            cp = prior.ContextPriors()
            return cp.binomial_prior(probs=self.context_space.context_parameters)
    
    def contextual_reward(self, reward_prob, context):
        reward_prob = reward_prob if context.mean() > .5 else 1-reward_prob # A Sample Implementation
        reward_prob = np.clip(reward_prob, a_min =0, a_max=1) 
        return self.binomial_reward(reward_prob)
    
    def compute_reward(self, reward_prob, context):
        if self.context_space.size == 0:
            return self.binomial_reward(reward_prob) # A Sample Implementation
        return self.contextual_reward(reward_prob, context)
    
    def compute_gap(self, action):
        gap = np.absolute(
            self.reward_space.reward_parameters[self.reward_space.optimal_arm] - self.reward_space.reward_parameters[action]
        )
        return gap
    
    def step(self, action, context):
        
        valid_action = True
        if (action is None or action < 0 or action >= self.action_space.size):
            print("Algorithm chose an invalid action; reset reward to -inf", flush = True)
            reward = float("-inf")
            gap = float("inf")
            valid_action = False
            
        if valid_action:
            if self.reward_space.reward_dist in ["bernoulli" , "given"]:
                reward_var = (self.reward_space.reward_parameters[:].var())/(self.action_space.size) # A Sample Implementation
                
                reward_prob = self.reward_space.reward_parameters[action]+self.random.normal(0,reward_var)
                reward_prob = np.clip(reward_prob, a_min = 0, a_max =1)
                
                reward = self.compute_reward(reward_prob, context)

        return reward
    