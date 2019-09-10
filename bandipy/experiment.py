import numpy as np

###____________ Experiment  Interface ____________###
class Experiment():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.con_act_rew_hist = list()
        self.cumulative_reward = 0.0
        self.cumulative_regret = 0.0
    
    def run_bandit(self, max_number_of_trials=1000):
        pass

###____________  RandomExperiment ____________###
class RandomExperiment(Experiment):
    """
    Experiment on synthesised random datasets with simple MAB algorithms
    """
    def __init__(self, env, agent):
        Experiment.__init__(self, env, agent) 
        
    def run_bandit(self, max_number_of_trials=1000):
        
        for t in range(max_number_of_trials):
            context = self.env.get_context()
            contex_code = 0
            if self.env.context_space.context_type == "binary":
                contex_code = int("".join(str(x) for x in context), 2) 
            action = self.agent.act(context, contex_code)
            reward = self.env.step(action, context)       
            self.agent.feedback(action, reward, context)
            
            self.con_act_rew_hist.append([contex_code, action, reward])
            
            self.cumulative_reward += reward
            gap = self.env.compute_gap(action)
            if action != self.env.reward_space.optimal_arm:
                self.cumulative_regret += gap

        hist = np.array(self.con_act_rew_hist)
        return hist


###____________  ExperimentOnData ____________###
class ExperimentOnData(Experiment):
    """
    Experiments on prepared datasets with contextual MAB algorithms
    """
    
    def __init__(self, env, agent):
        Experiment.__init__(self, env, agent)
    
    def run_bandit(self, max_number_of_trials=1000, alph = 0.2, return_model = False):

        for t in range(max_number_of_trials):
            context = self.env.get_context(trial=t)
            action = self.agent.act(context)
            reward = self.env.step(action, context, trial=t)
            self.agent.feedback(context, action, reward)
            
            self.cumulative_reward += reward
            if reward == 0:
                self.cumulative_regret += 1
            self.con_act_rew_hist.append([context, action, reward])

        hist = np.array(self.con_act_rew_hist)
        return hist


###____________  ExperimentWithSampling ____________###
class ExperimentWithSampling(Experiment):
    """
    Experiments on prepared datasets with contextual MAB algorithms with data sharing
    """
    def __init__(self, env, agent):
        Experiment.__init__(self, env, agent)
        self.shared_data = list()
    
    def run_bandit(self, number_of_trials, alpha, sampling_prob=0.0, neg_rew_sam_rate = 0.0):
        data_tuple_1 = list()
        data_tuple_0 = list()
        for t in range(number_of_trials):
            context = self.env.get_context(trial=t)
            action = self.agent.act(context, alpha)
            reward = self.env.step(action, context, trial=t)
            self.agent.feedback(context, action, reward)
            self.con_act_rew_hist.append([context, action, reward])
            if reward == 1:
                data_tuple_1.append(np.concatenate((context, [action, reward])))
            else:
                data_tuple_0.append(np.concatenate((context, [action, reward])))
            
        # Sampling
        rnd_samp = np.random.rand()
        if rnd_samp < sampling_prob:
            rnd_neg = np.random.rand()
            if rnd_neg < neg_rew_sam_rate:
                if len(data_tuple_0) != 0:
                    shared_indx = np.random.randint(len(data_tuple_0))
                    self.shared_data = [data_tuple_0[shared_indx]]
            else:
                if len(data_tuple_1) != 0:
                    shared_indx = np.random.randint(len(data_tuple_1))
                    self.shared_data = [data_tuple_1[shared_indx]]
                elif len(data_tuple_0) != 0:
                    shared_indx = np.random.randint(len(data_tuple_0))
                    self.shared_data = [data_tuple_0[shared_indx]]
        else:
            self.shared_data = []
        
    
    def run_bandit_on_data(self, shared_contexts, shared_actions, shared_responses):
        for t in range(len(shared_contexts)):
            context = shared_contexts[t]
            action =  shared_actions[t].astype(int)
            reward =  shared_responses[t].astype(int)
            self.agent.feedback(context, action, reward)
            self.con_act_rew_hist.append([context, action, reward])


###____________  ExperimentWithSampling ____________###
class ExperimentWithSampling2(Experiment):
    """
    Experiments on prepared datasets with contextual MAB algorithms with data sharing
    """
    def __init__(self, env, agent):
        Experiment.__init__(self, env, agent)
        self.shared_data = list()
    
    def run_bandit(self, number_of_trials, alpha, sampling_prob=0.0, neg_rew_sam_rate = 0.0, private=True):
        data_tuple_1 = list()
        data_tuple_0 = list()
        for t in range(number_of_trials):
            context = self.env.get_context(trial=t)
            action = self.agent.act(context, alpha)
            reward = self.env.step(action, context, trial=t)
            self.agent.feedback(context, action, reward)
            self.con_act_rew_hist.append([context, action, reward])
            if reward == 1:
                data_tuple_1.append(np.concatenate((context, [action, reward])))
            else:
                data_tuple_0.append(np.concatenate((context, [action, reward])))
            
        # Sampling
        if private:
            rnd_samp = np.random.rand()
            if rnd_samp < sampling_prob:
                rnd_neg = np.random.rand()
                if rnd_neg < neg_rew_sam_rate:
                    if len(data_tuple_0) != 0:
                        shared_indx = np.random.randint(len(data_tuple_0))
                        self.shared_data = [data_tuple_0[shared_indx]]
                else:
                    if len(data_tuple_1) != 0:
                        shared_indx = np.random.randint(len(data_tuple_1))
                        self.shared_data = [data_tuple_1[shared_indx]]
                    elif len(data_tuple_0) != 0:
                        shared_indx = np.random.randint(len(data_tuple_0))
                        self.shared_data = [data_tuple_0[shared_indx]]
            else:
                self.shared_data = []
        else:
            if len(data_tuple_1) != 0:
                self.shared_data = data_tuple_1
            elif len(data_tuple_0) != 0:
                shared_indx = (int)(len(data_tuple_0)/(len(data_tuple_0)*neg_rew_sam_rate))
                if len(self.shared_data) != 0:
                    self.shared_data = np.append(self.shared_data, data_tuple_0[::shared_indx], axis =0)
                else:
                    self.shared_data = data_tuple_0[::shared_indx]
        
    
    def run_bandit_on_data(self, shared_contexts, shared_actions, shared_responses):
        for t in range(len(shared_contexts)):
            context = shared_contexts[t]
            action =  shared_actions[t].astype(int)
            reward =  shared_responses[t].astype(int)
            self.agent.feedback(context, action, reward)
            self.con_act_rew_hist.append([context, action, reward])







###____________  ExperimentOnPrivateData ____________###
class ExperimentOnPrivateData(Experiment):
    """
    Experiments on  prepared datasets with contextual MAB algorithms
    """
    
    def __init__(self, env, agent):
        Experiment.__init__(self, env, agent)
        self.shared_data = list()
    
    def run_bandit(self, max_number_of_trials=1000,
                        alph = 0.2, epsilon = 1., fed_rate = 10,
                        return_model = False):
        
        fed = 0
        for t in range(max_number_of_trials):
            context = self.env.get_context(trial=t)
            action = self.agent.act(context)
            reward = self.env.step(action, context, trial=t)
            self.agent.feedback(context, action, reward)

            ### Privacy ###
            if reward == 1 and fed%fed_rate==0:
                #noise =  np.random.laplace(0, 1./((t+1)*epsilon), 1)
                #context += noise
                #context = np.clip(context, 0, 1)
                self.shared_data.append(np.concatenate((context, [action])))
                fed += 1
            elif reward == 1:
                fed += 1
            ### Privacy ###
            
            self.cumulative_reward += reward
            if reward == 0:
                self.cumulative_regret += 1
            self.con_act_rew_hist.append([context, action, reward])

        hist = np.array(self.con_act_rew_hist)
        return hist

