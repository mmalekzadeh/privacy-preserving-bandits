import numpy as np

###____________  Policy Interface ____________###
class Policy:
    def __init__(self, num_actions, num_contexts):
        self.num_actions = num_actions
        self.num_contexts = num_contexts
        self.total_rewards = np.zeros(num_actions, dtype = np.longdouble)
        self.total_counts = np.zeros(num_actions, dtype = np.longdouble)
    
    def act(self, context):
        pass
        
    def feedback(self, action, reward, context):
        pass

###____________  LinearUCB ____________###
class LinUCB(Policy):
    def __init__(self, num_actions, num_contexts):
        Policy.__init__(self, num_actions, num_contexts)
        self.name = "LinUCB"

        self.A = np.zeros((num_actions, num_contexts, num_contexts))
        for a in range (0, num_actions):
            self.A[a] = np.identity(num_contexts)
        self.b = np.zeros((num_actions, num_contexts))
        
    def act(self, context, alpha=1.):
        th_hat = np.zeros((self.num_actions, self.num_contexts))
        p = np.zeros(self.num_actions)
        next_action = None
        for a in range (self.num_actions):
            A_inv = np.linalg.inv(self.A[a])
            th_hat[a]  = A_inv.dot(self.b[a]) 
            ta = context.dot(A_inv).dot(context)
            a_upper_ci = alpha * np.sqrt(ta)
            a_mean = th_hat[a].dot(context)
            p[a] = a_mean + a_upper_ci
        
        p = p + ( np.random.random(len(p)) * 0.0000001 )
        next_action = np.argmax(p)
        
        return next_action
    
    def feedback(self, context, action, reward):
        self.A[action] += np.outer(context,context)
        self.b[action] += reward*context

        self.total_rewards[action] += reward
        self.total_counts[action] += 1

    def get_model(self):
        return self.A, self.b 