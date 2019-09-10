import numpy as np

class PrefPriors():
    def __init__(self, rand_gen=np.random):
        self.random = rand_gen
    
    def random_prior(self, size):
        pr_mat = self.random.rand(size[0], size[1])
        pr_mat = pr_mat/(pr_mat.sum(axis=1)[np.newaxis].T)
        return pr_mat
    
    def dirichlet_prior(self, size, p_alpha=None):
        if p_alpha is None:
            p_alpha = np.ones(size[1])
        pr_mat = self.random.dirichlet(alpha = p_alpha, size=size[0])
        return pr_mat


class ContextPriors():
    def __init__(self, rand_gen=np.random):
        self.random = rand_gen

    def binomial_prior(self, probs):
        return self.random.binomial(1, probs)