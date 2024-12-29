import numpy as np
from typing import List
from abc import ABC, abstractmethod

# Adaptive sampling method w.r.t customized distribution function
class BasicDistribution(ABC):
    def __init__(self, dim : int, x_range:List[float], v_range:List[float]):
        self.dim = dim
        self.x_range = x_range
        self.v_range = v_range

    def get_target_prob(self, x:float, v:float):
        pass

    def get_proposal_prob(self, x:float, v:float):
        pass

    def get_sample(self, *args):
        pass

    def rejection_sampling(self, n_samples:int):

        x = np.random.uniform(self.x_range[0], self.x_range[1], n_samples)
        v = np.random.uniform(self.v_range[0], self.v_range[1], n_samples)

        X = np.linspace(self.x_range[0], self.x_range[1], 256)
        V = np.linspace(self.v_range[0], self.v_range[1], 256)

        x_mesh, v_mesh = np.meshgrid(X,V)
        max_ratio = max(self.get_target_prob(x, v) / self.get_proposal_prob(x, v) for x, v in zip(x_mesh, v_mesh))

        samples = []
        i = 0
        
        while len(samples) < n_samples:
            u = np.random.uniform(0, max_ratio * self.get_proposal_prob(x[i], v[i]))
            
            if u < self.get_target_prob(x[i], v[i]):
                samples.append([x,v])
                
            i += 1

        return samples

class BumpOnTailDistribution(BasicDistribution):
    def __init__(self, dim : int, x_range:List[float], v_range:List[float], eta:float, a:float, v0:float, sigma:float):
        self.dim = dim
        self.x_range = x_range
        self.v_range = v_range

        # parameters
        self.eta = eta
        self.a = a
        self.v0 = v0
        self.sigma = sigma

    def get_prob(self, x:np.array, v:np.array):
        prob = 1 / np.sqrt(2 * np.pi) / self.eta * np.exp(-0.5 / self.sigma**2 * x**2) * (
            1 / (1 + self.a) * 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * v**2)
            + self.a / (1 + self.a) * 1 / np.sqrt(2 * np.pi) / self.sigma * np.exp(-0.5 * (v-self.v0)**2 / self.sigma ** 2)
        )
        return prob
