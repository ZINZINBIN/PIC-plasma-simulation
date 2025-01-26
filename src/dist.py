import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod

# Adaptive sampling method w.r.t customized distribution function
class BasicDistribution(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def get_target_prob(self, *args, **kwargs):
        pass

    def get_proposal_prob(self, *args, **kwargs):
        pass

    def get_sample(self, *args, **kwargs):
        pass

    def rejection_sampling(self, n_samples:int):
        pass

class BumpOnTail1D(BasicDistribution):
    def __init__(
        self,
        eta: float = 10.0,
        a: float = 0.3,
        v0: float = 4.0,
        sigma: float = 0.5,
        beta: float = 5.95,
        n_samples: int = 40000,
    ):
        # parameters
        self.eta = eta
        self.a = a
        self.v0 = v0
        self.sigma = sigma
        self.beta = beta

        self.initialize(n_samples)

    def initialize(self, n_samples:int):
        state = self.rejection_sampling(n_samples)
        self.x_init = state[:, 0]
        self.v_init = state[:, 1]
        
    def get_sample(self):
        return self.x_init, self.v_init

    def update_params(self, params:Dict):
        for key in params.keys():
            if hasattr(self, key) is True:
                setattr(self, key, params[key])

    def get_proposal_prob(self, x:float, v:float):
        prob = np.exp(-abs(x)) * np.exp(-abs(v))
        return prob

    def get_target_prob(self, x:float, v:float):
        prob = 1 / np.sqrt(2 * np.pi) / self.eta * np.exp(-0.5 / self.sigma**2 * x**2) * (
            1 / (1 + self.a) * 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * v**2)
            + self.a / (1 + self.a) * 1 / np.sqrt(2 * np.pi) / self.sigma * np.exp(-0.5 * (v-self.v0)**2 / self.sigma ** 2)
        )
        return prob

    def rejection_sampling(self, n_samples: int, batch:int = 1000):
        pos = []
        vel = []

        while len(pos) < n_samples:

            x = np.random.uniform(-10, 10, size = batch)
            v = np.random.uniform(-10, 10, size = batch)
            u = np.random.uniform(0, 1.0, size = batch)

            pos += x[u < self.get_target_prob(x, v)].tolist()
            vel += v[u < self.get_target_prob(x, v)].tolist()

        pos = np.array(pos[:n_samples])
        vel = np.array(vel[:n_samples])
        
        samples = np.zeros((n_samples,2))
        samples[:,0] = pos
        samples[:,1] = vel
        
        return samples

    def compute_E_field(self, x):
        return self.beta ** 2 * x

    def compute_trajectory(self, t:float):
        x = self.v_init * np.sin(self.beta * t) / self.beta + self.x_init * np.cos(self.beta * t)
        v = (-1) * self.x_init * np.sin(self.beta * t) * self.beta + self.v_init * np.cos(self.beta * t)
        return x,v
