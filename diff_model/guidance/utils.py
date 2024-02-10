import torch as th
from torch import nn
import torchdiffeq as thd
import numpy as np
from typing import Dict

class NeuronalDynamics(nn.Module):
    def __init__(self, A, u=3.5, d=2):
        super(NeuronalDynamics, self).__init__()
        self.A = A
        self.u = u
        self.d = d
    
    def forward(self, t, x):
        if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
            f = -x + th.sparse.mm(self.A, 1 / (1 + th.exp(self.u - self.d*x)))
        else:
            f = -x + th.mm(self.A, 1/(1 + th.exp(self.u - self.d*x)))
        return f
    
def evolve(topology, mu, delta, x_0, t_eval, method):

    x_real = thd.odeint(NeuronalDynamics(topology, mu, delta), x_0, t_eval, method=method).squeeze(-1)
    x_real = x_real.transpose(0, 1)
    return x_real