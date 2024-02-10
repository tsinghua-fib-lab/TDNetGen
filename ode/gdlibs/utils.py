import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torchdiffeq as thd

class NeuronalDynamics(nn.Module):
    def __init__(self, A, u=3.5, d=2, beta_c=None):
        super(NeuronalDynamics, self).__init__()
        self.A = A
        self.u = u
        self.d = d

    def forward(self, t, x):
        if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
            f = -x + th.sparse.mm(self.A, 1 / (1 + th.exp(self.u - self.d * x)))
        else:
            f = -x + th.mm(self.A, 1 / (1 + th.exp(self.u - self.d * x)))
        return f
    
class MutualDynamics(nn.Module):
    #  dx/dt = b +
    def __init__(self, A, B=0.1, K=5., C=1., D=5., E=0.9, H=0.1, beta_c=None):
        super(MutualDynamics, self).__init__()
        self.A = A   # Adjacency matrix, symmetric
        self.b = B
        self.k = K
        self.c = C
        self.d = D
        self.e = E
        self.h = H

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = bi + xi(1-xi/ki)(xi/ci-1) + \sum_{j=1}^{N}Aij *xi *xj/(di +ei*xi + hi*xj)
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        n, d = x.shape
        f = self.b + x * (1 - x/self.k) * (x/self.c - 1)
        if d == 1:
            # one 1 dim can be computed by matrix form
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                outer = th.sparse.mm(self.A,
                                        th.mm(x, x.t()) / (self.d + (self.e * x).repeat(1, n) + (self.h * x.t()).repeat(n, 1)))
            else:
                outer = th.mm(self.A,
                                    th.mm(x, x.t()) / (
                                                self.d + (self.e * x).repeat(1, n) + (self.h * x.t()).repeat(n, 1)))
            f += th.diag(outer).view(-1, 1)
        else:
            # high dim feature, slow iteration
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                vindex = self.A._indices().t()
                for k in range(self.A._values().__len__()):
                    i = vindex[k, 0]
                    j = vindex[k, 1]
                    aij = self.A._values()[k]
                    f[i] += aij * (x[i] * x[j]) / (self.d + self.e * x[i] + self.h * x[j])
            else:
                vindex = self.A.nonzero()
                for index in vindex:
                    i = index[0]
                    j = index[1]
                    f[i] += self.A[i, j]*(x[i] * x[j]) / (self.d + self.e * x[i] + self.h * x[j])
        return f


    
def calc_beta(G):
    '''
    Input G: networkx Graph
    output beta: scalar
    '''
    # A = G.to_numpy_array()
    A = nx.to_numpy_array(G)
    # A = np.array(A)
    # print(A.shape)
    denominator = np.sum(np.sum(A))
    if denominator == 0:
        return 0
    else:
        molecular = np.sum(np.sum(np.dot(A, A)))
        beta = molecular/denominator
    return beta


def evolve_and_resilience(topology, param, t_eval, method, dynamics='mutual'):

    x_lo_init = th.rand((topology.shape[0], 1))
    if dynamics == 'mutual':
        x_lo_init = x_lo_init * 0.1
    x_hi_init = th.rand((topology.shape[0], 1)) * 5
    
    if dynamics == 'neuronal':
        x_real_lo = thd.odeint(NeuronalDynamics(topology, **param), x_lo_init, t_eval, method=method).squeeze(-1)
        x_real_lo = x_real_lo.transpose(0, 1)
        x_real_hi = thd.odeint(NeuronalDynamics(topology, **param), x_hi_init, t_eval, method=method).squeeze(-1)
        x_real_hi = x_real_hi.transpose(0, 1)

    elif dynamics == 'mutual':
        x_real_lo = thd.odeint(MutualDynamics(topology.to_sparse(), **param), x_lo_init, t_eval, method=method).squeeze(-1)
        x_real_lo = x_real_lo.transpose(0, 1)
        x_real_hi = thd.odeint(MutualDynamics(topology.to_sparse(), **param), x_hi_init, t_eval, method=method).squeeze(-1)
        x_real_hi = x_real_hi.transpose(0, 1)

    mmx = x_real_lo.mean(dim=0)[-1]
    mmn = x_real_hi.mean(dim=0)[-1]

    if dynamics == 'mutual':
        if abs(mmx.item() - mmn.item()) > 4.5:
            resi = 0
        else:
            resi = 1

    elif dynamics == 'neuronal':

        if abs(mmx.item() - mmn.item()) > 3:
            resi = 0

        else:
            if mmn > 3.5:
                resi = 1
            else:
                resi = 0
    else:
        raise NotImplementedError
    
    return x_real_lo, x_real_hi, resi

def count_parameters(model, mode='ind'):

    if mode == 'ind':

        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    else:

        raise NotImplementedError
    
def get_n_param_layer(net, layers):
    
    n_param = 0

    for name, p in net.named_parameters():
        
        if any(f"net.{layer}" in name for layer in layers):
            n_param += p.numel()
        
    return n_param

def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf

def generate_mask(net_a, mask_type='layer', layers=[0]):

    n_params_tot = count_parameters(net_a)

    if mask_type == "layer":
        mask_w = th.zeros(n_params_tot)
        count = 0
        for name, pa in net_a.named_parameters():
            if any(f"net.{layer}" in name for layer in layers):
                mask_w[count: count + pa.numel()] = 1.
            count += pa.numel()
    elif mask_type == "full":
        mask_w = th.ones(n_params_tot)
    else:
        raise Exception(f"Unknown mask {mask_type}")
    
    return mask_w


def trans_bipartite_to_weighted_block(topology, top_node):

    '''
    topology: th.Tensor (B, N, N)
    top_node: th.Tensor (B, 1)
    '''
    B, N, _ = topology.shape
    # all_nodes_num = topology.shape[0]

    scaling1 = topology.sum(dim=1, keepdim=True)
    scaling2 = topology.sum(dim=2, keepdim=True)

    scaling1[scaling1 == 0] = 1
    scaling2[scaling2 == 0] = 1

    for b in range(B):
        scaling1[b, :, :top_node[b]] = 1
        scaling2[b, top_node[b]:, :] = 1

    seperate_bipartite1 = th.bmm(topology/scaling1, topology.transpose(1, 2))
    seperate_bipartite2 = th.bmm(topology, topology.transpose(1, 2)/scaling2)

    mask_1 = th.zeros_like(seperate_bipartite1).float().to(seperate_bipartite1.device)
    mask_2 = th.zeros_like(seperate_bipartite2).float().to(seperate_bipartite2.device)

    for b in range(B):
        mask_1[b, :top_node[b], :top_node[b]] = 1
        mask_2[b, top_node[b]:, top_node[b]:] = 1

    seperate_bipartite = seperate_bipartite1 * mask_1 + seperate_bipartite2 * mask_2
    diag_mask = th.ones_like(seperate_bipartite) - th.eye(N).unsqueeze(0).repeat(B, 1, 1).to(seperate_bipartite.device)
    seperate_bipartite = seperate_bipartite * diag_mask.float().to(seperate_bipartite.device)

    return seperate_bipartite

