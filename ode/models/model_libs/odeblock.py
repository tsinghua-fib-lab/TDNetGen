# import sys
# sys.path.append('/data2/liuchang/workspace/GraphGene/GDiff')
import torch as th
from torch import nn
import torchdiffeq as thd
import numpy as np
from typing import Dict
from ode.models.model_libs.base_modules import GCNConv, GroupGCNConv
from ode.gdlibs.utils import *

class GroupODEFunc(nn.Module):

    def __init__(self, config: Dict, root=False):
        super().__init__()
        self.config = config
        self.root = root
        self.gnn_layers = self.create_gnn()

    def create_gnn(self) -> nn.ModuleList:
        if 'num_env' in self.config.keys() and not self.root:
            group = self.config['num_env']
        else:
            group = 1
        node_dim = self.config['node_dim'] + self.config['extra_dim']
        num_gnn_layers = self.config['num_gnn_layers']
        dropout_rate = self.config['dropout_rate']
        layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            gnn_layer = GroupGCNConv(node_dim, node_dim, dropout_rate, group=group)
            layers.append(gnn_layer)
        return layers
    
    
    def update_graph(self, X: th.tensor, E: th.tensor) -> None:
        self.X = X
        self.E = E

    def update_top_nodes(self, top_nodes) -> None:

        self.top_nodes = top_nodes

    def forward(self, t, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        B, N = self.E.shape[0], self.E.shape[1]
        
        A = self.E[..., 1:].sum(dim=-1).float()

        # th.save(A, 'before_A.pt')
        # th.save(self.top_nodes, 'before_top_nodes.pt')

        if 'bipartite' in self.config:
            if self.config.bipartite:
                A = trans_bipartite_to_weighted_block(A, self.top_nodes)
                # th.save(A, 'after_A.pt')
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, A)
    

        return x

class ODEFunc(nn.Module):  # A kind of ODECell in the view of RNN
    def __init__(self, config: Dict):
        super(ODEFunc, self).__init__()
        self.config = config
        self.gnn_layers = self.create_gnn()
        self.self_module = self.create_self_module() ## will be discard

    def create_gnn(self) -> nn.ModuleList:
        
        node_dim = self.config['node_dim'] + self.config['extra_dim']
        num_gnn_layers = self.config['num_gnn_layers']
        dropout_rate = self.config['dropout_rate']
        layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            gnn_layer = GCNConv(node_dim, node_dim, dropout_rate)
            layers.append(gnn_layer)
        return layers
    
    def create_self_module(self) -> nn.Sequential:
        hidden_dim = self.config['node_dim'] + self.config['extra_dim']
        self_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()
        )
        return self_module
    
    def update_graph(self, X: th.tensor, E: th.tensor) -> None:
        self.X = X
        self.E = E

    def forward(self, t, x):  # How to use t?
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        B, N = self.E.shape[0], self.E.shape[1]
        
        A = self.E[..., 1:].sum(dim=-1).float()
        x_self = x
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, A)
        
        # x = x - self.self_module(x_self)

        return x

class ODEBlock(nn.Module):
    def __init__(self,
                 specs: Dict, hyper=False, root=True):
        super().__init__()
        self.specs = specs
        self.hyper = hyper
        self.root = root
        if self.hyper:
            self.ode_func = GroupODEFunc(specs, self.root)
        else:
            self.ode_func = ODEFunc(specs)
        self.method = specs['th_ode_method']

        self.epsilon = specs['init_epsilon']
    
    def update_epsilon(self, epsilon):

        self.epsilon = epsilon

    def forward(self, t, x_real):
        
        '''
        # x: initial values of nodes, (B, N, 1)
        x_real: mappings of nodes, (B, N, T, F) but only use (B, N, F) each time
        '''

        if self.epsilon < 1e-3:
            epsilon = 0
        
        else:
            epsilon = self.epsilon
        
        if epsilon == 0:
            res = thd.odeint_adjoint(self.ode_func, x_real[:, :, 0, :], t, method=self.method)
        
        else:

            eval_points = np.random.random(len(t)) < epsilon
            eval_points[-1] = False
            eval_points = eval_points[1:]
            start_i, end_i = 0, None
            res = []

            for i, eval_point in enumerate(eval_points):
                if eval_point is True:
                    end_i = i + 1
                    t_seg = t[start_i:end_i + 1]
                    res_seg = thd.odeint_adjoint(self.ode_func, y0=x_real[:,:,start_i, :],\
                                                  t=t_seg, method=self.method)
                    if len(res) == 0:
                        res.append(res_seg)
                    else:
                        res.append(res_seg[1:])
                    start_i = end_i
            t_seg = t[start_i:]

            res_seg = thd.odeint_adjoint(self.ode_func, y0=x_real[:,:,start_i, :], \
                                         t=t_seg, method=self.method)
            if len(res) == 0:
                res.append(res_seg)
            else:
                res.append(res_seg[1:])
            
            res = th.cat(res, dim=0)

        return res.permute(1, 2, 0, 3)