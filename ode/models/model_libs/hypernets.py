import sys
sys.path.append('/data2/liuchang/workspace/GraphGene/GDiff/src/guidance')
import torch as th
from torch import nn
import torchdiffeq as thd
import numpy as np
from typing import Dict
from ode.models.model_libs.odeblock import ODEBlock
# from src.guidance.model_libs.godeclasf import GodeClassifier
from gdlibs.utils import *
import copy

class HyperEnvNet(nn.Module):

    def __init__(self, specs, net_a, ghost_structure, hypernet, codes, logger, net_mask=None):
        super().__init__()
        self.specs = specs
        self.net_a = net_a
        self.env_emb = codes
        self.n_env = self.env_emb.size(0)
        self.hypernet = hypernet
        self.nets = {'ghost_structure': ghost_structure, "mask": net_mask}
        self.logger = logger
    
    def update_ghost(self):
        net_ghost = copy.deepcopy(self.nets['ghost_structure'])
        set_requires_grad(net_ghost, False)
        self.nets["ghost"] = net_ghost
        self.nets["ghost"].update_epsilon(self.nets['ghost_structure'].epsilon)
        self.nets["ghost"].ode_func.update_graph(self.nets['ghost_structure'].ode_func.X, self.nets['ghost_structure'].ode_func.E)
        if 'bipartite' in self.specs:
            if self.specs['bipartite']:
                self.nets["ghost"].ode_func.update_top_nodes(self.nets['ghost_structure'].ode_func.top_nodes)
        param_hyper = self.hypernet(self.env_emb)
        count_f = 0
        count_p = 0
        
        param_mask = self.nets['mask']

        for (name_a, pa), (name_g, pg) in zip(self.net_a.named_parameters(), self.nets['ghost'].named_parameters()):
            # print(name_a)
            # print(name_g)
            # print(pa.shape)
            # print(pg.shape)
            # assert False
            
            phypers = []

            if param_mask is None:
                pmask_sum = int(pa.numel())
            else:
                pmask = param_mask[count_f: count_f + pa.numel()].reshape(*pa.shape)
                pmask_sum = int(pmask.sum())
            
            if pmask_sum == int(pa.numel()):
                for e in range(self.n_env):
                    phypers.append(param_hyper[e, count_p: count_p + pmask_sum].reshape(*pa.shape))
            else:
                for e in range(self.n_env):
                    phyper = th.zeros(*pa.shape).to(self.net_a.device)
                    if pmask_sum != 0:
                        phyper[pmask == 1] = param_hyper[count_p:count_p + pmask_sum]
                    phypers.append(phyper)
            
            count_p += pmask_sum
            count_f += int(pa.numel())

            phyper = th.cat(phypers, dim=0)
            pa_new = th.cat([pa] * self.n_env)

            # print(phyper.shape)
            # print(pa_new.shape)
            # print(pg.shape)
            pg.copy_(pa_new + phyper)

    def forward(self, *input, **kwargs):

        return self.nets['ghost'](*input, **kwargs)
    
### If we only want to generalize the ODE module, select this model
### maybe only genelize ODE is enough, as small extra parameters as possible
class HyperGODE(nn.Module):
    
    def __init__(self, specs, env_emb_init=None, layers=[0], mask=None):

        super().__init__()

        self.specs = specs
        self.env_emb = nn.Parameter(0.*th.randn(specs['num_env'], specs['code_dim'])) if env_emb_init is None else env_emb_init
        self.model_specs = specs
        self.is_layer = specs['is_layer']
        
        self.net_root = ODEBlock(self.specs, hyper=True, root=True)

        n_param_tot = count_parameters(self.net_root)

        n_param_mask = n_param_tot if not self.is_layer else get_n_param_layer(self.net_root, layers)

        n_params_hypernet = n_param_mask

        # hypernet
        self.net_hyper = nn.Linear(specs['code_dim'], n_params_hypernet, bias=False)

        # Ghost
        self.ghost_structure = ODEBlock(self.model_specs, hyper=True, root=False)

        set_requires_grad(self.ghost_structure, tf=False)

        # Mask
        if self.is_layer and (mask is None):

            self.mask = {"mask": generate_mask(self.net_root, "layer", layers)}
        else:
            self.mask = {"mask": mask}
        
        self.net_leaf = HyperEnvNet(specs, self.net_root, self.ghost_structure, self.net_hyper, self.env_emb, None, self.mask['mask'])
    
    def update_epsilon(self, epsilon):

        self.net_root.update_epsilon(epsilon)
        self.ghost_structure.update_epsilon(epsilon)

    def update_ghost(self, X_in, E_in, top_nodes=None):
        
        self.net_root.ode_func.update_graph(X_in, E_in)

        self.ghost_structure.ode_func.update_graph(X_in, E_in)

        if 'bipartite' in self.specs:
            if self.specs['bipartite']:
                self.net_root.ode_func.update_top_nodes(top_nodes)
                self.ghost_structure.ode_func.update_top_nodes(top_nodes)

        self.net_leaf.update_ghost()

    def forward(self, t, x_real):
        return self.net_leaf(t, x_real)

### If we want to generalize all model parameters, select this model
# class HyperForecaster(nn.Module):
#     def __init__(self, specs, env_emb_init=None,layers=[0], mask=None):
#         super().__init__()

#         self.specs = specs
#         self.env_emb = nn.Parameter(0.*th.randn(specs['num_env'], specs['code_dim'])) if env_emb_init is None else env_emb_init
        
#         self.model_specs = specs
        
#         self.is_layer = specs['is_layer']

#         self.net_root = GodeClassifier(self.model_specs)

#         n_param_tot = count_parameters(self.net_root)

#         n_param_mask = n_param_tot if not self.is_layer else get_n_param_layer(self.net_root, layers)

#         n_params_hypernet = n_param_mask

#         # hypernet
#         self.net_hyper = nn.Linear(specs['code_dim'], n_params_hypernet)

#         # Ghost

#         self.ghost_structure = GodeClassifier(self.model_specs)

#         set_requires_grad(self.ghost_structure, tf=False)

#         # Mask
#         if self.is_layer and mask is None:

#             self.mask = {"mask": generate_mask(self.net_root, "layer", layers)}
#         else:
#             self.mask = {"mask": mask}
        
#         self.net_leaf = HyperEnvNet(specs, self.net_root, self.ghost_structure, self.net_hyper, self.env_emb, None, self.mask['mask'])
        
#     def update_ghost(self):

#         self.net_leaf.update_ghost()

#     def forward(self, x_lo, x_hi, X_in, E_in, t_in, t_eval, extra_data, node_mask):
        
#         return self.net_leaf(x_lo, x_hi, X_in, E_in, t_in, t_eval, extra_data, node_mask)


        
        
