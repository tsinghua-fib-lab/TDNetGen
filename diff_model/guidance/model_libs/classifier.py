import torch as th
from torch import nn
import torchdiffeq as thd
import numpy as np
from typing import Dict
from diff_model.guidance.model_libs.transformer import TemporalEncoder
from diff_model.guidance.model_libs.base_modules import GCNConv, MaskedAvgNodePooling

class ResilienceClassifier(nn.Module):

    def __init__(self, specs:Dict):

        super(ResilienceClassifier, self).__init__()
        self.specs = specs
        self.is_map_time = specs['is_map_time']

        self.is_trm = specs['is_trm']

        if self.is_trm:
            self.temporal_encoder = TemporalEncoder(specs)

        self.gnn = nn.ModuleList()
        for i in range(specs['num_clasf_gnn_layers']):
            if i == 0:
                if self.is_trm:
                    self.gnn.append(GCNConv(specs['hidden_dim']*2, specs['hidden_dim']*2, specs['dropout_rate']))
                else:
                    self.gnn.append(GCNConv(specs['num_traj'] * specs['classification_steps'], specs['node_dim'], specs['dropout_rate']))
            else:
                if self.is_trm:
                    self.gnn.append(GCNConv(specs['hidden_dim']*2, specs['hidden_dim']*2, specs['dropout_rate']))
                    self.gnn.append(GCNConv(specs['node_dim'], specs['node_dim'], specs['dropout_rate']))

        self.node_pooling = MaskedAvgNodePooling() # (B, F)
        
        self.mlp = nn.Sequential()

        for i in range(specs['num_clasf_fc_layers']):
            if i == specs['num_clasf_fc_layers'] - 1:
                self.mlp.add_module(f'fc-{i}', nn.Linear(specs['node_dim'], specs['clasf_out_dim']))
                self.mlp.add_module('sigmoid', nn.Sigmoid())
            elif i == 0:
                if self.is_map_time:
                    self.mlp.add_module(f'fc-{i}', nn.Linear(specs['node_dim'] + specs['node_dim'], specs['node_dim']))
                else:
                    self.mlp.add_module(f'fc-{i}', nn.Linear(specs['node_dim'] + 1, specs['node_dim']))
                self.mlp.add_module('relu', nn.ReLU())
            else:
                self.mlp.add_module(f'fc-{i}', nn.Linear(specs['node_dim'], specs['node_dim']))
                self.mlp.add_module('relu', nn.ReLU())
        
        if self.is_map_time:
            self.time_mlp = nn.Sequential()
            self.time_mlp.add_module('linear', nn.Linear(1, specs['node_dim']))

    
    def forward(self, x, adj, mask,t):
        

        '''
        x : (B, 2N, T) if is_trm else (B, N, F)
        '''
        
        if self.is_trm:
            B, N_all, T = x.shape[0], x.shape[1], x.shape[2]
            x = self.temporal_encoder(x)
            x1 = x[:, :int(N_all/2), :]
            x2 = x[:, int(N_all/2):, :]
            x = th.cat([x1, x2], dim=2) # (B, N, 2F)

        
        # e_mask_1 = mask.unsqueeze(-1).float()
        # e_mask_2 = mask.unsqueeze(1).float()

        # adj = adj * e_mask_1 * e_mask_2

        # x = x * mask.unsqueeze(-1).float()


        for i in range(self.specs['num_clasf_gnn_layers']):
            x = self.gnn[i](x, adj) # (B, N, F)

        pooled_output = self.node_pooling(x, mask) # (B, F)

        if self.is_map_time:
            t = self.time_mlp(t)

        clasf_input = th.concat([pooled_output, t], dim=1) # (B, F+T)

        output = self.mlp(clasf_input) # (B, 1)

        return output