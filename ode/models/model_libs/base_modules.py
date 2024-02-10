import torch as th
from torch import nn
import torchdiffeq as thd
import numpy as np
from typing import Dict


class GroupGCNConv(nn.Module):

    def __init__(self, input_feat, output_feat, dropout_rate, group=1):    
        super().__init__()
        self.input_feat = input_feat
        self.output_feat = output_feat
        self.group = group
        # self.neighbor_linear = nn.Linear(input_feat*group, output_feat*group, bias=False)
        # self.self_linear = nn.Linear(input_feat*group, output_feat*group, bias=False) 
        self.neighbor_linear = nn.Conv1d(input_feat*group, output_feat*group, 1, groups=group)
        self.self_linear = nn.Conv1d(input_feat*group, output_feat*group, 1, groups=group)
        self.dropout_layer = nn.Dropout(dropout_rate)
        # self.layer_norm = nn.LayerNorm(input_feat)

    def normalized_laplacian(self, A):
        '''
        A: adjacency matrix
        '''
        out_degree = th.sum(A, dim=1)
        int_degree = th.sum(A, dim=2)

        out_degree_sqrt_inv = th.where(out_degree!=0, th.pow(out_degree, -0.5), th.zeros_like(out_degree))
        int_degree_sqrt_inv = th.where(int_degree!=0, th.pow(int_degree, -0.5), th.zeros_like(int_degree))
        # mx_operator = np.eye(A.shape[0]) - np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)

        diag_out_degree_sqrt_inv = th.diag_embed(out_degree_sqrt_inv)
        diag_int_degree_sqrt_inv = th.diag_embed(int_degree_sqrt_inv)

        mx_operator = th.eye(A.size(1)).unsqueeze(0).to(A.device) - diag_out_degree_sqrt_inv @ A @ diag_int_degree_sqrt_inv

        return mx_operator
    
    def forward(self, x, adj):
        '''
        x: tensor of shape [batch_size, num_nodes, num_features] B * N * F or (mB*E) * N * F
        '''
        
        B, N, F = x.shape
        mB = int(B/self.group)
        # x = self.layer_norm(x)
        adj = self.normalized_laplacian(adj)
        neighbor = th.bmm(adj, x)
        reshaped_neighbor = neighbor.reshape(mB, self.group, N, F).permute(0, 2, 1, 3).reshape(mB, N, F*self.group)

        reshaped_x = x.reshape(mB, self.group, N, F).permute(0, 2, 1, 3).reshape(mB, N, F*self.group)

        x = nn.Softplus()(self.dropout_layer(self.self_linear(reshaped_x.permute(0, 2, 1)) + self.neighbor_linear(reshaped_neighbor.permute(0, 2, 1))))
        x = x.permute(0, 2, 1)
        # now x is the shape of mb * N * F*group
        x = x.reshape(mB, N, self.group, F).permute(0, 2, 1, 3).reshape(B, N, F)

        return x


class GCNConv(nn.Module):

    def __init__(self, input_feat, output_feat, dropout_rate):    
        super().__init__()
        self.input_feat = input_feat
        self.output_feat = output_feat
        self.neighbor_linear = nn.Linear(input_feat, output_feat, bias=False)
        self.self_linear = nn.Linear(input_feat, output_feat, bias=False) 
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_feat)

    def normalized_laplacian(self, A):
        '''
        A: adjacency matrix
        '''
        out_degree = th.sum(A, dim=1)
        int_degree = th.sum(A, dim=2)

        out_degree_sqrt_inv = th.where(out_degree!=0, th.pow(out_degree, -0.5), th.zeros_like(out_degree))
        int_degree_sqrt_inv = th.where(int_degree!=0, th.pow(int_degree, -0.5), th.zeros_like(int_degree))
        # mx_operator = np.eye(A.shape[0]) - np.diag(out_degree_sqrt_inv) @ A @ np.diag(int_degree_sqrt_inv)

        diag_out_degree_sqrt_inv = th.diag_embed(out_degree_sqrt_inv)
        diag_int_degree_sqrt_inv = th.diag_embed(int_degree_sqrt_inv)

        mx_operator = th.eye(A.size(1)).unsqueeze(0).to(A.device) - diag_out_degree_sqrt_inv @ A @ diag_int_degree_sqrt_inv

        return mx_operator
    
    def forward(self, x, adj):
        '''
        x: tensor of shape [batch_size, num_nodes, num_features] B * N * F
        '''
        # x = x.to(th.double)

        x = self.layer_norm(x)
        adj = self.normalized_laplacian(adj)
        neighbor = th.bmm(adj, x)
        x = nn.Softplus()(self.dropout_layer(self.self_linear(x) + self.neighbor_linear(neighbor)))
        # x = self.neighbor_linear(neighbor)
        # x = self.dropout_layer(x)
        # x = nn.Softplus()(x)
        return x
    
class MaskedAvgNodePooling(nn.Module):
    def __init__(self):

        super(MaskedAvgNodePooling, self).__init__()

    def forward(self, x, mask):

        """
        :param x:  B * N * D
        :param mask: B * N
        :return:
        """
        
        B, N, D = x.shape
        mask = mask.reshape(B, N, 1).repeat(1, 1, D)
        masked_x = th.mul(x, mask)
        pooled_x = masked_x.sum(1)/mask.sum(1)
        return pooled_x

class TrajEncoder(nn.Module):

    def __init__(self, specs):

        super().__init__()
        self.specs = specs
        self.encoder = self.create_sturcture(self.specs)
    
    def create_sturcture(self, specs):

        encoder = nn.Sequential()
        for i in range(self.specs['num_encoder_layers']):
            if i == 0:
                encoder.add_module(f'fc-{i}',
                                   nn.Linear(self.specs['input_dim'], self.specs['node_dim']))
            else:
                encoder.add_module(f'fc-{i}',
                                   nn.Linear(self.specs['node_dim'], self.specs['node_dim']))
            encoder.add_module('act', nn.Softplus())

        return encoder

    def forward(self, x):

        x = self.encoder(x)

        return x
    
class TrajDecoder(nn.Module):

    def __init__(self, specs):

        super().__init__()
        self.specs = specs
        self.decoder = self.create_sturcture(self.specs)

    def create_structure(self, specs: Dict):

        decoder = nn.Sequential()
        for i in range(specs['num_decoder_layers']):
            if i == specs['num_decoder_layers'] - 1:
                decoder.add_module(f'fc-{i}',
                                   nn.Linear(specs['node_dim'] + specs['extra_dim'], self.output_dim))
                decoder.add_module('flatten', nn.Flatten(start_dim=2))
            else:
                decoder.add_module(f'fc-{i}',
                                   nn.Linear(specs['node_dim'] + specs['extra_dim'], specs['node_dim'] + specs['extra_dim']))
                decoder.add_module('act', nn.Softplus())

        return decoder

    def forward(self, x):
            
        x = self.decoder(x)

        return x
