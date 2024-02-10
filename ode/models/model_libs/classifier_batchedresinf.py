import torch as th
from torch import nn
import math
import torchdiffeq as thd
import numpy 
from typing import Dict
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    def __init__(self, dropout=0.1,  max_seq_len: int=5000, d_model: int=512, batch_first: bool=True):
        super().__init__()
        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        position = th.arange(max_seq_len).unsqueeze(1)

        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = th.zeros(max_seq_len, d_model)

        pe[:, 0::2] = th.sin(position * div_term)
        
        pe[:, 1::2] = th.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe.shape)
        x = x + self.pe.squeeze(1)[:x.size(self.x_dim)]

        return self.dropout(x)
    
    

class GCNConv(nn.Module):
    
    def __init__(self, input_features, output_features, is_self=True):
        super().__init__()

        self.input_features = input_features

        self.output_features = output_features

        self.linear = nn.Linear(input_features, output_features)

        self.is_self = is_self

        if is_self:
            self.s_linear = nn.Linear(input_features, output_features)

    

    def normalized_lap(self, A):
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



    def forward(self, adj_matrix, x):

        '''
        adj_matrix: (B, N, N); x: (B, N, C, F)
        '''

        adj_matrix = self.normalized_lap(adj_matrix)

        adj_matrix_expanded = adj_matrix.unsqueeze(1).expand(-1, x.size(2), -1, -1) # (B, C, N, N)

        x_input = x.permute(0, 2, 1, 3) # (B, C, N, F)

        neighbor = th.matmul(adj_matrix_expanded, x_input) # (B, C, N, F)

        neighbor = neighbor.permute(0, 2, 1, 3) # (B, N, C, F)

        if self.is_self:

            x = nn.Tanh()(self.s_linear(x)) + nn.Tanh()(self.linear(neighbor))

        else:
            x = nn.Tanh()(self.linear(neighbor))
    
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



class ResilienceClassifier(nn.Module):

    def __init__(self, specs:Dict):

        super(ResilienceClassifier, self).__init__()

        self.specs = specs

        self.dim_val = specs['trans_emb_size']

        self.input_plane = specs['input_plane']

        self.input_size = specs['input_size']

        self.with_gnn = specs['with_gnn']

        self.with_trm = specs['with_trm']

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if self.input_plane > 1:

            self.sharedMLP = nn.Sequential(
                nn.Conv2d(self.input_plane, self.input_plane // 2, 1 ,bias=False), nn.ReLU(),
                nn.Conv2d(self.input_plane // 2, self.input_plane, 1, bias=False)
            )

        self.encoder_input_layer = nn.Linear(self.input_size, self.dim_val)
        self.gcn_layers = specs['gcn_layers']
        self.trans_layers = specs['trans_layers']
        self.pool_type = specs['pool_type']
        self.n_heads = specs['n_heads']
        self.dropout_pos_enc = 0.1
        self.seq_len = specs['seq_len']
        self.gcn_emb_size = specs['gcn_emb_size']
        self.hidden_layers_num = specs['hidden_layers']
        self.positional_encoding_layer = PositionalEncoder(d_model=self.dim_val, dropout=0.1, max_seq_len=self.seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.dim_val, nhead=self.n_heads, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.trans_layers, norm=None)
        self.gcns = nn.ModuleList()
        self.gcns.append(GCNConv(self.dim_val, self.gcn_emb_size))

        if not self.with_trm:
            self.wo_trm_mlp = nn.Linear(1, self.dim_val)

        for i in range(self.gcn_layers-1):
            self.gcns.append(GCNConv(self.gcn_emb_size, self.gcn_emb_size))

        if not self.with_gnn:
            self.wo_gnn_mlp = nn.Linear(self.dim_val, self.gcn_emb_size)

        self.node_pooling = MaskedAvgNodePooling() # (B, F)

        self.hidden_layers = nn.ModuleList()

        for i in range(self.hidden_layers_num - 1):

            self.hidden_layers.append(nn.Linear(self.gcn_emb_size, self.gcn_emb_size))
            
            self.hidden_layers.append(nn.Tanh())

        self.resi_net_Linear = nn.Linear(self.gcn_emb_size, self.seq_len)

        if self.hidden_layers_num == 0:
            self.resi_net_down = nn.Linear(self.gcn_emb_size, 1, bias=True)
        else:
            self.resi_net_down = nn.Linear(self.seq_len, 1, bias=True)
        
        self.pred_linear = nn.Linear(1, 1, bias=True)

    def forward(self, x, adj, mask):

        '''
        # x : (B, 2N, T) if is_trm else (B, N, F)
        x: (B, N, C, T)
        '''

        # B, N_all, T = x.shape[0], x.shape[1], x.shape[2]

        # x = x.unsqueeze(-1)

        batch_size, nodes_num, num_traj, feat_len = x.shape
        
        x = x.reshape(-1, feat_len) # (-1, T)
        x = x.unsqueeze(-1) # (-1, T, 1)
        if self.with_trm:
            src = self.encoder_input_layer(x)

            src = self.positional_encoding_layer(src)

            src = self.encoder(src=src) # (-1, T, F)
        else:
            
            src = self.wo_trm_mlp(x) # (-1, T, F)

        embeddings = src[:, -1, :]

        embeddings = embeddings.reshape(batch_size, nodes_num, num_traj, self.dim_val)

        if self.with_gnn:

            for i in range(self.gcn_layers):

                embeddings = self.gcns[i](adj, embeddings) # (B, N, C, F')
        else:
            
            embeddings = self.wo_gnn_mlp(embeddings) # (B, N, C, F')


        embeddings = embeddings.permute(0, 2, 1, 3) # (B, C, N, F')

        if self.input_plane > 1:
            
            avgout = self.sharedMLP(self.avg_pool(embeddings)) # (B, C, 1, 1)
            maxout = self.sharedMLP(self.max_pool(embeddings)) # (B, C, 1, 1)

            channel_attention = nn.Sigmoid()((avgout + maxout)) # (B, C, 1, 1)

            all_node_emb = (embeddings * channel_attention).sum(dim=1) # (B, N, F')

        else:
            all_node_emb = embeddings.mean(dim=1)
        
        # res_emb = all_node_emb.mean(dim=1) # (B, F')
        res_emb = self.node_pooling(all_node_emb, mask) # (B, F')

        true_emb = res_emb.clone()

        if self.hidden_layers_num > 0:
            if self.hidden_layers_num > 1:
                for layer in self.hidden_layers:
                    res_emb = layer(res_emb)
            res_emb = nn.Tanh()(self.resi_net_Linear(res_emb))
        between_down = nn.Tanh()(self.resi_net_down(res_emb))
        resilience = nn.Sigmoid()(self.pred_linear(between_down))

        return resilience

        
        





