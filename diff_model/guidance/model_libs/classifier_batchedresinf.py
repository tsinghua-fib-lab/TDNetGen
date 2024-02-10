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

    

    def normalized_lap(self, adj):
        # 确保输入是torch.Tensor
        adj = th.Tensor(adj) if not isinstance(adj, th.Tensor) else adj

        # 计算出度和入度（沿最后两个维度）
        out_degree = adj.sum(dim=-1)
        in_degree = adj.sum(dim=-2)

        # 计算度的逆平方根，并避免除以0
        out_degree_sqrt_inv = th.pow(out_degree, -0.5)
        out_degree_sqrt_inv[out_degree == 0] = 0

        in_degree_sqrt_inv = th.pow(in_degree, -0.5)
        in_degree_sqrt_inv[in_degree == 0] = 0

        # 使用广播机制创建对角矩阵并进行矩阵乘法
        # 注意：torch.diag_embed需要更高版本的PyTorch
        B, N, _ = adj.shape
        out_degree_mat = th.diag_embed(out_degree_sqrt_inv)
        in_degree_mat = th.diag_embed(in_degree_sqrt_inv)

        # 对每个批次执行矩阵乘法
        mx_operator = th.bmm(th.bmm(out_degree_mat, adj), in_degree_mat)

        return mx_operator



    def forward(self, adj_matrix, x):

        '''
        adj_matrix: (B, N, N); x: (B, N, C, F)
        '''

        adj_matrix = self.normalized_lap(adj_matrix)

        neighbor = th.einsum('bnn, bncf->bncf', adj_matrix, x)

        # neighbor = th.matmul(adj_matrix, x)

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

        for i in range(self.gcn_layers-1):
            self.gcns.append(GCNConv(self.gcn_emb_size, self.gcn_emb_size))


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
        
        x = x.reshape(-1, feat_len)
        x = x.unsqueeze(-1)
        src = self.encoder_input_layer(x)

        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src)

        embeddings = src[:, -1, :]

        embeddings = embeddings.reshape(batch_size, nodes_num, num_traj, self.dim_val)

        for i in range(self.gcn_layers):

            embeddings = self.gcns[i](adj, embeddings) # (B, N, C, F')

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

        
        





