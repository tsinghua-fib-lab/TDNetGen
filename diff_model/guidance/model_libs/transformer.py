import torch
import torch.nn as nn
import numpy as np
import math
from torch import nn, Tensor
from typing import Dict

class PositionalEncoder(nn.Module):
    def __init__(self, dropout=0.1,  max_seq_len: int=5000, d_model: int=512, batch_first: bool=True):
        super().__init__()
        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe.shape)
        x = x + self.pe.squeeze(1)[:x.size(self.x_dim)]

        return self.dropout(x)

class TemporalEncoder(nn.Module):

    def __init__(self, specs: Dict):
        super(TemporalEncoder, self).__init__()
        
        self.encoder_input_layer = nn.Linear(specs['trm_input_dim'], specs['hidden_dim'])

        self.positional_encoding_layer = PositionalEncoder(d_model=specs['hidden_dim'], dropout=0.1, max_seq_len=specs['max_seq_len'])
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=specs['hidden_dim'], nhead=specs['n_heads'], batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=specs['n_trm_layers'])
    
    def forward(self, x):

        '''
        x: (B, N, T) -> (batch_size, num_nodes, seq_len) -> (batch_size*num_nodes, seq_len, num_feat)
        '''

        
        x = x.unsqueeze(-1) # (B, N, T, 1)
        x = self.encoder_input_layer(x) # (B, N, T, hidden_dim)

        B, N, T, F = x.shape

        cls_token = torch.zeros(B, N, 1, F).to(x.device) # (B, N, 1, hidden_dim)

        x = torch.cat((cls_token, x), dim=2) # (B, N, T+1, hidden_dim)
        x = x.reshape(B * N, T+1, F) # (B*N, T+1, hidden_dim)
        x = self.positional_encoding_layer(x) # (B*N, T+1, hidden_dim)
        x = self.encoder(x) # (B*N, T+1, hidden_dim)
        cls_representation = x[:, 0, :] # (B*N, hidden_dim)
        cls_representation = cls_representation.reshape(B, N, F) #(B, N, hidden_dim)

        return cls_representation



