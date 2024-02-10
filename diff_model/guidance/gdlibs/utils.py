import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torchdiffeq as thd

class NeuronalDynamics(nn.Module):
    def __init__(self, A, u=3.5, d=2):
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


def evolve_and_resilience(topology, mu, delta, t_eval, method):

    x_lo_init = th.rand((topology.shape[0], 1))
    x_hi_init = th.rand((topology.shape[0], 1)) * 5
    
    x_real_lo = thd.odeint(NeuronalDynamics(topology, mu, delta), x_lo_init, t_eval, method=method).squeeze(-1)
    x_real_lo = x_real_lo.transpose(0, 1)
    x_real_hi = thd.odeint(NeuronalDynamics(topology, mu, delta), x_hi_init, t_eval, method=method).squeeze(-1)
    x_real_hi = x_real_hi.transpose(0, 1)

    mmx = x_real_lo.mean(dim=0)[-1]
    mmn = x_real_hi.mean(dim=0)[-1]

    if abs(mmx.item() - mmn.item()) > 3:
        resi = 0

    else:
        if mmn > 3.5:
            resi = 1
        else:
            resi = 0
    
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


def trans_bipartite_to_weighted_block(topology, top_node, node_mask, all_nodes_num):

    '''
    topology: th.Tensor (B, N, N)
    top_node: th.Tensor (B, 1)
    '''
    B, N, _ = topology.shape
    # all_nodes_num = topology.shape[0]

    e_mask_1 = node_mask.unsqueeze(-1)
    e_mask_2 = node_mask.unsqueeze(1)

    topology = topology * e_mask_1 * e_mask_2

    scaling1 = topology.sum(dim=1, keepdim=True)
    scaling2 = topology.sum(dim=2, keepdim=True)

    scaling1[scaling1 == 0] = 1
    scaling2[scaling2 == 0] = 1



    for b in range(B):
        scaling1[b, :, :top_node[b]] = 1
        scaling2[b, top_node[b]:, :] = 1

    seperate_bipartite1 = th.bmm(topology/scaling1, topology.transpose(1, 2))
    seperate_bipartite2 = th.bmm(topology, topology.transpose(1, 2)/scaling2)

    # seperate_bipartite1 = topology
    # seperate_bipartite2 = topology


    mask_1 = th.zeros_like(seperate_bipartite1).float().to(seperate_bipartite1.device)
    mask_2 = th.zeros_like(seperate_bipartite2).float().to(seperate_bipartite2.device)

    for b in range(B):
        mask_1[b, :top_node[b], :top_node[b]] = 1
        mask_2[b, top_node[b]:int(all_nodes_num[b].item()), top_node[b]:int(all_nodes_num[b].item())] = 1

    seperate_bipartite = seperate_bipartite1 * mask_1 + seperate_bipartite2 * mask_2
    diag_mask = th.ones_like(seperate_bipartite) - th.eye(N).unsqueeze(0).repeat(B, 1, 1).to(seperate_bipartite.device)
    seperate_bipartite = seperate_bipartite * diag_mask.float().to(seperate_bipartite.device)


    return seperate_bipartite


def modify_batch_matrix(matrix):
    # 检查每个矩阵的每一行是否全为零
    zero_rows = th.all(matrix == 0, dim=2)

    # 获取批处理大小和维度大小
    B, N, _ = matrix.shape

    # 构建一个单位矩阵并根据zero_rows的值进行扩展
    identity = th.eye(N).unsqueeze(0).expand(B, N, N).to(matrix.device)

    # 将全零行的对角线元素置为1
    matrix[zero_rows] = identity[zero_rows]

    return matrix

def create_weighted_networks(M):
    # M 是一个 M x N 的二部图矩阵
    a, b = M.shape

    # 初始化网络 A 和 B
    A = np.zeros((a, a))
    B = np.zeros((b, b))

    # 计算每个节点在 N 集合中的度（degree）
    degree_N = np.sum(M, axis=0)

    # 避免除以零
    degree_N[degree_N == 0] = 1
    # print(degree_N)
    # assert False

    # 计算网络 A 的权重
    for i in range(a):
        for j in range(a):
            if i != j:
                A[i, j] = np.sum(M[i, :] * M[j, :] / degree_N)

    # 计算每个节点在 M 集合中的度（degree）
    degree_M = np.sum(M, axis=1)

    # 避免除以零
    degree_M[degree_M == 0] = 1

    # 计算网络 B 的权重
    for i in range(b):
        for j in range(b):
            if i != j:
                B[i, j] = np.sum(M[:, i] * M[:, j] / degree_M)

    return A, B
