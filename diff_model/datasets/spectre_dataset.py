import sys
sys.path.append("..")
sys.path.append('../..')
import os
import pickle

import torch
from torch.utils.data import random_split, Dataset
import torch_geometric.utils
import networkx as nx
import numpy as np

# from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


def calc_beta(A):

    '''
    Input A : adjacency matrix
    output beta: scalar
    '''

    if type(A) is nx.classes.graph.Graph:
        A = nx.to_numpy_array(A)
    A = np.array(A)
    # print(A.shape)
    denominator = np.sum(np.sum(A))
    if denominator == 0:
        return 0
    else:
        molecular = np.sum(np.sum(np.dot(A, A)))
        beta = molecular/denominator
    return beta

class SpectreGraphDataset(Dataset):
    def __init__(self, data_file):
        """ This class can be used to load the comm20, sbm and planar datasets. """
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)
        self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(
            filename)
        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        y = torch.zeros([1, 0]).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes)
        return data

class Comm20Dataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('community_12_21_100.pt')


class SBMDataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('sbm_200.pt')


class PlanarDataset(SpectreGraphDataset):
    def __init__(self):
        super().__init__('planar_64_200.pt')

class ResiDataset(Dataset):
    def __init__(self, mech, cfg):

        super().__init__()

        self.cfg = cfg
        """ This class can be used to load the comm20, sbm and planar datasets. """
        if mech == 1:
            # data_file = 'mech_1_mix.pt'
            # data_file = 'resilience/mech_1_discrete.pt'
            data_file = 'resilience/mech_1_minorcontrol.pkl'
        elif mech == 2:
            # data_file = 'mech_2_highb_use.pt'
            # print('Warning....Directed graph has not been implemented yet.')
            # print('Warning....Directed graph has not been implemented yet.')
            # print('Warning....Directed graph has not been implemented yet.')
            # data_file = 'resilience/mech_2_bi.pt'
            data_file = 'diffusion/regulatory/As_unlabeled_10.pt'
            
        elif mech == 3:
            # data_file = 'mech_3_use.pt'
            # data_file = 'resilience/mech_3.pt'
            # data_file = 'resilience/mech_3_inpaint.pkl'
            # data_file = 'resilience/mech_3_inpaint.pkl'
            
            # data_file = 'resilience/mech_3_minorcontrol.pkl'
            # data_file = 'resilience/mech_3_minorcontrol.pkl'
            # data_file = 'diffusion/neuronal/nets_from_test.pt'
            # data_file = 'diffusion/neuronal/bas.pt'
            # data_file = 'diffusion/neuronal/net_test_neg.pt'
            data_file = self.cfg.dataset.data_file_path
            
            
        else:
            raise NotImplementedError
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        # base_path = os.path.join('/data3/liuchang/workspace/data_enhance', 'data')
        filename = os.path.join(base_path, data_file)
        self.adjs = torch.load(filename)
        # self.adjs = pickle.load(open(filename, 'rb'))['original']

        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        adj = torch.from_numpy(np.array(self.adjs[idx]))
        # adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        y = torch.zeros([1, 0]).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        cond_mask = torch.ones(n, 1, dtype=torch.long)
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes, cond_mask=cond_mask)
        return data

class ResiBipartiteDataset(Dataset):

    def __init__(self, mech):

        if mech == 1:
           raise NotImplementedError
        elif mech == 2:
            raise NotImplementedError
        elif mech == 3:
            
            data_file = 'resilience/bipartite/bi_nets.pt'
        
        else:
            raise NotImplementedError
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)
        # self.adjs = torch.load(filename)
        # self.adjs = pickle.load(open(filename, 'rb'))['bipartites']

        dt = torch.load(filename)
        self.adjs = dt['adjs']
        self.bipartites = dt['bipartites']
        self.top_nets = dt['As']
        self.bot_nets = dt['Bs']

        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        adj = self.adjs[idx]
        top_net = self.top_nets[idx]
        bot_net = self.bot_nets[idx]

        all_nodes = adj.shape[-1]
        t_nodes = top_net.shape[-1]
        b_nodes = bot_net.shape[-1]

        X = torch.ones(all_nodes, 1, dtype=torch.float)
        y = torch.zeros([1, 0]).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = all_nodes * torch.ones(1, dtype=torch.long)
        bipartite_mask = torch.zeros(all_nodes, 1, dtype=torch.long)
        bipartite_mask[t_nodes:] = 1
        t_nodes = t_nodes * torch.ones(1, dtype=torch.long)
        b_nodes = b_nodes * torch.ones(1, dtype=torch.long)

        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes, t_nodes=t_nodes, b_nodes=b_nodes, bipartite_mask=bipartite_mask)
        return data

class ResiPairDataset(Dataset):
    def __init__(self, mech):
        if mech == 1:
            # data_file = 'mech_1_mix.pt'
            raise NotImplementedError
        elif mech == 2:
            # data_file = 'mech_2_highb_use.pt'
            # print('Warning....Directed graph has not been implemented yet.')
            # print('Warning....Directed graph has not been implemented yet.')
            # print('Warning....Directed graph has not been implemented yet.')
            raise NotImplementedError
        elif mech == 3:
            # data_file = 'mech_3_mix.pt'
            # data_pair = 'mech_3_mix_pair.pt'
            data_file = 'data_pair.pkl'
        else:
            raise NotImplementedError
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)
        # self.adjs = torch.load(filename)
        self.data_pair = pickle.load(open(filename, 'rb'))
        print(f'Dataset {filename} loaded from file')

    
    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, idx):
        data = self.data_pair[idx]
        assert data[0].shape[-1] == data[1].shape[-1]
        n = data[0].shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        y = torch.zeros([1, 0]).float()
        nonres_topo = data[0]
        res_topo = data[1]

        ### for nonres_topo
        edge_index, _ = torch_geometric.utils.dense_to_sparse(nonres_topo)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        data_nonres = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes)
        

        ### for res_topo
        edge_index, _ = torch_geometric.utils.dense_to_sparse(res_topo)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        data_res = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                            y=y, idx=idx, n_nodes=num_nodes)
        return data_nonres, data_res
    
class ResiBipartiteCondDataset(Dataset):

    def __init__(self, mech, cfg):
        self.cfg = cfg

        if self.cfg.general.heuristic:

            self.special = 'heuristic'
            self.perturb_rate = self.cfg.general.perturb_rate

        else:
            raise NotImplementedError
        
        if mech == 1:
           raise NotImplementedError
        elif mech == 2:
            raise NotImplementedError
        elif mech == 3:
            
            if cfg.general.test_general:
                if 'test_real' in cfg.general:
                    if cfg.general.test_real:
                        if cfg.general.real_suffix is not None:
                            data_file = f'resilience/bipartite/real_data_{cfg.general.real_suffix}.pkl'
                        else:
                            data_file = f'resilience/bipartite/real_data.pkl'
                    else:
                        raise NotImplementedError
                else:
                    data_file = 'resilience/bipartite/bi_general_test.pkl'
                # data_file = 'resilience/bipartite/bi_general_check.pkl'
            else:
                data_file = 'resilience/bipartite/bi_nets_minorcontrol.pkl'
        
        else:
            raise NotImplementedError

        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)

      
        
        if self.special == 'heuristic':

            if cfg.general.test_general:
                self.ori_adjs = pickle.load(open(filename, 'rb'))['original']
                self.adjs = pickle.load(open(filename, 'rb'))['perturbed']
                self.top_nodes = pickle.load(open(filename, 'rb'))['top_nodes']

            else:
                self.ori_adjs = pickle.load(open(filename, 'rb'))['original']
                self.adjs = pickle.load(open(filename, 'rb'))['perturbed']
                self.top_nodes = pickle.load(open(filename, 'rb'))['top_nodes']

            self.strategy = cfg.general.control_strategy
        else:
            raise NotImplementedError
        
       

        print(f'Dataset {filename} loaded from file')
        
    def __len__(self):
        return len(self.adjs)
    
    def __getitem__(self, idx):
        adj = torch.from_numpy(self.adjs[idx])
        ori_adj = torch.from_numpy(self.ori_adjs[idx])
        # adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        y = torch.zeros([1, 0]).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        t_nodes = self.top_nodes[idx]
        b_nodes = n - t_nodes
        bipartite_mask = torch.zeros(n, 1, dtype=torch.long)
        bipartite_mask[t_nodes:] = 1
        t_nodes = t_nodes * torch.ones(1, dtype=torch.long)
        b_nodes = b_nodes * torch.ones(1, dtype=torch.long)

        if self.special == 'heuristic':

            gra = nx.from_numpy_array(self.adjs[idx])

            if self.strategy == 'degree':
                
                degree_centrality = nx.degree_centrality(gra)
                nodes_sorted = sorted(degree_centrality.keys(), key=degree_centrality.get, reverse=self.cfg.general.reverse)

            elif self.strategy == 'closeness':
                
                close_centrality = nx.closeness_centrality(gra)
                nodes_sorted = sorted(close_centrality.keys(), key=close_centrality.get, reverse=self.cfg.general.reverse)

            elif self.strategy == 'betweenness':

                between_centrality = nx.betweenness_centrality(gra)
                nodes_sorted = sorted(between_centrality.keys(), key=between_centrality.get, reverse=self.cfg.general.reverse)
            
            elif self.strategy == 'eigenvector':

                eigenvec_centrality = nx.eigenvector_centrality(gra)
                nodes_sorted = sorted(eigenvec_centrality.keys(), key=eigenvec_centrality.get, reverse=self.cfg.general.reverse)

            elif self.strategy == 'resilience':

                degrees = [val for (node, val) in gra.degree()]
                degrees = np.array(degrees).reshape(-1, 1)
                nearest_neighbor_degree = np.dot(adj.numpy(), degrees).reshape(-1)
                degrees = degrees.reshape(-1)
                num_nodes = gra.number_of_nodes()

                
                mean_degree = np.mean(degrees)
                std_degree = np.std(degrees)

                resilience_centralities = (2*nearest_neighbor_degree+degrees*(degrees-2*calc_beta(adj.numpy())))/(num_nodes*(mean_degree**2+std_degree**2))
                # print(resilience_centralities)
                resilience_centrality_dict = {idx: value for idx, value in enumerate(resilience_centralities)}
                nodes_sorted = sorted(resilience_centrality_dict.keys(), key=resilience_centrality_dict.get, reverse=self.cfg.general.reverse)
                # nodes_sorted = sorted(range(len(resilience_centralities)), key=lambda k: resilience_centralities[k], reverse=True)
            
            
            else:
                print('Warning if is not community')
                nodes_sorted = list(range(n))
            
            total_num = len(nodes_sorted)
                
            select_num = int(total_num * self.cfg.general.control_ratio)

            selected_nodes = nodes_sorted[:select_num]
            
            if self.strategy == 'community':
                
                graph = nx.from_numpy_array(adj.numpy())
                communities = nx.community.louvain_communities(graph)
                communities_filt = []
                for community in communities:
                    if len(community) >= 2:
                        communities_filt.append(list(community))
                community_betas = []
                for community in communities_filt:
                    community_betas.append(calc_beta(graph.subgraph(community)))
                
                community_beta_dict = {tuple(idx): value for idx, value in zip(communities_filt, community_betas)}
                community_sorted = sorted(community_beta_dict.keys(), key=community_beta_dict.get, reverse=False)
                total_num = len(community_sorted)
                select_num = int(total_num * self.cfg.general.control_ratio)
                if select_num == 0:
                    select_num = 1
                selected_community = community_sorted[:select_num]
                selected_nodes = [item for s in selected_community for item in s]

            cond_mask = torch.zeros(n, 1, dtype=torch.long)
            cond_mask[selected_nodes] = 1
        else:
            raise NotImplementedError
        
        
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes, t_nodes=t_nodes, b_nodes=b_nodes, cond_mask=cond_mask, bipartite_mask=bipartite_mask)
        
        if self.special == 'minorcontrol' or self.special == 'heuristic':
            n_ori = ori_adj.shape[-1]
            X_ori = torch.ones(n_ori, 1, dtype=torch.float)
            y_ori = torch.zeros([1, 0]).float()
            edge_index_ori, _ = torch_geometric.utils.dense_to_sparse(ori_adj)
            edge_attr_ori = torch.zeros(edge_index_ori.shape[-1], 2, dtype=torch.float)
            edge_attr_ori[:, 1] = 1
            num_nodes_ori = n_ori * torch.ones(1, dtype=torch.long)
            top_nodes_ori = self.top_nodes[idx]
            bot_nodes_ori = n_ori - top_nodes_ori
            t_nodes_ori = top_nodes_ori * torch.ones(1, dtype=torch.long)
            b_nodes_ori = bot_nodes_ori * torch.ones(1, dtype=torch.long)
            bipartite_mask_ori = torch.zeros(n_ori, 1, dtype=torch.long)
            bipartite_mask_ori[top_nodes_ori:] = 1

            # cond_mask = torch.from_numpy(self.masks[idx].reshape(-1,1)).long()
            data_ori = torch_geometric.data.Data(x=X_ori, edge_index=edge_index_ori, edge_attr=edge_attr_ori,
                                                y=y_ori, idx=idx, n_nodes=num_nodes_ori, t_nodes=t_nodes_ori, b_nodes=b_nodes_ori, cond_mask=cond_mask, bipartite_mask=bipartite_mask)
            return data, data_ori
        
        else:
            return data

class ResiCondDataset(Dataset):
    def __init__(self, mech, cfg):
        self.cfg = cfg
        # self.is_minorcontrol = self.cfg.general.minorcontrol
        if self.cfg.general.minorcontrol:
            self.special = 'minorcontrol'
            self.remove_rate = self.cfg.general.remove_rate
        elif self.cfg.general.heuristic:
            self.special = 'heuristic'
            self.perturb_rate = self.cfg.general.perturb_rate
        else:
            self.special = 'empty'
        """ This class can be used to load the comm20, sbm and planar datasets. """
        if mech == 1:
            # data_file = 'mech_1_use.pt'
            # raise NotImplementedError
            if self.special == 'heuristic':
                data_file = 'resilience/mech_1_minorcontrol.pkl'
            else:
                raise NotImplementedError
        elif mech == 2:
            # data_file = 'mech_2_highb_use.pt'
            print('Warning....Directed graph has not been implemented yet.')
            print('Warning....Directed graph has not been implemented yet.')
            print('Warning....Directed graph has not been implemented yet.')
            raise NotImplementedError
        elif mech == 3:
            # data_file = 'mech_3_use_non.pt'
            # data_file = 'mech_3_non_res.pt'
            if self.special == 'empty':

                data_file = 'resilience/mech_3_sub.pt'
                
            elif self.special == 'minorcontrol':

                if self.remove_rate is None:
                    data_file = 'resilience/mech_3_inpaint.pkl'
                else:
                    data_file = 'resilience/mech_3_rmnodes_{}.pkl'.format(self.remove_rate)

            elif self.special == 'heuristic':

                if self.cfg.general.larger_dataset:
                    # data_file = 'resilience/mech_3_larger_heuris_0.2.pkl'
                    data_file = 'resilience/mech_3_larger_1800_heuris_0.2.pkl'
                
                elif self.cfg.general.test_general:
                    data_file = 'resilience/general_base_nets.pkl'

                else:

                    if self.perturb_rate is None:
                        data_file = 'resilience/mech_3_minorcontrol.pkl'
                    else:
                        data_file = 'resilience/mech_3_heuris_{}.pkl'.format(self.perturb_rate)

        else:
            raise NotImplementedError
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)

        if self.special == 'empty':
            self.adjs = torch.load(filename)
        
        elif self.special == 'minorcontrol':
            self.ori_adjs = pickle.load(open(filename, 'rb'))['original']
            self.adjs = pickle.load(open(filename, 'rb'))['perturbed']
            self.masks = pickle.load(open(filename, 'rb'))['mask']
        
        elif self.special == 'heuristic':
            if self.cfg.general.test_general:
                self.ori_adjs = pickle.load(open(filename, 'rb'))
                self.adjs = pickle.load(open(filename, 'rb'))
            else:
                self.ori_adjs = pickle.load(open(filename, 'rb'))['original']
                self.adjs = pickle.load(open(filename, 'rb'))['perturbed']

            
            self.strategy = cfg.general.control_strategy
            
        
        else:
            raise NotImplementedError

        print(f'Dataset {filename} loaded from file')
    
    def __len__(self):
        return len(self.adjs)
    
    def __getitem__(self, idx):
        adj = torch.from_numpy(self.adjs[idx])
        ori_adj = torch.from_numpy(self.ori_adjs[idx])
        # adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        y = torch.zeros([1, 0]).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        if self.special == 'empty':
            cond_mask = torch.ones(n, 1, dtype=torch.long)

        elif self.special == 'heuristic':

            gra = nx.from_numpy_array(self.adjs[idx])

            if self.strategy == 'degree':
                
                degree_centrality = nx.degree_centrality(gra)
                nodes_sorted = sorted(degree_centrality.keys(), key=degree_centrality.get, reverse=self.cfg.general.reverse)

            elif self.strategy == 'closeness':
                
                close_centrality = nx.closeness_centrality(gra)
                nodes_sorted = sorted(close_centrality.keys(), key=close_centrality.get, reverse=self.cfg.general.reverse)

            elif self.strategy == 'betweenness':

                between_centrality = nx.betweenness_centrality(gra)
                nodes_sorted = sorted(between_centrality.keys(), key=between_centrality.get, reverse=self.cfg.general.reverse)
            
            elif self.strategy == 'eigenvector':

                eigenvec_centrality = nx.eigenvector_centrality(gra)
                nodes_sorted = sorted(eigenvec_centrality.keys(), key=eigenvec_centrality.get, reverse=self.cfg.general.reverse)

            elif self.strategy == 'resilience':

                degrees = [val for (node, val) in gra.degree()]
                degrees = np.array(degrees).reshape(-1, 1)
                nearest_neighbor_degree = np.dot(adj.numpy(), degrees).reshape(-1)
                degrees = degrees.reshape(-1)
                num_nodes = gra.number_of_nodes()

                
                mean_degree = np.mean(degrees)
                std_degree = np.std(degrees)

                resilience_centralities = (2*nearest_neighbor_degree+degrees*(degrees-2*calc_beta(adj.numpy())))/(num_nodes*(mean_degree**2+std_degree**2))
                # print(resilience_centralities)
                resilience_centrality_dict = {idx: value for idx, value in enumerate(resilience_centralities)}
                nodes_sorted = sorted(resilience_centrality_dict.keys(), key=resilience_centrality_dict.get, reverse=self.cfg.general.reverse)
                # nodes_sorted = sorted(range(len(resilience_centralities)), key=lambda k: resilience_centralities[k], reverse=True)
            
            
            else:
                print('Warning if is not community')
                nodes_sorted = list(range(n))
            
            total_num = len(nodes_sorted)
                
            select_num = int(total_num * self.cfg.general.control_ratio)

            selected_nodes = nodes_sorted[:select_num]
            
            if self.strategy == 'community':
                
                graph = nx.from_numpy_array(adj.numpy())
                communities = nx.community.louvain_communities(graph)
                communities_filt = []
                for community in communities:
                    if len(community) >= 2:
                        communities_filt.append(list(community))
                community_betas = []
                for community in communities_filt:
                    community_betas.append(calc_beta(graph.subgraph(community)))
                
                community_beta_dict = {tuple(idx): value for idx, value in zip(communities_filt, community_betas)}
                community_sorted = sorted(community_beta_dict.keys(), key=community_beta_dict.get, reverse=False)
                total_num = len(community_sorted)
                select_num = int(total_num * self.cfg.general.control_ratio)
                if select_num == 0:
                    select_num = 1
                selected_community = community_sorted[:select_num]
                selected_nodes = [item for s in selected_community for item in s]

            cond_mask = torch.zeros(n, 1, dtype=torch.long)
            cond_mask[selected_nodes] = 1
        
        elif self.special == 'minorcontrol':
            cond_mask = torch.from_numpy(self.masks[idx].reshape(-1, 1)).long()
        
        else:
            raise NotImplementedError

        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes, cond_mask=cond_mask)
        
        if self.special == 'minorcontrol' or self.special == 'heuristic':
            n_ori = ori_adj.shape[-1]
            X_ori = torch.ones(n_ori, 1, dtype=torch.float)
            y_ori = torch.zeros([1, 0]).float()
            edge_index_ori, _ = torch_geometric.utils.dense_to_sparse(ori_adj)
            edge_attr_ori = torch.zeros(edge_index_ori.shape[-1], 2, dtype=torch.float)
            edge_attr_ori[:, 1] = 1
            num_nodes_ori = n_ori * torch.ones(1, dtype=torch.long)
            # cond_mask = torch.from_numpy(self.masks[idx].reshape(-1,1)).long()
            data_ori = torch_geometric.data.Data(x=X_ori, edge_index=edge_index_ori, edge_attr=edge_attr_ori,
                                                y=y_ori, idx=idx, n_nodes=num_nodes_ori, cond_mask=cond_mask)
            return data, data_ori
        
        else:
            return data

class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        super().__init__(cfg)
        self.cfg = cfg
        self.n_graphs = n_graphs
        self.prepare_data()
        self.inner = self.train_dataloader()

    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self, graphs, cond_graphs=None):
        test_len = int(round(len(graphs) * 0.2))
        train_len = int(round((len(graphs) - test_len) * 0.8))
        val_len = len(graphs) - train_len - test_len
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        splits = random_split(graphs, [train_len, val_len, test_len])
        datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        if cond_graphs is not None:
            
            if self.cfg.general.test_general:
                train_len_cond = 1
                val_len_cond = 1
                test_len_cond = len(cond_graphs) - train_len_cond - val_len_cond
            else:
                test_len_cond = int(round(len(cond_graphs) * 0.2))
                train_len_cond = int(round((len(cond_graphs) - test_len_cond) * 0.8))

                val_len_cond = len(cond_graphs) - train_len_cond - test_len_cond

            print(f'Conditional Dataset sizes: val {val_len_cond}, test {test_len_cond}')
            splits_cond = random_split(cond_graphs, [train_len_cond, val_len_cond, test_len_cond])
            datasets_cond = {'train': splits_cond[0], 'val': splits_cond[1], 'test': splits_cond[2]}
            super().prepare_data(datasets, datasets_cond=datasets_cond)
        else:
            super().prepare_data(datasets)


class Comm20DataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = Comm20Dataset()
        return super().prepare_data(graphs)


class SBMDataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = SBMDataset()
        return super().prepare_data(graphs)


class PlanarDataModule(SpectreGraphDataModule):
    def prepare_data(self):
        graphs = PlanarDataset()
        return super().prepare_data(graphs)

class ResiDataModule(SpectreGraphDataModule):
    def __init__(self, cfg, n_graphs=200, mech=1):
        self.mech = mech
        self.cfg = cfg
        
        super().__init__(cfg, n_graphs)
    def prepare_data(self):
        if self.cfg.general.bipartite:
            graphs = ResiBipartiteDataset(self.mech)
        else:
            graphs = ResiDataset(self.mech, self.cfg)
        if self.cfg.general.conditional:

            if self.cfg.general.bipartite:

                cond_graphs = ResiBipartiteCondDataset(self.mech, self.cfg)
            
            else:
                cond_graphs = ResiCondDataset(self.mech, self.cfg)
        else:
            cond_graphs = None
        return super().prepare_data(graphs, cond_graphs)


# class ResiDataSplitModule(SpectreGraphDataModule):
#     def __init__(self, cfg, n_graphs=200, mech=1):
#         self.mech = mech
#         self.cfg = cfg
#         super().__init__(cfg, n_graphs)
    
#     def prepare_data(self):
#         graphs = ResiDataset(self.mech)
#         if self.cfg.general.conditional:
#             cond_graph_train = ResiCondDataset(self.mech, self.cfg, mode='train')
#             cond_graph
#             cond_graphs = ResiCondDataset(self.mech, self.cfg)
#         else:
#             cond_graphs = None
#         return super().prepare_data(graphs, cond_graphs)

class ResiPairDataModule(SpectreGraphDataModule):
    def __init__(self, cfg, n_graphs=200, mech=3):
        self.cfg = cfg
        self.mech = mech
        super().__init__(cfg, n_graphs)
    def prepare_data(self):
        graphs = ResiPairDataset()
        return super().prepare_data(graphs)
    
class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        if hasattr(self.datamodule, 'node_counts_val'):
            self.n_nodes_val = self.datamodule.node_counts_val()
        if hasattr(self.datamodule, 'node_counts_test'):
            self.n_nodes_test = self.datamodule.node_counts_test()
        if hasattr(self.datamodule, 'top_node_counts') and self.datamodule.cfg.general.bipartite:
            self.n_nodes_top = self.datamodule.top_node_counts()
        else:
            self.n_nodes_top = None
        self.node_types = torch.Tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types, self.n_nodes_val, self.n_nodes_test, self.n_nodes_top)

class ResiPairDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts_pair()
        self.node_types = torch.Tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts_pair()
        super().complete_infos(self.n_nodes, self.node_types, self.n_nodes_val, self.n_nodes_test)

