import sys
sys.path.append("..")
sys.path.append('../..')
# sys.path.append('/data2/liuchang/workspace/GraphGene/GDiff/src')
import os
import pickle

import torch
import numpy as np
from torch.utils.data import random_split, Dataset
import torch_geometric.utils

# from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from datasets.utils import network_features
from datasets.spectre_dataset import SpectreGraphDataModule

class ResiGeneralDataset(Dataset):
    def __init__(self, mech, cfg, params, mode=None, traj_min=None, traj_max=None):
        """ This class can be used to load the comm20, sbm and planar datasets. """

        self.cfg = cfg
        self.mode = mode
        if self.mode is None:
            raise NotImplementedError

        if self.cfg.general.adapt:
            self.n_data_per_env = cfg.train.n_data_per_env_adapt[str(mode)]
        else:
            self.n_data_per_env = cfg.train.n_data_per_env[str(mode)]
        self.num_env = cfg.train.num_env
        self.params_eq = params
        self.indices = [list(range(env * self.n_data_per_env, (env + 1) * self.n_data_per_env)) for env in range(self.num_env)]
        if mech == 1:
            raise NotImplementedError
        
        elif mech == 2:
            raise NotImplementedError
        
        elif mech == 3:
            if self.mode is None:
                if self.cfg.general.for_test:
                    data_file = cfg.general.for_test_data_file
                else:
                    data_file = cfg.general.data_file
            else:
                if self.cfg.general.for_test:
                    if self.mode == 'train':
                        data_file = cfg.general.for_test_calc_traj_data_file
                    elif self.mode == 'val':
                        data_file = cfg.general.for_test_calc_traj_data_file
                    else:
                        data_file = cfg.general.for_test_data_file
                else:
                    if cfg.general.adapt:
                        data_file = os.path.join(cfg.general.data_file_split_folder_adapt, f'{self.mode}.pt')
                    else:
                        data_file = os.path.join(cfg.general.data_file_split_folder, f'{self.mode}.pt')
        else:
            raise NotImplementedError
        
        # base_path = os.path.join('/data2/liuchang/workspace/GraphGene/GDiff', 'data')
        base_path = os.path.join('/data3/liuchang/workspace/data_enhance', 'data')

        filename = os.path.join(base_path, data_file)

        dataset = torch.load(filename)

        self.adjs = dataset['As']

        self.traj_hi = dataset['x_reals_his']

        self.traj_lo = dataset['x_reals_los']
        
        if cfg.general.adapt:
            self.traj_max, self.traj_min = cfg.general.traj_max, cfg.general.traj_min

        else:

            if self.mode == 'train':
            
                    self.traj_max, self.traj_min = self.scaling_coeff(self.traj_hi, self.traj_lo)
            
            elif self.mode is None:

                self.traj_max, self.traj_min = self.scaling_coeff(self.traj_hi, self.traj_lo)
                # raise RuntimeError('Unknown mode, cannot determine min-max scaling.')
            
            else:

                if traj_min is None or traj_max is None:
                    raise RuntimeError('Val/test mode, cannot determine min-max scaling.')
                
                else:
                    self.traj_min = traj_min
                    self.traj_max = traj_max

        self.labels = dataset['labels']
        
        self.ts = dataset['ts']

        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)
    
    def __getitem__(self, idx):
        env = idx // self.n_data_per_env
        env_index = idx % self.n_data_per_env
        graph_index = idx % len(self.adjs)
        adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        # y = torch.zeros([1, 0]).float()
        y = torch.from_numpy(np.array([self.labels[idx]])).float().reshape(-1,1)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        cond_mask = torch.ones(n, 1, dtype=torch.long)

        traj_hi = self.traj_hi[idx]
        traj_hi = (traj_hi - self.traj_min) / (self.traj_max - self.traj_min)

        traj_hi_fused = network_features(adj, traj_hi)
        traj_lo = self.traj_lo[idx]
        traj_lo = (traj_lo - self.traj_min) / (self.traj_max - self.traj_min)
        traj_lo_fused = network_features(adj, traj_lo)

        t = self.ts[idx]
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes, cond_mask=cond_mask, traj_hi=traj_hi_fused, traj_lo=traj_lo_fused,t=t, env=env)
        return data

    def scaling_coeff(self, traj_his, traj_los):
        all_traj_his = torch.cat(traj_his, dim=0)
        all_traj_los = torch.cat(traj_los, dim=0)
        all_traj = torch.cat([all_traj_his, all_traj_los], dim=0)
        max_val = torch.max(all_traj)
        min_val = torch.min(all_traj)
        return max_val, min_val
    

class ResiGeneralBipartiteDataset(Dataset):
    def __init__(self, mech, cfg, params, mode=None, traj_min=None, traj_max=None):
        """ This class can be used to load the comm20, sbm and planar datasets. """

        self.cfg = cfg
        self.mode = mode
        if self.mode is None:
            raise NotImplementedError

        if self.cfg.general.adapt:
            self.n_data_per_env = cfg.train.n_data_per_env_adapt[str(mode)]
        else:
            self.n_data_per_env = cfg.train.n_data_per_env[str(mode)]
        self.num_env = cfg.train.num_env
        self.params_eq = params
        self.indices = [list(range(env * self.n_data_per_env, (env + 1) * self.n_data_per_env)) for env in range(self.num_env)]
        if mech == 1:
            raise NotImplementedError
        
        elif mech == 2:
            raise NotImplementedError
        
        elif mech == 3:
            if self.mode is None:
                if self.cfg.general.for_test:
                    data_file = cfg.general.for_test_data_file
                else:
                    data_file = cfg.general.data_file
            else:
                if self.cfg.general.for_test:
                    if self.mode == 'train':
                        data_file = cfg.general.for_test_calc_traj_data_file
                    elif self.mode == 'val':
                        data_file = cfg.general.for_test_calc_traj_data_file
                    else:
                        data_file = cfg.general.for_test_data_file
                else:
                    if cfg.general.adapt:
                        data_file = os.path.join(cfg.general.data_file_split_folder_adapt, f'{self.mode}.pt')
                    else:
                        data_file = os.path.join(cfg.general.data_file_split_folder, f'{self.mode}.pt')
        else:
            raise NotImplementedError
        
        base_path = os.path.join('/data2/liuchang/workspace/GraphGene/GDiff', 'data')

        filename = os.path.join(base_path, data_file)

        dataset = torch.load(filename)

        self.adjs = dataset['As']

        self.traj_hi = dataset['x_reals_his']

        self.traj_lo = dataset['x_reals_los']

        self.traj_hi_a = dataset['x_reals_his_a']
        self.traj_hi_b = dataset['x_reals_his_b']

        self.traj_lo_a = dataset['x_reals_los_a']
        self.traj_lo_b = dataset['x_reals_los_b']
        
        if cfg.general.adapt:
            self.traj_max, self.traj_min = cfg.general.traj_max, cfg.general.traj_min
        else:
            if self.mode == 'train':
            
                    self.traj_max, self.traj_min = self.scaling_coeff(self.traj_hi, self.traj_lo)
            
            elif self.mode is None:

                self.traj_max, self.traj_min = self.scaling_coeff(self.traj_hi, self.traj_lo)
                # raise RuntimeError('Unknown mode, cannot determine min-max scaling.')
            
            else:

                if traj_min is None or traj_max is None:
                    raise RuntimeError('Val/test mode, cannot determine min-max scaling.')
                
                else:
                    self.traj_min = traj_min
                    self.traj_max = traj_max

        self.labels = dataset['labels']
        
        self.ts = dataset['ts']

        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)
    
    def __getitem__(self, idx):
        env = idx // self.n_data_per_env
        env_index = idx % self.n_data_per_env
        graph_index = idx % len(self.adjs)
        adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        # y = torch.zeros([1, 0]).float()
        y = torch.from_numpy(np.array([self.labels[idx]])).float().reshape(-1,1)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1

        traj_hi_a = self.traj_hi_a[idx]

        top_nodes = traj_hi_a.shape[0]
        top_nodes = top_nodes * torch.ones(1, dtype=torch.long)
        num_nodes = n * torch.ones(1, dtype=torch.long)
        cond_mask = torch.ones(n, 1, dtype=torch.long)

        bipartite_mask = torch.zeros(n, 1, dtype=torch.long)
        bipartite_mask[top_nodes:] = 1

        traj_hi = self.traj_hi[idx]
        traj_hi = (traj_hi - self.traj_min) / (self.traj_max - self.traj_min)

        traj_hi_fused = network_features(adj, traj_hi)
        traj_lo = self.traj_lo[idx]
        traj_lo = (traj_lo - self.traj_min) / (self.traj_max - self.traj_min)
        traj_lo_fused = network_features(adj, traj_lo)
        
        t = self.ts[idx]
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes, cond_mask=cond_mask, traj_hi=traj_hi_fused, traj_lo=traj_lo_fused,t=t, env=env, bipartite_mask=bipartite_mask, top_nodes=top_nodes)
        return data

    def scaling_coeff(self, traj_his, traj_los):
        all_traj_his = torch.cat(traj_his, dim=0)
        all_traj_los = torch.cat(traj_los, dim=0)
        all_traj = torch.cat([all_traj_his, all_traj_los], dim=0)
        max_val = torch.max(all_traj)
        min_val = torch.min(all_traj)
        return max_val, min_val
    



    
class ResiGeneralSplitDataModule(SpectreGraphDataModule):

    def __init__(self, cfg, n_graphs=200, mech=3):

        self.mech = mech
        self.cfg = cfg
        if self.cfg.general.adapt:

            self.params = [
            {"mu": 3.68, "delta": 2.06, "beta_c": 7.58},
            {"mu": 3.56, "delta": 1.95, "beta_c": 7.15},
            {"mu": 3.66, "delta": 1.85, "beta_c": 8.28},
            ]

        else:
            self.params = [
                {"mu": 3.5, "delta": 2},
                {"mu": 3, "delta": 2},
                {"mu": 2.7, "delta": 1.5},
                {"mu": 3.2, "delta": 2},
                {"mu": 3, "delta": 1.5},
                {"mu": 3.6, "delta": 1.8},
                {"mu": 3.65, "delta": 2.12},
                {"mu": 3.65, "delta": 1.78},
                {"mu": 3.85, "delta": 2.1},
                ]
        super().__init__(cfg, n_graphs=n_graphs)

    def prepare_data(self):
        
        if self.cfg.dataset.pregene:

            if 'bipartite' in self.cfg.general:
                if self.cfg.general.bipartite:
                    graphs_train = ResiGeneralBipartiteDataset(self.mech, self.cfg, self.params, mode='train')
                    traj_min, traj_max = graphs_train.traj_min, graphs_train.traj_max
                    print('train traj_min: ', traj_min)
                    print('train traj_max: ', traj_max)
                    graphs_val = ResiGeneralBipartiteDataset(self.mech, self.cfg, self.params, mode='val', traj_min=traj_min, traj_max=traj_max)
                    graphs_test = ResiGeneralBipartiteDataset(self.mech, self.cfg, self.params, mode='test', traj_min=traj_min, traj_max=traj_max)
                    self.traj_min = traj_min
                    self.traj_max = traj_max

            else:
                graphs_train = ResiGeneralDataset(self.mech, self.cfg, self.params, mode='train')
                traj_min, traj_max = graphs_train.traj_min, graphs_train.traj_max
                print('train traj_min: ', traj_min)
                print('train traj_max: ', traj_max)
                graphs_val = ResiGeneralDataset(self.mech, self.cfg, self.params, mode='val', traj_min=traj_min, traj_max=traj_max)
                graphs_test = ResiGeneralDataset(self.mech, self.cfg, self.params, mode='test', traj_min=traj_min, traj_max=traj_max)
                self.traj_min = traj_min
                self.traj_max = traj_max
        else:
            raise NotImplementedError


        cond_graphs = None
        if self.cfg.dataset.pregene:
            return super().prepare_data([graphs_train, graphs_val, graphs_test], cond_graphs)
        else:
            raise RuntimeError('Not implemented yet.')
