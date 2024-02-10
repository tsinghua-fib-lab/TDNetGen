import sys
sys.path.append("..")
sys.path.append('../..')
sys.path.append('/data2/liuchang/workspace/GraphGene/GDiff/src')
import os
import pickle

import torch
import numpy as np
from torch.utils.data import random_split, Dataset
import torch_geometric.utils

# from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from datasets.utils import network_features

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
    def __init__(self, mech):
        """ This class can be used to load the comm20, sbm and planar datasets. """
        if mech == 1:
            # data_file = 'mech_1_mix.pt'
            data_file = 'resilience/mech_1_discrete.pt'
        elif mech == 2:
            # data_file = 'mech_2_highb_use.pt'
            # print('Warning....Directed graph has not been implemented yet.')
            # print('Warning....Directed graph has not been implemented yet.')
            # print('Warning....Directed graph has not been implemented yet.')
            data_file = 'resilience/mech_2_bi.pt'
        elif mech == 3:
            # data_file = 'mech_3_use.pt'
            data_file = 'resilience/mech_3.pt'
            # data_file = 'resilience/mech_3_inpaint.pkl'
        else:
            raise NotImplementedError
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)
        self.adjs = torch.load(filename)
        # self.adjs = pickle.load(open(filename, 'rb'))['original']

        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        # adj = torch.from_numpy(self.adjs[idx])
        adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        y = torch.zeros([1, 0]).float()
        # y = torch.ones([1,1]).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        cond_mask = torch.ones(n, 1, dtype=torch.long)
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes, cond_mask=cond_mask)
        return data


class ResiGuideDataset(Dataset):
    def __init__(self,mech, dtpath):
        """ This class can be used to load the comm20, sbm and planar datasets. """

        self.dtpath = dtpath
        if mech == 1:
            # data_file = 'mech_1_mix.pt'
            # data_file = 'resilience/mech_1_discrete.pt'
            raise NotImplementedError
        
        elif mech == 2:
            # data_file = 'mech_2_highb_use.pt'
            # print('Warning....Directed graph has not been implemented yet.')
            # print('Warning....Directed graph has not been implemented yet.')
            # print('Warning....Directed graph has not been implemented yet.')
            # data_file = 'resilience/mech_2_bi.pt'
            raise NotImplementedError
        
        elif mech == 3:
            # data_file = 'graphode/mech_3.pt'
            data_file = self.dtpath
        else:
            raise NotImplementedError

        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')

        filename = os.path.join(base_path, data_file)

        self.adjs = torch.load(filename)['graphs']

        self.labels = torch.load(filename)['labels']

        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)
    
    def __getitem__(self, idx):
        # adj = torch.from_numpy(self.adjs[idx])
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
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes, cond_mask=cond_mask)
        return data

class ResiGuideTrajDataset(Dataset):
    def __init__(self, mech, cfg, mode=None, traj_min=None, traj_max=None):
        """ This class can be used to load the comm20, sbm and planar datasets. """

        self.cfg = cfg
        self.mode = mode
        if mech == 1:
            # data_file = 'mech_1_mix.pt'
            # data_file = 'resilience/mech_1_discrete.pt'
            raise NotImplementedError
        
        elif mech == 2:
            # data_file = 'mech_2_highb_use.pt'
            # print('Warning....Directed graph has not been implemented yet.')
            # print('Warning....Directed graph has not been implemented yet.')
            # print('Warning....Directed graph has not been implemented yet.')
            # data_file = 'resilience/mech_2_bi.pt'
            raise NotImplementedError
        
        elif mech == 3:
            # data_file = 'graphode/mech_3.pt'
            # data_file = self.dtpath
            # data_file = 'graphode/mech_3_small_{}_{}.pt'.format(self.cfg.train.T, self.cfg.train)
            # data_file = 'graphode/mech_3_large.pt'
            # data_file = 'graphode/mech_3_small_1_50.pt'
            # data_file = 'graphode/mech_3_large_{}_{}_noiselabel.pt'.format(self.cfg.train.T, self.cfg.train.time_ticks)
            # data_file = 'graphode/mech_3_small_{}_{}_new.pt'.format(self.cfg.train.T, self.cfg.train.time_ticks)
            # data_file = 'graphode/mech_3_large_{}_{}_ori_labels.pt'.format(self.cfg.train.T, self.cfg.train.time_ticks)
            # data_file = 'graphode/mech_3_large_{}_{}_noiselabel.pt'.format(self.cfg.train.T, self.cfg.train.time_ticks)
            if self.mode is None:
                # data_file = 'graphode/test_data_0.pt'
                if self.cfg.general.for_test:
                    data_file = cfg.general.for_test_data_file
                else:
                    data_file = cfg.general.data_file
                    # data_file = f'graphode/split_test/mech_3_test_noiselables.pt'
                    # print('Warning: the dataset may not be true!!!!!')
                    # print('Warning: the dataset may not be true!!!!!')
                    # print('Warning: the dataset may not be true!!!!!')
                    # raise RuntimeError('Please specify the dataset file.')
            else:
                # data_file = f'graphode/split_test/mech_3_{mode}_noiselables_{cfg.train.T}_{cfg.train.time_ticks}.pt'

                if self.cfg.general.for_test:
                    if self.mode == 'train':
                        data_file = cfg.general.for_test_calc_traj_data_file
                    elif self.mode == 'val':
                        data_file = cfg.general.for_test_calc_traj_data_file
                    else:
                        data_file = cfg.general.for_test_data_file
                else:
                    data_file = os.path.join(cfg.general.data_file_split_folder, f'{self.mode}.pt')

        else:
            raise NotImplementedError

        # base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        # base_path = os.path.join('/data2/liuchang/workspace/GraphGene/GDiff', 'data')
        base_path = os.path.join('/data3/liuchang/workspace/data_enhance', 'data')

        filename = os.path.join(base_path, data_file)

        dataset = torch.load(filename)

        if self.mode == 'train':

            train_percent = cfg.general.train_percent
            all_len = len(dataset['As'])
            train_len = int(all_len * train_percent)
            idxs = np.arange(all_len)
            train_idxs = np.random.choice(idxs, train_len, replace=False)
            
        if self.mode == 'train':
            self.adjs = [dataset['As'][i] for i in train_idxs]

            self.traj_hi = [dataset['x_reals_his'][i] for i in train_idxs]

            self.traj_lo = [dataset['x_reals_los'][i] for i in train_idxs]
        else:
            self.adjs = dataset['As']
            self.traj_hi = dataset['x_reals_his']
            self.traj_lo = dataset['x_reals_los']


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

        if self.mode == 'train':
            self.labels = [dataset['labels'][i] for i in train_idxs]
            
            self.ts = [dataset['ts'][i] for i in train_idxs]
        else:
            self.labels = dataset['labels']
            self.ts = dataset['ts']

        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)
    
    def __getitem__(self, idx):
        # adj = torch.from_numpy(self.adjs[idx])
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
                                         y=y, idx=idx, n_nodes=num_nodes, cond_mask=cond_mask, traj_hi=traj_hi_fused, traj_lo=traj_lo_fused,t=t)
        return data

    def scaling_coeff(self, traj_his, traj_los):
        all_traj_his = torch.cat(traj_his, dim=0)
        all_traj_los = torch.cat(traj_los, dim=0)
        all_traj = torch.cat([all_traj_his, all_traj_los], dim=0)

        
        max_val = torch.max(all_traj)
        min_val = torch.min(all_traj)

        return max_val, min_val
    

class SemiResiGuideTrajDataset(Dataset):
    def __init__(self, mech, cfg, mode=None, traj_min=None, traj_max=None):
        """ This class can be used to load the comm20, sbm and planar datasets. """

        self.cfg = cfg
        self.mode = mode
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
                    data_file = os.path.join(cfg.general.data_file_split_folder, f'{self.mode}.pt')
        else:
            raise NotImplementedError

        base_path = os.path.join('/data3/liuchang/workspace/data_enhance', 'data')

        filename = os.path.join(base_path, data_file)

        dataset = torch.load(filename)

        # if self.mode == 'train':

        #     train_percent = cfg.general.train_percent
        #     all_len = len(dataset['As'])
        #     train_len = int(all_len * train_percent)
        #     idxs = np.arange(all_len)
        #     train_idxs = np.random.choice(idxs, train_len, replace=False)
            
        # if self.mode == 'train':
        #     self.adjs = [dataset['As'][i] for i in train_idxs]

        #     self.traj_hi = [dataset['x_reals_his'][i] for i in train_idxs]

        #     self.traj_lo = [dataset['x_reals_los'][i] for i in train_idxs]
        # else:
        #     self.adjs = dataset['As']
        #     self.traj_hi = dataset['x_reals_his']
        #     self.traj_lo = dataset['x_reals_los']

        self.adjs = dataset['As']
        self.traj_hi = dataset['x_reals_his']
        self.traj_lo = dataset['x_reals_los']

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

        # self.labels = dataset['labels']

        # self.ts = dataset['ts']

        # if self.mode == 'train':
        #     self.labels = [dataset['labels'][i] for i in train_idxs]
            
        #     self.ts = [dataset['ts'][i] for i in train_idxs]
        # else:
        #     self.labels = dataset['labels']
        #     self.ts = dataset['ts']

        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)
    
    def __getitem__(self, idx):
        # adj = torch.from_numpy(self.adjs[idx])
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
                                         y=y, idx=idx, n_nodes=num_nodes, cond_mask=cond_mask, traj_hi=traj_hi_fused, traj_lo=traj_lo_fused,t=t)
        return data

    def scaling_coeff(self, traj_his, traj_los):
        all_traj_his = torch.cat(traj_his, dim=0)
        all_traj_los = torch.cat(traj_los, dim=0)
        all_traj = torch.cat([all_traj_his, all_traj_los], dim=0)

        
        max_val = torch.max(all_traj)
        min_val = torch.min(all_traj)

        return max_val, min_val



    


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

class ResiCondDataset(Dataset):
    def __init__(self, mech, minor_control=False):
        """ This class can be used to load the comm20, sbm and planar datasets. """
        self.is_minor_control = minor_control
        if mech == 1:
            # data_file = 'mech_1_use.pt'
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
            if self.is_minor_control:
                data_file = 'resilience/mech_3_inpaint.pkl'
            else:
                data_file = 'resilience/mech_3_sub.pt'
        else:
            raise NotImplementedError
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)
        if self.is_minor_control:
            self.ori_adjs = pickle.load(open(filename, 'rb'))['original']
            self.adjs = pickle.load(open(filename, 'rb'))['perturbed']
            self.masks = pickle.load(open(filename, 'rb'))['mask']
        else:
            self.adjs = torch.load(filename)
        print(f'Dataset {filename} loaded from file')
    
    def __len__(self):
        return len(self.adjs)
    
    def __getitem__(self, idx):
        if self.is_minor_control:
            adj = torch.from_numpy(self.adjs[idx])
            ori_adj = torch.from_numpy(self.ori_adjs[idx])
        else:
            adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        y = torch.zeros([1, 0]).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        if self.is_minor_control:
            cond_mask = torch.from_numpy(self.masks[idx].reshape(-1, 1)).long()
        else:
            cond_mask = torch.ones(n, 1, dtype=torch.long)
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=idx, n_nodes=num_nodes, cond_mask=cond_mask)
        
        if self.is_minor_control:
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
        self.n_graphs = n_graphs
        self.prepare_data()
        self.inner = self.train_dataloader()
        self.cfg = cfg

    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self, graphs, cond_graphs=None):
        
        if isinstance(graphs, list):
            test_len = int(len(graphs[2]))
            val_len = int(len(graphs[1]))
            train_len = int(len(graphs[0]))
            datasets = {'train': graphs[0], 'val': graphs[1], 'test': graphs[2]}

        else:
            if self.cfg.general.for_test:
                train_len = 1
                val_len = 1
                test_len = len(graphs) - train_len - val_len
            else:
                test_len = int(round(len(graphs) * 0.2))
                train_len = int(round((len(graphs) - test_len) * 0.8))
                val_len = len(graphs) - train_len - test_len
            
            print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
            splits = random_split(graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(3407))
            datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        if cond_graphs is not None:
            test_len_cond = test_len
            val_len_cond = val_len
            train_len_cond = len(cond_graphs) - test_len_cond - val_len_cond
            print(f'Conditional Dataset sizes: val {val_len_cond}, test {test_len_cond}')
            splits_cond = random_split(cond_graphs, [train_len_cond, val_len_cond, test_len_cond], generator=torch.Generator().manual_seed(3407))
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
        graphs = ResiDataset(self.mech)
        if self.cfg.general.conditional:
            cond_graphs = ResiCondDataset(self.mech, self.cfg.general.minorcontrol)
        else:
            cond_graphs = None
        return super().prepare_data(graphs, cond_graphs)
    
class ResiGuideDataModule(SpectreGraphDataModule):
    def __init__(self, cfg, n_graphs=200, mech=1):
        self.mech = mech
        self.cfg = cfg
        super().__init__(cfg, n_graphs)
    def prepare_data(self):
        if self.cfg.dataset.pregene:
            graphs = ResiGuideTrajDataset(self.mech, self.cfg)
        else:
            graphs = ResiGuideDataset(self.mech, self.cfg.general.datapath)
        if self.cfg.general.conditional:
            cond_graphs = ResiCondDataset(self.mech)
        else:
            cond_graphs = None
        return super().prepare_data(graphs, cond_graphs)




class ResiGuideSplitDataModule(SpectreGraphDataModule):
    def __init__(self, cfg, n_graphs=200, mech=1):
        self.mech = mech
        self.cfg = cfg
        super().__init__(cfg, n_graphs)
    def prepare_data(self):
        if self.cfg.dataset.pregene:
            graphs_train = ResiGuideTrajDataset(self.mech, self.cfg, mode='train')
            traj_min, traj_max = graphs_train.traj_min, graphs_train.traj_max
            print('traj_min:', traj_min)
            print('traj_max:', traj_max)
            graphs_val = ResiGuideTrajDataset(self.mech, self.cfg, mode='val', traj_min=traj_min, traj_max=traj_max)
            graphs_test = ResiGuideTrajDataset(self.mech, self.cfg, mode='test', traj_min=traj_min, traj_max=traj_max)
            self.traj_min = traj_min
            self.traj_max = traj_max
        else:
            graphs = ResiGuideDataset(self.mech, self.cfg.general.datapath)
        if self.cfg.general.conditional:
            cond_graphs = ResiCondDataset(self.mech)
        else:
            cond_graphs = None
        if self.cfg.dataset.pregene:
            return super().prepare_data([graphs_train, graphs_val, graphs_test], cond_graphs)
        else:
            raise RuntimeError('Not implemented yet. Use ResiGuideDataModule.')

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
        self.node_types = torch.Tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types, self.n_nodes_val, self.n_nodes_test)

class ResiPairDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts_pair()
        self.node_types = torch.Tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts_pair()
        super().complete_infos(self.n_nodes, self.node_types, self.n_nodes_val, self.n_nodes_test)

