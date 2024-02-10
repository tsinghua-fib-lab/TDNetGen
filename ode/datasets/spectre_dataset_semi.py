import sys
sys.path.append("..")
sys.path.append('../..')
import os
import pickle

import torch
import numpy as np
from torch.utils.data import random_split, Dataset
import torch_geometric.utils
import random

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

class FullResiGuideTrajDataset(Dataset):
    def __init__(self, mech, cfg, mode=None, traj_min=None, traj_max=None, type=None):
        """ This class can be used to load the comm20, sbm and planar datasets. """

        self.cfg = cfg
        self.mode = mode
        self.ode_use_length = cfg.train.ode_use_length
        self.here_type = type
        if mech == 1:
            raise NotImplementedError
        
        elif mech == 2:
        
            raise NotImplementedError
        
        elif mech == 3:
            
            # if type == 'ode_train':

                # data_file = 
            
           
            base_data_file_path = 'graphode/' + cfg.dataset.dyna + f'/mech_{cfg.dataset.mech_id}_train_5.pt'

            if type == 'ode_train':
                
                
                data_file = cfg.dataset.ode_train_data_file
                

            elif type == 'resinf_train':

                # data_file = cfg.dataset.resinf_train_data_file
                data_file = base_data_file_path
            
            elif type == 'finetune':

                # data_file = cfg.dataset.finetune_data_file
                
                data_file = base_data_file_path

            elif type == 'retrain':

                if cfg.dataset.uncond:

                    data_file = cfg.dataset.retrain_data_file_uncond

                    data_file_3 = base_data_file_path

                else:

                    pos_file = cfg.dataset.retrain_data_file_pos

                    neg_file = cfg.dataset.retrain_data_file_neg
                    
                    data_file = pos_file

                    data_file_2 = neg_file

                    # data_file_3 = cfg.dataset.resinf_train_data_file
                    data_file_3 = base_data_file_path

            elif type == 'test':

                data_file = cfg.dataset.test_data_file


            else:

                if self.mode is None:
                    if self.cfg.general.for_test:
                        data_file = cfg.general.for_test_data_file
                    else:
                        data_file = cfg.general.labeled_data_file
                        
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

        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')

        filename = os.path.join(base_path, data_file)

        if type == 'retrain':

            if cfg.dataset.uncond:

                filename_3 = os.path.join(base_path, data_file_3)
                dataset_3 = torch.load(filename_3)
            else:
            
                filename_2 = os.path.join(base_path, data_file_2)
                dataset_2 = torch.load(filename_2)

                filename_3 = os.path.join(base_path, data_file_3)
                dataset_3 = torch.load(filename_3)


        dataset = torch.load(filename)


        if type == 'givelabel':

            if cfg.dataset.give_baseline_label:

                self.data_list = list(dataset)
            else:

                self.adjs = dataset['As']
                self.traj_hi = None
                self.traj_lo = None
                self.labels = None
                self.ts = None
        else:

            self.adjs = dataset['As'] + dataset_2['As'] if type == 'retrain' and not cfg.dataset.uncond else dataset['As']
            self.traj_hi = dataset['x_reals_his'] + dataset_2['x_reals_his'] if type == 'retrain' and not cfg.dataset.uncond else dataset['x_reals_his']
            self.traj_lo = dataset['x_reals_los'] + dataset_2['x_reals_los'] if type == 'retrain' and not cfg.dataset.uncond else dataset['x_reals_los']
            self.labels = dataset['labels'] + dataset_2['labels'] if type == 'retrain' and not cfg.dataset.uncond else dataset['labels']
            self.ts = dataset['ts'] + dataset_2['ts'] if type == 'retrain' and not cfg.dataset.uncond else dataset['ts']

        if type == 'retrain':

            enhance_len = int(len(self.adjs) * self.cfg.general.enhance_ratio)
            Idx = random.sample(range(len(self.adjs)), k=enhance_len)

        elif type == 'givelabel':

            if not cfg.dataset.give_baseline_label:

                Idx = random.sample(range(len(self.adjs)), k=len(self.adjs))

        
        else:
            
            Idx = random.sample(range(len(self.adjs)), k=len(self.adjs))

        if type == 'givelabel':

            if not cfg.dataset.give_baseline_label:

                self.adjs = [self.adjs[i] for i in Idx]

        else:

           
            self.adjs = [self.adjs[i] for i in Idx]
            self.traj_hi = [self.traj_hi[i] for i in Idx]
            self.traj_lo = [self.traj_lo[i] for i in Idx]
            self.labels = [self.labels[i] for i in Idx]
            self.ts = [self.ts[i] for i in Idx]

        if type == 'retrain':

            if cfg.general.fuse_original:

                self.adjs = self.adjs + dataset_3['As']
                self.traj_hi = self.traj_hi + dataset_3['x_reals_his']
                self.traj_lo = self.traj_lo + dataset_3['x_reals_los']
                self.labels = self.labels + dataset_3['labels']
                self.ts = self.ts + dataset_3['ts']

        if not cfg.dataset.give_baseline_label:
            Idx = random.sample(range(len(self.adjs)), k=len(self.adjs))

        if type == 'givelabel':

            if not cfg.dataset.give_baseline_label:

                self.adjs = [self.adjs[i] for i in Idx]
        else:
            # pass
            self.adjs = [self.adjs[i] for i in Idx]
            self.traj_hi = [self.traj_hi[i] for i in Idx]
            self.traj_lo = [self.traj_lo[i] for i in Idx]
            self.labels = [self.labels[i] for i in Idx]
            self.ts = [self.ts[i] for i in Idx]


        self.traj_min = traj_min
        self.traj_max = traj_max

        print(f'Dataset {filename} loaded from file')



    def __len__(self):

        if self.here_type == 'givelabel':

            if self.cfg.dataset.give_baseline_label:

                return len(self.data_list)
            else:

                return len(self.adjs)
        else:
            
            return len(self.adjs)
    
    
    
    def __getitem__(self, idx):
        
        if self.here_type == 'givelabel':

            if self.cfg.dataset.give_baseline_label:
                
                dt_base = self.data_list[idx]
                n = dt_base.num_nodes
                X = torch.ones(n, 1, dtype=torch.float)
                y = dt_base.y * torch.ones(1, dtype=torch.long).reshape(-1, 1)
                num_nodes = n * torch.ones(1, dtype=torch.long)
                cond_mask = torch.ones(n, 1, dtype=torch.long)
                edge_index = dt_base.edge_index
                edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                edge_attr[:, 1] = 1

                data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                                 y=y, traj_hi=None, traj_lo=None, t=None, idx=idx, n_nodes=num_nodes, cond_mask=cond_mask)
                
            
            else:
                adj = self.adjs[idx]
                n = adj.shape[-1]
                X = torch.ones(n, 1, dtype=torch.float)
                edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                edge_attr[:, 1] = 1
                num_nodes = n * torch.ones(1, dtype=torch.long)
                cond_mask = torch.ones(n, 1, dtype=torch.long)
                y = torch.from_numpy(np.zeros([0])).float().reshape(-1,1)

                data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,y=y,traj_hi=None,traj_lo=None, t=None,
                                                idx=idx, n_nodes=num_nodes, cond_mask=cond_mask)
        else:

            adj = self.adjs[idx]
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            
            y = torch.from_numpy(np.array([self.labels[idx]])).float().reshape(-1,1)
           
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            cond_mask = torch.ones(n, 1, dtype=torch.long)

            traj_hi = self.traj_hi[idx]
            traj_hi[np.abs(traj_hi) < 1e-5] = 0
            traj_hi = (traj_hi - self.traj_min) / (self.traj_max - self.traj_min)
            traj_hi = traj_hi[:, :self.ode_use_length + 1]
            

            traj_hi_fused = network_features(adj, traj_hi)
            traj_lo = self.traj_lo[idx]
            traj_lo[np.abs(traj_lo) < 1e-5] = 0
            traj_lo = (traj_lo - self.traj_min) / (self.traj_max - self.traj_min)
            traj_lo = traj_lo[:, :self.ode_use_length + 1]
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

class FullResiGuideTrajDatasetDebug(Dataset):
    def __init__(self, mech, cfg, path, mode=None, traj_min=None, traj_max=None, type=None):
        """ This class can be used to load the comm20, sbm and planar datasets. """

        self.cfg = cfg
        self.mode = mode
        self.ode_use_length = cfg.train.ode_use_length
        self.path = path
        if mech == 1:
            raise NotImplementedError
        
        elif mech == 2:
        
            raise NotImplementedError
        
        elif mech == 3:
            
            # if type == 'ode_train':

                # data_file = 


            if type == 'retrain':

                data_file = cfg.general.retrain_data_file_pos

                data_file_2 = cfg.general.retrain_data_file_neg

            elif type == 'test':

                data_file = cfg.general.test_data_file

            else:

                if self.mode is None:
                    if self.cfg.general.for_test:
                        data_file = cfg.general.for_test_data_file
                    else:
                        data_file = self.path
                        
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

        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')

        filename = os.path.join(base_path, data_file)

        if type == 'retrain':
            
            filename_2 = os.path.join(base_path, data_file_2)
            dataset_2 = torch.load(filename_2)


        dataset = torch.load(filename)

        self.adjs = dataset['As'] + dataset_2['As'] if type == 'retrain' else dataset['As']
        self.traj_hi = dataset['x_reals_his'] + dataset_2['x_reals_his'] if type == 'retrain' else dataset['x_reals_his']
        self.traj_lo = dataset['x_reals_los'] + dataset_2['x_reals_los'] if type == 'retrain' else dataset['x_reals_los']
        self.labels = dataset['labels'] + dataset_2['labels'] if type == 'retrain' else dataset['labels']
        self.ts = dataset['ts'] + dataset_2['ts'] if type == 'retrain' else dataset['ts']

        if type == 'debug_train':
            
            selected_ratio = self.cfg.general.debug_train_ratio
            Idx = random.sample(range(len(self.adjs)), k=int(len(self.adjs) * selected_ratio))
            print('Traing length', len(Idx))

            self.adjs = [self.adjs[i] for i in Idx]
            self.traj_hi = [self.traj_hi[i] for i in Idx]
            self.traj_lo = [self.traj_lo[i] for i in Idx]
            self.labels = [self.labels[i] for i in Idx]
            self.ts = [self.ts[i] for i in Idx]



        self.traj_min = traj_min
        self.traj_max = traj_max

        print(f'Dataset {filename} loaded from file')



    def __len__(self):
        return len(self.adjs)
    
    
    
    def __getitem__(self, idx):
        
        adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        y = torch.from_numpy(np.array([self.labels[idx]])).float().reshape(-1,1)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        cond_mask = torch.ones(n, 1, dtype=torch.long)

        traj_hi = self.traj_hi[idx]
        traj_hi = (traj_hi - self.traj_min) / (self.traj_max - self.traj_min)
        traj_hi = traj_hi[:, :self.ode_use_length + 1]
        

        traj_hi_fused = network_features(adj, traj_hi)
        traj_lo = self.traj_lo[idx]
        traj_lo = (traj_lo - self.traj_min) / (self.traj_max - self.traj_min)
        traj_lo = traj_lo[:, :self.ode_use_length + 1]
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

# class FullResiGuideTrajDataset_debug(Dataset):

#     def __init__(self, )

    

class SemiResiGuideTrajDataset(Dataset):
    def __init__(self, mech, cfg, type, mode=None, traj_min=None, traj_max=None):
        """ This class can be used to load the comm20, sbm and planar datasets. """

        self.cfg = cfg
        self.mode = mode
        self.ode_use_length = cfg.train.ode_use_length
        self.type = type

        if mech == 1:
            raise NotImplementedError
        
        elif mech == 2:
            raise NotImplementedError
        
        elif mech == 3:
            if self.mode is None:
                if self.cfg.general.for_test:
                    data_file = cfg.general.for_test_data_file
                else:
                    data_file = cfg.dataset.semi_data_file
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

        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')

        filename = os.path.join(base_path, data_file)

        dataset = torch.load(filename)

        self.adjs = dataset['As']
        self.traj_hi = dataset['x_reals_his']
        self.traj_lo = dataset['x_reals_los']


        self.traj_min = traj_min
        self.traj_max = traj_max




        self.labels = dataset['labels']
        self.ts = dataset['ts']

        

        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)
    
    def __getitem__(self, idx):
        
        adj = self.adjs[idx]
        n = adj.shape[-1]
        X = torch.ones(n, 1, dtype=torch.float)
        
        y = torch.from_numpy(np.array([self.labels[idx]])).float().reshape(-1,1)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        cond_mask = torch.ones(n, 1, dtype=torch.long)

        traj_hi = self.traj_hi[idx]
        traj_hi[np.abs(traj_hi) < 1e-5] = 0.
        traj_hi = (traj_hi - self.traj_min) / (self.traj_max - self.traj_min)

        traj_hi = traj_hi[:, :self.ode_use_length + 1]

        traj_hi_fused = network_features(adj, traj_hi)
        traj_lo = self.traj_lo[idx]
        traj_lo[np.abs(traj_lo) < 1e-5] = 0.
        traj_lo = (traj_lo - self.traj_min) / (self.traj_max - self.traj_min)
        
        traj_lo = traj_lo[:, :self.ode_use_length + 1]
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


class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        super().__init__(cfg)
        self.n_graphs = n_graphs
        self.prepare_data()
        self.inner = self.train_dataloader()
        self.cfg = cfg

    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self, graphs, cond_graphs=None, type=None):

        if type == 'ode_train':
            
            test_len = 1
            val_len = int(len(graphs) * 0.2)
            train_len = len(graphs) - test_len - val_len
            splits = random_split(graphs, [train_len, val_len, test_len])
            datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        
        elif type == 'resinf_train':

            assert len(graphs) == 2

            test_len = 1
            val_len = len(graphs[1]) - 1
            train_len = int(len(graphs[0]))

            splits = random_split(graphs[1], [val_len, test_len])

            datasets = {'train': graphs[0], 'val': splits[0], 'test': splits[1]}
        
        elif type == 'finetune':

            assert len(graphs) == 2

            test_len = 1
            val_len = len(graphs[1]) - 1

            train_len = int(len(graphs[0]))

            splits = random_split(graphs[1], [val_len, test_len])

            datasets = {'train': graphs[0], 'val': splits[0], 'test': splits[1]}

        elif type == 'retrain':

            assert len(graphs) == 2
            test_len = 1
            val_len = len(graphs[1]) - 1

            train_len = int(len(graphs[0]))

            splits = random_split(graphs[1], [val_len, test_len])

            datasets = {'train': graphs[0], 'val': splits[0], 'test': splits[1]}
            


        elif type == 'test' or type == 'givelabel' or type == 'self_training':
            
            test_len = int(len(graphs))
            val_len = int(len(graphs))
            train_len = int(len(graphs))
            datasets = {'train': graphs, 'val': graphs, 'test': graphs}

        else:
        
            if isinstance(graphs, list):

                if len(graphs) == 3:
                    test_len = int(len(graphs[2]))
                    val_len = int(len(graphs[1]))
                    train_len = int(len(graphs[0]))
                    datasets = {'train': graphs[0], 'val': graphs[1], 'test': graphs[2]}
                elif len(graphs) == 2:

                    print('Right Split.')
                    train_len = int(len(graphs[0]))
                    # test_len = round(int(len(graphs[0]) + len(graphs[1]))*0.2)
                    test_len = 1
                    val_len = len(graphs[1]) - test_len
                    splits = random_split(graphs[1], [val_len, test_len], generator=torch.Generator().manual_seed(3407))
                    datasets = {'train': graphs[0], 'val': splits[0], 'test': splits[1]}
                    

            else:
                if self.cfg.general.for_test:
                    train_len = 1
                    val_len = 1
                    test_len = len(graphs) - train_len - val_len
                else:
                    # test_len = int(round(len(graphs) * 0.2))
                    # train_len = int(round((len(graphs) - test_len) * 0.8))
                    # val_len = len(graphs) - train_len - test_len

                    test_len = int(round(len(graphs) * self.cfg.general.test_ratio))
                    train_len = int(round(len(graphs) * self.cfg.general.train_ratio))
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
            super().prepare_data(datasets, type=type)


class ResiDataModule(SpectreGraphDataModule):
    def __init__(self, cfg, n_graphs=200, mech=1):
        self.mech = mech
        self.cfg = cfg
        super().__init__(cfg, n_graphs)
    def prepare_data(self):
        graphs = ResiDataset(self.mech)
        cond_graphs = None
        return super().prepare_data(graphs, cond_graphs)
    
class ResiGuideDataModule(SpectreGraphDataModule):
    def __init__(self, cfg, type, n_graphs=200, mech=1):
        self.mech = mech
        self.cfg = cfg
        self.type = type
        super().__init__(cfg, n_graphs)

    def prepare_data(self):
        
        if self.cfg.dataset.pregene:
            if self.type == 'ode_train':
                graphs = SemiResiGuideTrajDataset(self.mech, self.cfg, type='ode_train', traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max)
            elif self.type == 'resinf_train':
                graphs0 = FullResiGuideTrajDataset(self.mech, self.cfg, type='resinf_train', traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max)
                graphs1 = FullResiGuideTrajDataset(self.mech, self.cfg, type='test', traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max)

                graphs = [graphs0, graphs1]

            elif self.type == 'finetune':
                graphs0 = FullResiGuideTrajDataset(self.mech, self.cfg, type='finetune', traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max)
                graphs1 = FullResiGuideTrajDataset(self.mech, self.cfg, type='test', traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max)
                graphs = [graphs0, graphs1]
            elif self.type == 'test':
                graphs = FullResiGuideTrajDataset(self.mech, self.cfg, type='test', traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max)

            elif self.type == 'self_training':

                graphs = FullResiGuideTrajDataset(self.mech, self.cfg, type='self_training', traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max)

            elif self.type == 'givelabel':
                graphs = FullResiGuideTrajDataset(self.mech, self.cfg, type='givelabel', traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max)
            
            elif self.type == 'retrain':
                graphs0 = FullResiGuideTrajDataset(self.mech, self.cfg, type='retrain', traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max)
                graphs1 = FullResiGuideTrajDataset(self.mech, self.cfg, type='test', traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max)

                graphs = [graphs0, graphs1]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            # graphs = ResiGuideDataset(self.mech, self.cfg.general.datapath)
        
        cond_graphs = None

        return super().prepare_data(graphs, cond_graphs, self.type)


class ResiGuideDataModuleDebug(SpectreGraphDataModule):
    def __init__(self, cfg, type, n_graphs=200, mech=1):
        self.mech = mech
        self.cfg = cfg
        self.type = type
        super().__init__(cfg, n_graphs)
    def prepare_data(self):
        if self.cfg.dataset.pregene:
            # if self.type == 'semi':
                
            # elif self.type == 'full':
            graphs_train = FullResiGuideTrajDatasetDebug(self.mech, self.cfg, self.cfg.general.debug_path_train, traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max, type='debug_train')
            # elif self.type == 'test':
                # graphs = FullResiGuideTrajDataset(self.mech, self.cfg, type='test', traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max)
            # elif self.type == 'retrain':
            graphs_valtest = FullResiGuideTrajDatasetDebug(self.mech, self.cfg, self.cfg.general.debug_path_val, traj_min=self.cfg.dataset.traj_min, traj_max=self.cfg.dataset.traj_max, type='debug_test')
            # else:
            #     raise NotImplementedError
        else:
            raise NotImplementedError
            # graphs = ResiGuideDataset(self.mech, self.cfg.general.datapath)
        
        cond_graphs = None
        return super().prepare_data([graphs_train, graphs_valtest], cond_graphs, self.type)




    
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


