from ode.diffusion.distributions import DistributionNodes
import ode.utils as utils
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from ode.datasets.utils import SubsetRamdomSampler, SubsetSequentialSampler


class AbstractDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataloaders = None
        self.input_dims = None
        self.output_dims = None

    def prepare_data(self, datasets, datasets_cond=None, type=None) -> None:
        
        if type == 'ode_train':

            batch_size = self.cfg.train.ode_train_batch_size

        elif type == 'resinf_train':

            batch_size = self.cfg.train.resinf_train_batch_size

        elif type == 'finetune':

            batch_size = self.cfg.train.finetune_batch_size

        else:

            batch_size = self.cfg.train.batch_size
        
        num_workers = self.cfg.train.num_workers
        
        if 'num_env' in self.cfg.train:
            minibatch_size = self.cfg.train.minibatch_size
            n_env = self.cfg.train.num_env
            sampler_train = SubsetRamdomSampler(indices=datasets['train'].indices, minibatch_size=minibatch_size, same_order_in_groups=False)
            sampler_val = SubsetRamdomSampler(indices=datasets['val'].indices, minibatch_size=minibatch_size, same_order_in_groups=False)
            sampler_test = SubsetRamdomSampler(indices=datasets['test'].indices, minibatch_size=minibatch_size, same_order_in_groups=False)
            samplers = {'train': sampler_train, 'val': sampler_val, 'test': sampler_test}
            dataloader_params = {
                'batch_size': minibatch_size * n_env,
                'num_workers': 8, 
                'pin_memory': True,
                'drop_last': False,
                'shuffle': False
            }

            self.dataloaders = {'train': DataLoader(**dataloader_params, dataset=datasets['train'], sampler=samplers['train']),
                                'val': DataLoader(**dataloader_params, dataset=datasets['val'], sampler=samplers['val']),
                                'test': DataLoader(**dataloader_params, dataset=datasets['test'], sampler=samplers['test'])}
            
        else:
            self.dataloaders = {split: DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                shuffle='debug' not in self.cfg.general.name)
                                for split, dataset in datasets.items()}
            
        if datasets_cond is not None:
            self.dataloaders_cond = {split: DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                         shuffle='debug' not in self.cfg.general.name)
                                        for split, dataset in datasets_cond.items()}
        else:
            self.dataloaders_cond = None
    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        if self.cfg.general.conditional:
            return [self.dataloaders["val"], self.dataloaders_cond["val"]]
        else:
            return self.dataloaders["val"]

    def test_dataloader(self):
        if self.cfg.general.conditional:
            return [self.dataloaders["test"], self.dataloaders_cond["test"]]
        else:
            return self.dataloaders["test"]

    def __getitem__(self, idx):
        return self.dataloaders['train'][idx]

    def node_counts(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts
    
    def node_counts_pair(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                data = data[1]
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_counts_val(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['val']:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_counts_test(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['test']:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for data in self.dataloaders['train']:
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.dataloaders['train']:
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.Tensor(num_classes)

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)

                all_pairs = 0
                for count in counts:
                    all_pairs += count * (count - 1)

                num_edges = data.edge_index.shape[1]
                num_non_edges = all_pairs - num_edges

                edge_types = data.edge_attr.sum(dim=0)
                assert num_non_edges >= 0
                d[0] += num_non_edges
                d[1:] += edge_types[1:]

        d = d / d.sum()
        return d

    def edge_counts_pair(self):
        num_classes = None
        for data in self.dataloaders['train']:
            data = data[1]
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.Tensor(num_classes)

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                data = data[1]
                unique, counts = torch.unique(data.batch, return_counts=True)

                all_pairs = 0
                for count in counts:
                    all_pairs += count * (count - 1)

                num_edges = data.edge_index.shape[1]
                num_non_edges = all_pairs - num_edges

                edge_types = data.edge_attr.sum(dim=0)
                assert num_non_edges >= 0
                d[0] += num_non_edges
                d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        multiplier = torch.Tensor([0, 1, 2, 3, 1.5])

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                n = data.x.shape[0]

                for atom in range(n):
                    edges = data.edge_attr[data.edge_index[0] == atom]
                    edges_total = edges.sum(dim=0)
                    valency = (edges_total * multiplier).sum()
                    valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types, n_nodes_val=None, n_nodes_test=None):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)
        self.n_nodes_val = DistributionNodes(n_nodes_val)
        self.n_nodes_test = DistributionNodes(n_nodes_test)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        self.input_dims = {'X': example_batch['x'].size(1),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {'X': example_batch['x'].size(1),
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}
