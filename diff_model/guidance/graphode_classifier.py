from typing import Dict
import pytorch_lightning as pl
import sys
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
# sys.path.append('..')
# sys.path.append('/data3/liuchang/workspace/data_enhance')
from ode.models.model_libs.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete ,MarginalUniformTransition
import torch as th
from torch import nn
import torchdiffeq as thd
import numpy as np
import utils
from diffusion import diffusion_utils
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score,roc_curve, auc
from ode.models.model_libs.odeblock import *
from utils import *
# from ode.models.model_libs.classifier import *
from ode.models.model_libs.classifier_batchedresinf import ResilienceClassifier
import random

class GraphODEClassifier(pl.LightningModule):
    EPSILON = 1e-10
    def __init__(self,
                 config,
                 dataset_infos,
                 extra_features,
                 domain_features,
                 lr = 1e-3, 
                 input_dim = 1,
                 output_dim = 1,
                 stable_coef = 1.0,
                 resi_coef = 1.0,):
        super().__init__()
        self.config = config
        self.lr = lr
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stable_coef = stable_coef
        self.dynamic_coef = config.train['dynamic_coef']
        self.resi_coef = resi_coef
        self.resilience_threshold = config.train['resilience_threshold']
        self.is_test = False
        

        self.dataset_info = dataset_infos
        self.extra_features = extra_features
        self.domain_features = domain_features

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist


        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist
        
        # if self.conditional:
        self.node_dist_conditional_val = dataset_infos.n_nodes_val
        self.node_dist_conditional_test = dataset_infos.n_nodes_test

        self.classification_steps = config.train.classification_steps
        self.num_traj = config.train.num_traj

        self.loss = nn.L1Loss()
        self.resilience_loss = nn.BCELoss()

        self.stable_window = config.train['window']
        self.T = config.train['T']
        self.time_ticks = config.train['time_ticks']
        self.ode_use_length = config.train['ode_use_length']
        self.diffusion_steps = config.model.diffusion_steps
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(config.model.diffusion_noise_schedule, timesteps=config.model.diffusion_steps)
        # self.is_trm = config.train.is_trm
        
        self.counttt = 0

        if config.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = th.ones(self.Xdim_output) / self.Xdim_output
            e_limit = th.ones(self.Edim_output) / self.Edim_output
            y_limit = th.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
            x_paddings = th.tensor([1.], device=x_marginals.device)
            e_paddings = th.tensor([1.0, 0.0], device=e_marginals.device)
            y_paddings = th.tensor([], device=x_marginals.device)
            self.padding_dist = utils.PlaceHolder(X=x_paddings, E=e_paddings,y=y_paddings)

        elif config.model.transition == 'marginal':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / th.sum(node_types)
            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / th.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=th.ones(self.ydim_output) / self.ydim_output)
            # print('Y::')
            # print(th.ones(self.ydim_output) / self.ydim_output)
            # assert False
            x_paddings = th.tensor([1.], device=x_marginals.device)
            e_paddings = th.tensor([1.0, 0.0], device=e_marginals.device)
            y_paddings = th.tensor([], device=x_marginals.device)
            self.padding_dist = utils.PlaceHolder(X=x_paddings, E=e_paddings,y=y_paddings)
        
        if self.config.model.extra_features == 'all':
            # print(type(self.config.train))
            self.config.train._set_flag("struct", False)
            self.config.train['extra_dim'] = 6
            self.config.train._set_flag("struct", True)
        else:
            raise RuntimeError('Unknown extra data dimensions.')

        self.encoder = self.create_encoder(self.config['train'])
        self.ode = self.create_ode(self.config['train'])
        self.decoder = self.create_decoder(self.config['train'])
        self.classifier = self.create_classifier(self.config['train'])

        self.update_epsilon_every = config.train['update_epsilon_every']

    def create_encoder(self, config: Dict):
        encoder = nn.Sequential()
        for i in range(config['num_encoder_layers']):
            if i == 0:
                encoder.add_module(f'fc-{i}',
                                   nn.Linear(self.input_dim, config['node_dim']))
            else:
                encoder.add_module(f'fc-{i}',
                                   nn.Linear(config['node_dim'], config['node_dim']))
            encoder.add_module('act', nn.Softplus())
        return encoder
    
    def create_ode(self, config: Dict):
        return ODEBlock(config)
    
    def create_decoder(self, specs: Dict):
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
    

    def create_classifier(self, specs: Dict):

        classifier = ResilienceClassifier(specs)

        return classifier
    
    def transfer_to_adjacent(self, X: th.Tensor, E: th.Tensor):

        A = E[:, :, :, 1:].sum(dim=-1).float()
        return A

    @property
    def latent_dim(self):
        return self.specs['node_dim']
    
    @th.enable_grad()
    def model_forward(self, X, E, t, node_mask, noisy_data):

        self.mode = 'test'
        self.is_test = True
        self.ode.update_epsilon(0)

        extra_data = self.compute_extra_data(noisy_data)

        t_eval = th.linspace(0, self.T, self.time_ticks).to(X.device)

        t_eval = t_eval / self.T

        t_eval = t_eval[:self.ode_use_length + 1]

        B, N = E.shape[0], E.shape[1]
        traj_min = self.config.dataset['traj_min']
        traj_max = self.config.dataset['traj_max']

        if self.config.dataset.dyna == 'regulatory':
            inte = random.randint(1, 5)
            x_0_lo = inte * th.ones((B, N, self.ode_use_length + 1, 1)).to(E.device)
        else:
            x_0_lo = th.zeros((B, N, self.ode_use_length + 1, 1)).to(E.device)
        x_0_lo = (x_0_lo - traj_min) / (traj_max - traj_min)

        x_0_lo_feat = th.zeros((B, N, self.ode_use_length + 1, 2)).to(E.device)
        x_0_lo = th.cat([x_0_lo, x_0_lo_feat], dim=-1)

        if self.config.dataset.dyna == 'regulatory':

            inte = random.randint(1, 5)
            x_0_hi = inte * th.ones((B, N, self.ode_use_length + 1, 1)).to(E.device)
            
        else:
            x_0_hi = 5 * th.ones((B, N, self.ode_use_length + 1, 1)).to(E.device)

        x_0_hi = (x_0_hi - traj_min) / (traj_max - traj_min)

        x_0_hi_feat = th.zeros((B, N, self.ode_use_length + 1, 2)).to(E.device)
        x_0_hi = th.cat([x_0_hi, x_0_hi_feat], dim=-1)

        x_hat_lo = self.forward(x_0_lo.to(th.float32), X, E, t_eval, extra_data) # (B, N, T)
        x_hat_hi = self.forward(x_0_hi.to(th.float32), X, E, t_eval, extra_data) # (B, N, T)
        
        x_hat_lo_clasf_input = x_hat_lo[:, :, 1:self.ode_use_length + 1].unsqueeze(2) # (B, N, 1, T)
        x_hat_hi_clasf_input = x_hat_hi[:, :, 1:self.ode_use_length + 1].unsqueeze(2) # (B, N, 1, T)

        clasf_feat = th.cat([x_hat_hi_clasf_input, x_hat_lo_clasf_input], dim=2)
        
        adj = self.transfer_to_adjacent(X, E)

        y_hat = self.classifier(clasf_feat, adj, node_mask) # (B, 1)

        return y_hat, x_hat_hi, x_hat_lo


    def forward(self, x, X_in, E_in, t_eval, extra_data):

        node_features = extra_data.X  # (bs, n, f)
        node_features = node_features.unsqueeze(2)
        node_features = node_features.repeat(1, 1, self.ode_use_length + 1, 1)
        
        if self.config.general.use_encoder:
            x = self.encoder(x)
            self.ode.ode_func.update_graph(X_in, E_in)
            x = th.cat([x, node_features], dim=-1)
            # print(x.shape)
            # assert False
            x = self.ode(t_eval, x)
            x = self.decoder(x)
        else:
            self.ode.ode_func.update_graph(X_in, E_in)
            x = self.ode(t_eval, x)
            x = x.flatten(start_dim=2)
        return x
    
    def unwrapped_forward(self, noisy_data, extra_data, node_mask, t_eval, x, type):

        '''
        node_mask: (B, N)
        x: List of (B, N, T)
        '''
        num_stable_steps = self.stable_window
        
        X = noisy_data['X_t']
        E = noisy_data['E_t']
        t = noisy_data['t']
        y = noisy_data['y_t']
        B, N = E.shape[0], E.shape[1]
        

        x_hat_hi = self.forward(x['hi'].to(th.float32), X, E, t_eval,extra_data) # (B, N, T)
        x_hat_lo = self.forward(x['lo'].to(th.float32), X, E, t_eval, extra_data) # (B, N, T)
        
        # if self.config.general.for_test or self.is_test:
        
        #     th.save(x['lo'].to(th.float32), f'intermediate_results/x_real_lo_{self.counttt}.pt')
        #     th.save(x['hi'].to(th.float32), f'intermediate_results/x_real_hi_{self.counttt}.pt')
        #     th.save(x_hat_lo, f'intermediate_results/x_hat_lo_{self.counttt}.pt')
        #     th.save(x_hat_hi, f'intermediate_results/x_hat_hi_{self.counttt}.pt')
            
        #     self.counttt = self.counttt + 1

        if type == 'full':
            
            x_hat_lo_clasf_input = x_hat_lo[:, :, 1:self.ode_use_length + 1]
            x_hat_hi_clasf_input = x_hat_hi[:, :, 1:self.ode_use_length + 1]

            if self.is_trm:
                clasf_feat = th.cat([x_hat_lo_clasf_input, x_hat_hi_clasf_input], dim=1) # (B, 2N, T)
            else:
                clasf_feat = th.cat([x_hat_lo_clasf_input, x_hat_hi_clasf_input], dim=2) # (B, N, 2T)

            adj = self.transfer_to_adjacent(X, E)

            y_hat = self.classifier(clasf_feat, adj, node_mask, t) # (B, 1)

            if self.config.general.for_test or self.is_test:
                th.save(y_hat.cpu().numpy(), f'intermediate_results/y_pred.pt')
                th.save(y.cpu().numpy(), f'intermediate_results/y_true.pt')
        
            resilience_loss = self.resilience_loss(y_hat, y)

            if self.training:
                resi_metrics = {'auc': 0, 'f1': 0}
            else:

                try:
                   
                    fpr, tpr, thresholds = roc_curve(y.cpu().numpy(), y_hat.cpu().numpy(), pos_label=1)
                    resi_aucs = auc(fpr, tpr)

                except:

                    print('AUC calculation goes wrong. The AUC is not meaningful anymore.')
                    print('AUC calculation goes wrong. The AUC is not meaningful anymore.')
                    print('AUC calculation goes wrong. The AUC is not meaningful anymore.')
                    resi_aucs = 0
                
                resi_f1s = f1_score(y.cpu().numpy(), y_hat.cpu().numpy() > self.resilience_threshold)
                resi_acc = accuracy_score(y.cpu().numpy(), y_hat.cpu().numpy() > self.resilience_threshold)

                resi_rates = y.cpu().sum()/y.cpu().shape[0]

                resi_metrics = {'auc': resi_aucs, 'f1': resi_f1s, 'resi_rate':resi_rates, 'acc': resi_acc}
        
        else:
            resi_metrics = {'auc': 0, 'f1': 0, 'acc': 0, 'resi_rate': 0}

            resilience_loss = th.FloatTensor([0.0]).to(x_hat_lo.device)

            y_hat = th.zeros_like(y)

            self.resi_coef = 0.0

        
        if len(x['lo'].shape) > 3:

            # dynamic_loss = self.loss(x_hat_lo.reshape(-1, self.time_ticks)[:, :], x['lo'][..., 0].reshape(-1, self.time_ticks)[:, :-num_stable_steps]) + \
            #     self.loss(x_hat_hi.reshape(-1, self.time_ticks)[:, :-num_stable_steps], x['hi'][..., 0].reshape(-1, self.time_ticks)[:, :-num_stable_steps])
            
            dynamic_loss = self.loss(x_hat_lo.reshape(-1, self.ode_use_length + 1), x['lo'][..., 0].reshape(-1, self.ode_use_length + 1)) + \
                self.loss(x_hat_hi.reshape(-1, self.ode_use_length + 1), x['hi'][..., 0].reshape(-1, self.ode_use_length + 1))

            # stable_loss = self.loss(x_hat_lo.reshape(-1, self.time_ticks)[:, -num_stable_steps:], x['lo'][..., 0].reshape(-1, self.time_ticks)[:, -num_stable_steps:]) + \
            #     self.loss(x_hat_hi.reshape(-1, self.time_ticks)[:, -num_stable_steps:], x['hi'][..., 0].reshape(-1, self.time_ticks)[:, -num_stable_steps:])

            stable_loss = th.FloatTensor([0.0]).to(x_hat_lo.device)
            
            loss = self.dynamic_coef * dynamic_loss + self.stable_coef * stable_loss + self.resi_coef * resilience_loss


            
            relative_error = (th.mean(
                th.abs(x_hat_lo - x['lo'][..., 0]) / th.where(th.gt(th.abs(x['lo'][..., 0]), self.EPSILON), th.abs(x['lo'][..., 0]), th.ones_like(x['lo'][..., 0]))) + \
                th.mean(th.abs(x_hat_hi - x['hi'][..., 0]) / th.where(th.gt(th.abs(x['hi'][..., 0]), self.EPSILON), th.abs(x['hi'][..., 0]), th.ones_like(x['hi'][..., 0]))))/2
            
            
            # dynamic_relative_error = (th.mean(
            #     th.abs(x_hat_lo[:, :-num_stable_steps] - x['lo'][..., 0][:, :-num_stable_steps]) /
            #     th.where(th.gt(th.abs(x['lo'][..., 0][:, :-num_stable_steps]), self.EPSILON),
            #             th.abs(x['lo'][..., 0][:, :-num_stable_steps]),
            #             th.ones_like(x['lo'][..., 0][:, :-num_stable_steps]))) + \
            #             th.mean(
            #                 th.abs(x_hat_hi[:, :-num_stable_steps] - x['hi'][..., 0][:, :-num_stable_steps]) /
            #                 th.where(th.gt(th.abs(x['hi'][..., 0][:, :-num_stable_steps]), self.EPSILON),
            #                         th.abs(x['hi'][..., 0][:, :-num_stable_steps]),
            #                         th.ones_like(x['hi'][..., 0][:, :-num_stable_steps])))) / 2

            dynamic_relative_error = relative_error
            
            # stable_relative_error = (th.mean(
            #     th.abs(x_hat_lo[:, -num_stable_steps:] - x['lo'][..., 0][:, -num_stable_steps:]) /
            #     th.where(th.gt(th.abs(x['lo'][..., 0][:, -num_stable_steps:]), self.EPSILON),
            #             th.abs(x['lo'][..., 0][:, -num_stable_steps:]),
            #             th.ones_like(x['lo'][..., 0][:, -num_stable_steps:]))) + \
            #              th.mean(
            #                 th.abs(x_hat_hi[:, -num_stable_steps:] - x['hi'][..., 0][:, -num_stable_steps:]) /
            #                 th.where(th.gt(th.abs(x['hi'][..., 0][:, -num_stable_steps:]), self.EPSILON),
            #                         th.abs(x['hi'][..., 0][:, -num_stable_steps:]),
            #                         th.ones_like(x['hi'][..., 0][:, -num_stable_steps:]))))/2

            stable_relative_error = th.FloatTensor([0.0]).to(x_hat_lo.device)

        else:

            # dynamic_loss = self.loss(x_hat_lo.reshape(-1, self.time_ticks)[:, :-num_stable_steps], x['lo'].reshape(-1, self.time_ticks)[:, :-num_stable_steps]) + \
            #     self.loss(x_hat_hi.reshape(-1, self.time_ticks)[:, :-num_stable_steps], x['hi'].reshape(-1, self.time_ticks)[:, :-num_stable_steps])

            dynamic_loss = self.loss(x_hat_lo.reshape(-1, self.ode_use_length + 1), x['lo'].reshape(-1, self.ode_use_length + 1)) + \
                self.loss(x_hat_hi.reshape(-1, self.ode_use_length + 1), x['hi'].reshape(-1, self.ode_use_length + 1))

            # stable_loss = self.loss(x_hat_lo.reshape(-1, self.time_ticks)[:, -num_stable_steps:], x['lo'].reshape(-1, self.time_ticks)[:, -num_stable_steps:]) + \
            #     self.loss(x_hat_hi.reshape(-1, self.time_ticks)[:, -num_stable_steps:], x['hi'].reshape(-1, self.time_ticks)[:, -num_stable_steps:])

            stable_loss = th.FloatTensor([0.0]).to(x_hat_lo.device)
            
            loss = self.dynamic_coef * dynamic_loss + self.stable_coef * stable_loss + self.resi_coef * resilience_loss

            relative_error = (th.mean(
                th.abs(x_hat_lo - x['lo']) / th.where(th.gt(th.abs(x['lo']), self.EPSILON), th.abs(x['lo']), th.ones_like(x['lo']))) + \
                    th.mean(
                        th.abs(x_hat_hi - x['hi']) / th.where(th.gt(th.abs(x['hi']), self.EPSILON), th.abs(x['hi']), th.ones_like(x['hi']))))/2
            
            
            # dynamic_relative_error = (th.mean(
            #     th.abs(x_hat_lo[:, :-num_stable_steps] - x['lo'][:, :-num_stable_steps]) /
            #     th.where(th.gt(th.abs(x['lo'][:, :-num_stable_steps]), self.EPSILON),
            #             th.abs(x['lo'][:, :-num_stable_steps]),
            #             th.ones_like(x['lo'][:, :-num_stable_steps]))) + \
            #             th.mean(
            #                 th.abs(x_hat_hi[:, :-num_stable_steps] - x['hi'][:, :-num_stable_steps]) /
            #                 th.where(th.gt(th.abs(x['hi'][:, :-num_stable_steps]), self.EPSILON),
            #                         th.abs(x['hi'][:, :-num_stable_steps]),
            #                         th.ones_like(x['hi'][:, :-num_stable_steps]))))/2

            dynamic_relative_error = relative_error
            
            # stable_relative_error = (th.mean(
            #     th.abs(x_hat_lo[:, -num_stable_steps:] - x['lo'][:, -num_stable_steps:]) /
            #     th.where(th.gt(th.abs(x['lo'][:, -num_stable_steps:]), self.EPSILON),
            #             th.abs(x['lo'][:, -num_stable_steps:]),
            #             th.ones_like(x['lo'][:, -num_stable_steps:]))) + \
            #              th.mean(
            #                 th.abs(x_hat_hi[:, -num_stable_steps:] - x['hi'][:, -num_stable_steps:]) /
            #                 th.where(th.gt(th.abs(x['hi'][:, -num_stable_steps:]), self.EPSILON),
            #                         th.abs(x['hi'][:, -num_stable_steps:]),
            #                         th.ones_like(x['hi'][:, -num_stable_steps:])))) / 2
            stable_relative_error = th.FloatTensor([0.0]).to(x_hat_lo.device)
    
        noisy_data['y_hat'] = y_hat.cpu()
        
        return loss, dynamic_loss, stable_loss, resilience_loss, relative_error, \
            dynamic_relative_error, stable_relative_error, resi_metrics, noisy_data
    
    

    def pre_noise(self, X, E, y, node_mask, t, x_reals_hi, x_reals_lo):

        noisy_data = {'t': t, 'X_t': X, 'E_t': E, 'y_t': y, 'node_mask': node_mask, 'real_traj': {'hi': x_reals_hi, 'lo': x_reals_lo}}

        return noisy_data

    
    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = th.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = th.cat((extra_features.E, extra_molecular_features.E), dim=-1)

        t = noisy_data['t']
        
        return utils.PlaceHolder(X=extra_X, E=extra_E, y=t)

    def training_step(self, data, i):
        
        self.is_test = False
        

        if self.config.dataset.pregene:
            dense_data, node_mask = utils.to_dense_guide(data.x, data.edge_index, data.edge_attr, data.batch, data.traj_hi, data.traj_lo, data.t)
        else:
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        if data.y[0] == -1:
            here_type = 'semi'
            if self.global_step == 1:
                print('Training ODE only......')
        else:
            here_type = 'full'
            if self.global_step == 1:
                print('Training full model......')

        t_eval = th.linspace(0, self.T, self.time_ticks).to(X.device)

        t_eval = t_eval / self.T

        t_eval = t_eval[:self.ode_use_length + 1]
                
        if self.config.dataset.pregene:
            noisy_data = self.pre_noise(X, E, data.y, node_mask, dense_data.t, dense_data.traj_hi, dense_data.traj_lo)
        else:
            noisy_data = self.apply_noise(X, E, data.y, node_mask, t_eval)

        extra_data = self.compute_extra_data(noisy_data)

        batch_size = X.shape[0]

        loss, dynamic_loss, stable_loss, resilience_loss, relative_error, dynamic_relative_error, stable_relative_error, resi_metrics,_ = \
            self.unwrapped_forward(noisy_data, extra_data, node_mask, t_eval, noisy_data['real_traj'], here_type)

        self.log('train_loss', loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dynamic_loss', dynamic_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_stable_loss', stable_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)        
        self.log('train_resilience_loss', resilience_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_relative_error', relative_error,
                 batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dynamic_relative_error', dynamic_relative_error,
                 batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_stable_relative_error', stable_relative_error,
                 batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        
        if self.global_step % self.update_epsilon_every == 0:

            print(self.global_step)
            self.ode.update_epsilon(self.ode.epsilon * self.ode.epsilon)
            print('Epsilon updated to:', self.ode.epsilon)

        return {'loss': loss, 'dynamic_loss': dynamic_loss, 'stable_loss': stable_loss, 'resilience_loss': resilience_loss, 'relative_error': relative_error, 'dynamic_relative_error': dynamic_relative_error, 'stable_relative_error': stable_relative_error, 'resi_metrics': resi_metrics}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        losses = []
        dynamic_losses = []
        stable_losses = []
        resilience_losses = []
        relative_errors = []
        dynamic_relative_errors = []
        stable_relative_errors = []
        for output in outputs:
            losses.append(output['loss'])
            dynamic_losses.append(output['dynamic_loss'])
            stable_losses.append(output['stable_loss'])
            resilience_losses.append(output['resilience_loss'])
            relative_errors.append(output['relative_error'])
            dynamic_relative_errors.append(output['dynamic_relative_error'])
            stable_relative_errors.append(output['stable_relative_error'])
        mean_loss = th.stack(losses).mean()
        mean_dynamic_loss = th.stack(dynamic_losses).mean()
        mean_stable_loss = th.stack(stable_losses).mean()
        mean_resilience_loss = th.stack(resilience_losses).mean()
        mean_relative_error = th.stack(relative_errors).mean()
        mean_dynamic_relative_error = th.stack(dynamic_relative_errors).mean()
        mean_stable_relative_error = th.stack(stable_relative_errors).mean()
        
        table = PrettyTable()
        table.field_names = ['Loss', 'Dynamic-loss', 'Stable-loss', 'Resilience-loss', 'Relative-error', 'Dynamic-relative-error', 'Stable-relative-error']
        table.add_row([mean_loss.item(), mean_dynamic_loss.item(), mean_stable_loss.item(), mean_resilience_loss.item(), mean_relative_error.item(), mean_dynamic_relative_error.item(), mean_stable_relative_error.item()])
        print(table)

    def validation_step(self, data, i):
        
        self.is_test = False
        ori_epsilon = self.ode.epsilon

        self.ode.update_epsilon(0)
        if self.config.dataset.pregene:
            dense_data, node_mask = utils.to_dense_guide(data.x, data.edge_index, data.edge_attr, data.batch, data.traj_hi, data.traj_lo, data.t)
        else:
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        
        if data.y[0] == -1:
            here_type = 'semi'
            print('Validate ODE only......')
        else:
            here_type = 'full'
            print('Validate full model......')

        t_eval = th.linspace(0, self.T, self.time_ticks).to(X.device)

        t_eval = t_eval / self.T

        t_eval = t_eval[:self.ode_use_length + 1]

        if self.config.dataset.pregene:
            noisy_data = self.pre_noise(X, E, data.y, node_mask, dense_data.t, dense_data.traj_hi, dense_data.traj_lo)
        else:
            noisy_data = self.apply_noise(X, E, data.y, node_mask, t_eval)
        extra_data = self.compute_extra_data(noisy_data)
        batch_size = X.shape[0]
        loss, dynamic_loss, stable_loss, resilience_loss, relative_error, dynamic_relative_error, stable_relative_error, resi_metrics, _ = self.unwrapped_forward(noisy_data, extra_data, node_mask, t_eval, noisy_data['real_traj'], here_type)

        self.log('val_loss', loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dynamic_loss', dynamic_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_stable_loss', stable_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_resilience_loss', resilience_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_relative_error', relative_error,
                 batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dynamic_relative_error', dynamic_relative_error,
                 batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_stable_relative_error', stable_relative_error,
                 batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_resi_auc', resi_metrics['auc'], batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_resi_f1', resi_metrics['f1'], batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)

        print("Resume epsilon to:", ori_epsilon)
        self.ode.update_epsilon(ori_epsilon)

        return {'loss': loss, 'dynamic_loss': dynamic_loss, 'stable_loss': stable_loss, 'resilience_loss': resilience_loss, 'relative_error': relative_error, 'dynamic_relative_error': dynamic_relative_error, 'stable_relative_error': stable_relative_error, 'resi_metrics': resi_metrics}


    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):

        losses = []
        dynamic_losses = []
        stable_losses = []
        resilience_losses = []
        relative_errors = []
        dynamic_relative_errors = []
        stable_relative_errors = []
        resi_auc = []
        resi_f1 = []
        resi_acc = []
        resi_rate = []

        for output in outputs:

            losses.append(output['loss'])
            dynamic_losses.append(output['dynamic_loss'])
            stable_losses.append(output['stable_loss'])
            resilience_losses.append(output['resilience_loss'])
            relative_errors.append(output['relative_error'])
            dynamic_relative_errors.append(output['dynamic_relative_error'])
            stable_relative_errors.append(output['stable_relative_error'])
            resi_auc.append(output['resi_metrics']['auc'])
            resi_f1.append(output['resi_metrics']['f1'])
            resi_acc.append(output['resi_metrics']['acc'])
            resi_rate.append(output['resi_metrics']['resi_rate'])


        mean_loss = th.stack(losses).mean()
        mean_dynamic_loss = th.stack(dynamic_losses).mean()
        mean_stable_loss = th.stack(stable_losses).mean()
        mean_resilience_loss = th.stack(resilience_losses).mean()
        mean_relative_error = th.stack(relative_errors).mean()
        mean_dynamic_relative_error = th.stack(dynamic_relative_errors).mean()
        mean_stable_relative_error = th.stack(stable_relative_errors).mean()

        resi_auc = np.array(resi_auc).mean()
        resi_f1 = np.array(resi_f1).mean()
        resi_acc = np.array(resi_acc).mean()
        resi_rate = np.array(resi_rate).mean()
        
        

        table = PrettyTable()
        table.field_names = ['Loss', 'Dynamic-loss', 'Stable-loss', 'Resilience-loss', 'Relative-error', 'Dynamic-relative-error', 'Stable-relative-error', 'AUC', 'F1', 'ACC', 'resi_rate']
        
        table.add_row([mean_loss.item(), mean_dynamic_loss.item(), mean_stable_loss.item(), mean_resilience_loss.item(), mean_relative_error.item(), mean_dynamic_relative_error.item(), mean_stable_relative_error.item(), resi_auc, resi_f1, resi_acc, resi_rate])
        print(table)

    def configure_optimizers(self):
        return th.optim.AdamW(self.parameters(), lr=self.config.train.lr, amsgrad=True,
                                 weight_decay=self.config.train.weight_decay)
    
    def test_step(self, data, i):

        self.is_test = True

        # ori_epsilon = self.ode.epsilon
        self.ode.update_epsilon(0)
        if self.config.dataset.pregene:
            dense_data, node_mask = utils.to_dense_guide(data.x, data.edge_index, data.edge_attr, data.batch, data.traj_hi, data.traj_lo, data.t)
        else:
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        if data.y[0] == -1:
            here_type = 'semi'
            # print('Test ODE only......')
        else:
            here_type = 'full'
            # print('Test full model......')

        t_eval = th.linspace(0, self.T, self.time_ticks).to(X.device)

        t_eval = t_eval / self.T

        t_eval = t_eval[:self.ode_use_length + 1]
        
        if self.config.dataset.pregene:
            noisy_data = self.pre_noise(X, E, data.y, node_mask, dense_data.t, dense_data.traj_hi, dense_data.traj_lo)
        else:
            noisy_data = self.apply_noise(X, E, data.y, node_mask, t_eval)
        extra_data = self.compute_extra_data(noisy_data)
        batch_size = X.shape[0]
        loss, dynamic_loss, stable_loss, resilience_loss, relative_error, dynamic_relative_error, stable_relative_error, resi_metrics,forsave = \
            self.unwrapped_forward(noisy_data, extra_data, node_mask, t_eval, noisy_data['real_traj'], here_type)
        
        self.log('test_loss', loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_dynamic_loss', dynamic_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_stable_loss', stable_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_resilience_loss', resilience_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_relative_error', relative_error,
                 batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_dynamic_relative_error', dynamic_relative_error,
                 batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_stable_relative_error', stable_relative_error,
                 batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss, 'dynamic_loss': dynamic_loss, 'stable_loss': stable_loss, 'resilience_loss': resilience_loss, 'relative_error': relative_error, 'dynamic_relative_error': dynamic_relative_error, 'stable_relative_error': stable_relative_error, 'resi_metrics': resi_metrics, 'forsave':forsave}

    def test_epoch_end(self, outputs: EPOCH_OUTPUT):
        
        losses = []
        dynamic_losses = []
        stable_losses = []
        resilience_losses = []
        relative_errors = []
        dynamic_relative_errors = []
        stable_relative_errors = []
        resi_auc = []
        resi_f1 = []
        resi_acc = []

        resi_rate = []
        

        forsaves = []

        for output in outputs:

            losses.append(output['loss'])
            dynamic_losses.append(output['dynamic_loss'])
            stable_losses.append(output['stable_loss'])
            resilience_losses.append(output['resilience_loss'])
            relative_errors.append(output['relative_error'])
            dynamic_relative_errors.append(output['dynamic_relative_error'])
            stable_relative_errors.append(output['stable_relative_error'])
            resi_auc.append(output['resi_metrics']['auc'])
            resi_f1.append(output['resi_metrics']['f1'])
            resi_acc.append(output['resi_metrics']['acc'])
            resi_rate.append(output['resi_metrics']['resi_rate'])
            forsaves.append(output['forsave'])


        mean_loss = th.stack(losses).mean()
        mean_dynamic_loss = th.stack(dynamic_losses).mean()
        mean_stable_loss = th.stack(stable_losses).mean()
        mean_resilience_loss = th.stack(resilience_losses).mean()
        mean_relative_error = th.stack(relative_errors).mean()
        mean_dynamic_relative_error = th.stack(dynamic_relative_errors).mean()
        mean_stable_relative_error = th.stack(stable_relative_errors).mean()
        resi_auc = np.array(resi_auc).mean()
        resi_f1 = np.array(resi_f1).mean()
        resi_acc = np.array(resi_acc).mean()
        resi_rate = np.array(resi_rate).mean()
    
        # resi_auc = th.stack(th.from_numpy(np.array(resi_auc))).mean()
        # resi_f1 = th.stack(th.from_numpy(np.array(resi_f1))).mean()

        table = PrettyTable()
        table.field_names = ['Loss', 'Dynamic-loss', 'Stable-loss', 'Resilience-loss', 'Relative-error', 'Dynamic-relative-error', 'Stable-relative-error', 'AUC', 'F1', 'ACC', 'Resi_Rate']
        # table.add_row([mean_loss.item(), mean_dynamic_loss.item(), mean_stable_loss.item(), mean_resilience_loss.item(), mean_relative_error.item(), mean_dynamic_relative_error.item(), mean_stable_relative_error.item(), outputs['resi_metrics']['auc'], outputs['resi_metrics']['f1']])
        table.add_row([mean_loss.item(), mean_dynamic_loss.item(), mean_stable_loss.item(), mean_resilience_loss.item(), mean_relative_error.item(), mean_dynamic_relative_error.item(), mean_stable_relative_error.item(), resi_auc, resi_f1, resi_acc, resi_rate])
        print(table)

        th.save(forsaves, 'test_saved.pt')



    

    def apply_noise(self, X, E, y, node_mask, t_eval):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = th.randint(lowest_t, self.diffusion_steps + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.diffusion_steps
        s_float = s_int / self.diffusion_steps

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).float()
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).float()
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        # x_0 = th.zeros((X_t.shape[1], X_t.shape[0], 1)).to(X_t.device)

        B, N = E_t.shape[0], E_t.shape[1]
        target_shape = (B, N, N, 2)
        check_mat = th.tensor([0, 1]).repeat(*target_shape[:3], 1).to(E_t.device)

        A = th.mul(E_t, check_mat).sum(-1)
        A = A.cpu()
        t_eval = t_eval.cpu()
        # x_reals = th.empty((N, t_eval.shape[0])).to(X_t.device).unsqueeze(0)
        print('Simulating ground truth trajectories...')
        _evolve_lo = partial(evolve, mu=self.config.general['mu'], delta=self.config.general['delta'], t_eval=t_eval, method='dopri5', x_0=th.zeros((N, 1)))
        pool = multiprocessing.Pool(processes=B)
        A_list = th.unbind(A, dim=0)
        x_reals_lo = pool.map(_evolve_lo, A_list)
        x_reals_lo = th.stack(x_reals_lo, dim=0).to(X_t.device)
        # x_reals_lo_final = x_reals_lo[..., -1]

        _evolve_hi = partial(evolve, mu=self.config.general['mu'], delta=self.config.general['delta'], t_eval=t_eval, method='dopri5', x_0=th.ones((N, 1))*5)
        pool = multiprocessing.Pool(processes=B)
        A_list = th.unbind(A, dim=0)
        x_reals_hi = pool.map(_evolve_hi, A_list)
        x_reals_hi = th.stack(x_reals_hi, dim=0).to(X_t.device)
        # x_reals_hi_final = x_reals_hi[..., -1]

        # masked_lo_final = x_reals_lo_final.masked_fill(~node_mask, 0)
        # masked_hi_final = x_reals_hi_final.masked_fill(~node_mask, 0)
        # sum_values_hi = masked_hi_final.sum(dim=1, keepdim=True)
        # sum_values_lo = masked_lo_final.sum(dim=1, keepdim=True)
        # count_values = node_mask.sum(dim=1, keepdim=True)
        # mean_values_hi = th.div(sum_values_hi, count_values)
        # mean_values_lo = th.div(sum_values_lo, count_values)

        # y = th.zeros((B, 1)).to(X_t.device)
        # y = th.where(abs(mean_values_hi - mean_values_lo) < 3 and mean_values_lo > 3.5, th.ones_like(y), y)


        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        pool.close()
        print('Finish!')

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask, 'real_traj': {'hi': x_reals_hi, 'lo': x_reals_lo}}
        
        return noisy_data
    


