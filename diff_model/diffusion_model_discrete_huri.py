import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import numpy as np
# import wandb
import os
import matplotlib.pyplot as plt
from models.transformer_model_ori import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from diff_model import utils
import inspect
# from gpu_mem_track import MemTracker
# from gpu_memory_log import gpu_memory_log
from tqdm import tqdm

class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps
        
        self.conditional = cfg.general.conditional

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

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)
        
        if cfg.model.transition == 'uniform':
            node_types = self.dataset_info.node_types.float()
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
            x_paddings = torch.tensor([1.], device=node_types.device)
            e_paddings = torch.tensor([1.0, 0.0], device=node_types.device)
            y_paddings = torch.tensor([], device=node_types.device)
            self.padding_dist = utils.PlaceHolder(X=x_paddings, E=e_paddings,y=y_paddings)

        elif cfg.model.transition == 'marginal':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)
            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)
            # print('Y::')
            # print(torch.ones(self.ydim_output) / self.ydim_output)
            # assert False
            x_paddings = torch.tensor([1.], device=x_marginals.device)
            e_paddings = torch.tensor([1.0, 0.0], device=e_marginals.device)
            y_paddings = torch.tensor([], device=x_marginals.device)
            self.padding_dist = utils.PlaceHolder(X=x_paddings, E=e_paddings,y=y_paddings)
        # self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.best_resi_rate = None
        self.no_improvement_counter = 0
        self.patience = cfg.general.patience

    def training_step(self, data, i):
        
        if data.edge_index.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return
    
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)

        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.train.batch_size)
        
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)
    
    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        print("Size of the input features", self.Xdim, self.Edim, self.ydim)
    
    def on_train_epoch_start(self) -> None:
        print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        epoch_loss = self.train_loss.log_epoch_metrics(self.current_epoch, self.start_epoch_time)
        self.train_metrics.log_epoch_metrics(self.current_epoch)
        self.log('train_loss', epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.train.batch_size)
    
    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i, dataloader_idx=0):

        if self.conditional:
            if dataloader_idx == 0:
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                dense_data = dense_data.mask(node_mask)
                noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
                extra_data = self.compute_extra_data(noisy_data)
                pred = self.forward(noisy_data, extra_data, node_mask)
                nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)

                # return {'loss': nll, 'gt_data': dense_data}
                return {'loss':nll}
            else:
                dense_data, node_mask = utils.to_dense_condition(data[0].x, data[0].edge_index, data[0].edge_attr, data[0].batch, data[0].cond_mask)
                dense_data = dense_data.mask(node_mask)
                noisy_data = self.apply_noise(dense_data.X, dense_data.E, data[0].y, node_mask)
                extra_data = self.compute_extra_data(noisy_data)
                pred = self.forward(noisy_data, extra_data, node_mask)
                nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data[0].y,  node_mask, test=False)

                ori_data, ori_data_node_mask = utils.to_dense_condition(data[1].x, data[1].edge_index, data[1].edge_attr, data[1].batch, data[1].cond_mask)
                ori_data = ori_data.mask(ori_data_node_mask)

                return {'loss': nll, 'gt_data': dense_data, 'ori_data':ori_data}
        else:
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)

            return {'loss': nll}

    def validation_epoch_end(self, outs) -> None:
        # gt_data = outs['gt_data']
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        
        # wandb.log({"val/epoch_NLL": metrics[0],
        #            "val/X_kl": metrics[1],
        #            "val/E_kl": metrics[2],
        #            "val/X_logp": metrics[3],
        #            "val/E_logp": metrics[4]}, commit=False)

        # self.log("performance",{"val/epoch_NLL": metrics[0], "val/X_kl": metrics[1], "val/E_kl": metrics[2], "val/X_logp": metrics[3], "val/E_logp": metrics[4]})
        
        print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
              f"Val Edge type KL: {metrics[2] :.2f}")
        
        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, on_epoch=True, on_step=False)
        
        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.train.batch_size * 2
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            # if (self.current_epoch - 14) % 100 == 0 or self.current_epoch == 1000:

            #     samples_left_to_generate = 100

            print(f"Generating {samples_left_to_generate} samples")
            # if self.conditional:
            #     samples_left_to_generate = samples_left_to_generate // 2
            #     samples_left_to_save = samples_left_to_save // 2
            #     chains_left_to_save = chains_left_to_save // 2

            samples = []
            ori_list = []
        
            ident = 0
            count = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                if self.conditional:
                    bs = bs // 2
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                if self.conditional:
                    gt_data = outs[1][count]['gt_data']
                    

                    ori_data = outs[1][count]['ori_data']

                    _, counts_ori = torch.unique(ori_data.batch, return_counts=True, sorted=True)
                    b_size = len(counts_ori)
                    here_ori_list = []
                    for i in range(b_size):
                        n = counts_ori[i]
                        node_types = ori_data.X[i, :n].cpu()
                        edge_types = ori_data.E[i, :n, :n].cpu()
                        cond_masks = ori_data.cond_mask[i, :n].cpu()
                        here_ori_list.append((node_types, edge_types, cond_masks))
                    ori_list.extend(here_ori_list)





                    unique, counts = torch.unique(gt_data.batch, return_counts=True, sorted=True)
                    to_generate = len(counts)
                    counts = counts.reshape(to_generate,).contiguous()
                    if self.cfg.general.type == 'fixed':
                        samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=self.cfg.general.fix_num,
                                                        save_final=to_save,
                                                        keep_chain=chains_save,
                                                        number_chain_steps=self.number_chain_steps, cond_data=gt_data, state='val', cond_counts=None))
                    else:
                        samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=counts,
                                                        save_final=to_save,
                                                        keep_chain=chains_save,
                                                        number_chain_steps=self.number_chain_steps, cond_data=gt_data, state='val', cond_counts=counts))
                else:
                    samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                    save_final=to_save,
                                                    keep_chain=chains_save,
                                                    number_chain_steps=self.number_chain_steps, cond_counts=None))
                ident += to_generate
                count = (count + 1) % len(outs[1])
                
                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            print("Computing sampling metrics...")
            to_log = self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False)
            print(to_log)
            self.log_dict(to_log, on_epoch=True, on_step=False)
            print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            self.sampling_metrics.reset()
            torch.save(samples, 'graph_adjs/samples_epoch{}_val{}.pt'.format(self.current_epoch, self.val_counter))

            torch.save(ori_list, 'graph_adjs/ori_data_epoch{}_val{}.pt'.format(self.current_epoch, self.val_counter))

            if self.best_resi_rate is None or to_log['resi_rate'] > self.best_resi_rate:
                self.best_resi_rate = to_log['resi_rate']
                print('After %d not improving calculation:' % self.no_improvement_counter)
                print('New best resi rate: %.4f' % self.best_resi_rate)
                self.no_improvement_counter = 0
                self.save_best_model()
            else:
                self.no_improvement_counter += 1
            
            if (self.current_epoch - 14) % 100 == 0 or self.current_epoch == 1000:
                self.save_val_ckpt(self.current_epoch, to_log)
            # if self.no_improvement_counter >= self.patience:
            #     print('Early stopping')
            #     self.trainer.should_stop = True
    

    def save_best_model(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'state_dict': self.state_dict(),
            'best_resi_rate': self.best_resi_rate
        }
        torch.save(checkpoint, f'best_model.pth')
    
    def save_val_ckpt(self, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, f'val_ckpt_{epoch}.pth')

    def save_test_ckpt(self, metrics):
        checkpoint = {
            'state_dict': self.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, f'test_ckpt.pth')

    def on_test_epoch_start(self) -> None:
        print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()

    def test_step(self, data, i, dataloader_idx=0):

        if self.conditional:
            if dataloader_idx == 0:
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                dense_data = dense_data.mask(node_mask)
                noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
                extra_data = self.compute_extra_data(noisy_data)
                pred = self.forward(noisy_data, extra_data, node_mask)
                nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
                return {'loss': nll}

            else:
                dense_data, node_mask = utils.to_dense_condition(data[0].x, data[0].edge_index, data[0].edge_attr, data[0].batch, data[0].cond_mask)
                dense_data = dense_data.mask(node_mask)
                noisy_data = self.apply_noise(dense_data.X, dense_data.E, data[0].y, node_mask)
                extra_data = self.compute_extra_data(noisy_data)
                pred = self.forward(noisy_data, extra_data, node_mask)
                nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data[0].y, node_mask, test=True)

                ori_data, ori_data_node_mask = utils.to_dense_condition(data[1].x, data[1].edge_index, data[1].edge_attr, data[1].batch, data[1].cond_mask)
                ori_data = ori_data.mask(ori_data_node_mask)

                return {'loss': nll, 'gt_data': dense_data, 'ori_data':ori_data}

        else:
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask)
            noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
            return {'loss': nll}

    def test_epoch_end(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        
        # wandb.log({"test/epoch_NLL": metrics[0],
        #            "test/X_kl": metrics[1],
        #            "test/E_kl": metrics[2],
        #            "test/X_logp": metrics[3],
        #            "test/E_logp": metrics[4]}, commit=False)

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
              f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
        # wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        print(f'Test loss: {test_nll :.4f}')

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        
        samples = []
        ori_list = []
        forpaint_list = []
        id = 0
        count = 0
        while samples_left_to_generate > 0:
            print(f'Samples left to generate: {samples_left_to_generate}/'
                  f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            if self.conditional:
                bs = bs // 2
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            if self.conditional:
                gt_data = outs[1][count]['gt_data']
                ori_data = outs[1][count]['ori_data']
                _, counts_ori = torch.unique(ori_data.batch, return_counts=True, sorted=True)
                b_size = len(counts_ori)
                here_ori_list = []
                for i in range(b_size):
                    n = counts_ori[i]
                    node_types = ori_data.X[i, :n].cpu()
                    edge_types = ori_data.E[i, :n, :n].cpu()
                    cond_masks = ori_data.cond_mask[i, :n].cpu()
                    edge_types = torch.argmax(edge_types, dim=-1)
                    here_ori_list.append((node_types, edge_types, cond_masks))
                ori_list.extend(here_ori_list)

                _, counts_forpaint = torch.unique(gt_data.batch, return_counts=True, sorted=True)
                forpaint_size = len(counts_forpaint)
                here_forpaint_list = []
                for i in range(forpaint_size):
                    n = counts_forpaint[i]
                    node_types = gt_data.X[i, :n].cpu()
                    edge_types = gt_data.E[i, :n, :n].cpu()
                    # cond_masks = gt_data.cond_mask[i, :n].cpu()
                    edge_types = torch.argmax(edge_types, dim=-1)
                    here_forpaint_list.append((node_types, edge_types))
                forpaint_list.extend(here_forpaint_list)

                unique, counts = torch.unique(gt_data.batch, return_counts=True, sorted=True)
                to_generate = len(counts)
                counts = counts.reshape(to_generate,).contiguous()
                if self.cfg.general.type == 'fixed':
                    samples.extend(self.sample_batch(id, to_generate, num_nodes=self.cfg.general.fix_num, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps, cond_data=gt_data, state='test'))
                else:
                    samples.extend(self.sample_batch(id, to_generate, num_nodes=counts, save_final=to_save,
                                                 keep_chain=chains_save, number_chain_steps=self.number_chain_steps, cond_data=gt_data, state='test', cond_counts=counts))

            else:
                samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            count = (count + 1) % len(outs[1])
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        print("Saving the generated graphs")
        filename = f'generated_samples1.txt'
        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f'generated_samples{i}.txt'
            else:
                break
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        torch.save(samples, 'graph_adjs/samples_test.pt')

        torch.save(ori_list, 'graph_adjs/ori_data_test.pt')

        torch.save(forpaint_list, 'graph_adjs/forpaint_data_test.pt')

        print("Saved.")
        print("Computing sampling metrics...")
        self.sampling_metrics.reset()
        to_log = self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True,comm=self.cfg.general.control_strategy=='community')

        file_path = 'test_results.txt'
        file = open(file_path, "w")
        dict_str = str(to_log)
        file.write(dict_str)
        file.close()
        self.save_test_ckpt(to_log)
        self.sampling_metrics.reset()
        print("Done.")

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)
        
        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

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

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def apply_noise_for_known(self, X, E, y, node_mask, t):

        ones = torch.ones((X.size(0), 1), device=X.device)
        t_int = t * ones
        t_float = t_int / self.T

        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        # print(probX[0])
        # assert False

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).float()
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).float()
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)
        
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        return z_t
    
    def apply_noise_single_step(self, X, E, y, node_mask, t):
        ones = torch.ones((X.size(0), 1), device=X.device)
        t_int = t * ones
        t_float = t_int / self.T

        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()
        try:
            probX = X @ Qtb.X  # (bs, n, dx_out)
        except:
            print(X.shape, Qtb.X.shape)
            print(X.type(), Qtb.X.type())
            assert False
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        # print(probX[0])
        # assert False

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).float()
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).float()
        assert (E_t == torch.transpose(E_t, 1, 2)).all()
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)
        
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        return z_t

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

       
        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)


        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)
        # print(loss_all_t)
        # assert False

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        # wandb.log({"kl prior": kl_prior.mean(),
        #            "Estimator loss terms": loss_all_t.mean(),
        #            "log_pn": log_pN.mean(),
        #            "loss_term_0": loss_term_0,
        #            'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)


    @torch.no_grad()
    def _check_times(self,times, t_0, t_T):
        # Check end
        assert times[0] > times[1], (times[0], times[1])

        # Check beginning
        assert times[-1] == -1, times[-1]

        # Steplength = 1
        for t_last, t_cur in zip(times[:-1], times[1:]):
            assert abs(t_last - t_cur) == 1, (t_last, t_cur)

        # Value range
        for t in times:
            assert t >= t_0, (t, t_0)
            assert t <= t_T, (t, t_T)
    
    @torch.no_grad()
    def get_schedule_jump(self, t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):

        jumps = {}
        for j in range(0, t_T - jump_length, jump_length):
            jumps[j] = jump_n_sample - 1

        jumps2 = {}
        for j in range(0, t_T - jump2_length, jump2_length):
            jumps2[j] = jump2_n_sample - 1

        jumps3 = {}
        for j in range(0, t_T - jump3_length, jump3_length):
            jumps3[j] = jump3_n_sample - 1

        t = t_T
        ts = []

        while t >= 1:
            t = t-1
            ts.append(t)

            if (
                t + 1 < t_T - 1 and
                t <= start_resampling
            ):
                for _ in range(n_sample - 1):
                    t = t + 1
                    ts.append(t)

                    if t >= 0:
                        t = t - 1
                        ts.append(t)

            if (
                jumps3.get(t, 0) > 0 and
                t <= start_resampling - jump3_length
            ):
                jumps3[t] = jumps3[t] - 1
                for _ in range(jump3_length):
                    t = t + 1
                    ts.append(t)

            if (
                jumps2.get(t, 0) > 0 and
                t <= start_resampling - jump2_length
            ):
                jumps2[t] = jumps2[t] - 1
                for _ in range(jump2_length):
                    t = t + 1
                    ts.append(t)
                jumps3 = {}
                for j in range(0, t_T - jump3_length, jump3_length):
                    jumps3[j] = jump3_n_sample - 1

            if (
                jumps.get(t, 0) > 0 and
                t <= start_resampling - jump_length
            ):
                jumps[t] = jumps[t] - 1
                for _ in range(jump_length):
                    t = t + 1
                    ts.append(t)
                jumps2 = {}
                for j in range(0, t_T - jump2_length, jump2_length):
                    jumps2[j] = jump2_n_sample - 1

                jumps3 = {}
                for j in range(0, t_T - jump3_length, jump3_length):
                    jumps3[j] = jump3_n_sample - 1

        ts.append(-1)

        self._check_times(ts, -1, t_T)

        return ts

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, cond_data=None, state=None, cond_counts=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        # if self.conditional:
        #     if num_nodes is None:
        #         if state == 'val':
        #             n_nodes = self.node_dist_conditional_val.sample_n(batch_size, self.device)
        #         elif state == 'test':
        #             n_nodes = self.node_dist_conditional_test.sample_n(batch_size, self.device)
        #         else:
        #             raise NotImplementedError
        #     elif type(num_nodes) == int:
        #         raise NotImplementedError
        #     else:
        #         raise NotImplementedError
        
        # else:
        if num_nodes is None:
            ### sample n from the training data distribution
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        if cond_counts is not None:
            n_nodes = torch.where(n_nodes > cond_counts, n_nodes, cond_counts)
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)

        # if self.conditional:
        #     X_origin, E_origin, y_origin, gt_mask = cond_data.X, cond_data.E, cond_data.y, cond_data.cond_mask
        #     # print(gt_mask.shape)
        #     # cond_mask_E = E_origin.nonzero().....
        #     # cond_mask_E = (E_origin == 0) + 0
        #     # cond_mask_E = torch.ones_like(E_origin)
        #     cond_mask_E = (E_origin == 1) + 0
        #     X, E, y = X_origin.clone(), E_origin.clone(), torch.empty((gt_mask.shape[0], 0)).type_as(X_origin)

        #     X = F.pad(X, (0,0,0,n_max - X.shape[1]), 'constant', 0)
        #     gt_mask = F.pad(gt_mask, (0,0,0,n_max - gt_mask.shape[1]), 'constant', 0)
        #     y = torch.zeros([gt_mask.shape[0], 0]).type_as(X_origin)
        #     E = F.pad(E, (0,0,0,n_max - E.shape[1], 0, n_max - E.shape[2]), 'constant', 0)
        #     cond_mask_E = F.pad(cond_mask_E, (0,0,0,n_max - cond_mask_E.shape[1], 0, n_max - cond_mask_E.shape[2]), 'constant', 0)
        #     X_origin = F.pad(X_origin, (0,0,0,n_max - X_origin.shape[1]), 'constant', 0)
        #     E_origin = F.pad(E_origin, (0,0,0,n_max- E_origin.shape[1], 0, n_max - E_origin.shape[2]), 'constant', 0)
        #     ori_mask = ((X_origin == 1) + 0).squeeze(-1)
        #     e_mask1 = ori_mask.unsqueeze(-1).unsqueeze(2)
        #     e_mask2 = ori_mask.unsqueeze(-1).unsqueeze(1)

        #     # assert False
        #     # print(gt_mask.shape)
        #     # assert False
        #     # z_T = self.apply_noise_for_known(X, E, y, node_mask, self.T)
        #     # X, E, y = z_T.X, z_T.E, z_T.y

        #     # X_T = X_origin * ori_mask.unsqueeze(-1) + X_T * (1 - ori_mask.unsqueeze(-1))
        #     # E_T = E_origin * e_mask1 * e_mask2 + E_T * (1 - e_mask1 * e_mask2)
            
        #     known_T = self.apply_noise_for_known(X_origin, E_origin, y, ori_mask.type_as(node_mask), self.T)
        #     X_known, E_known, y_known = known_T.X, known_T.E, known_T.y

        #     z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)


        #     X_unknown, E_unknown, y = z_T.X, z_T.E, z_T.y
        #     X = X_known * ori_mask.unsqueeze(-1) + X_unknown * (1 - ori_mask.unsqueeze(-1))

        #     if self.cfg.general.ne:
        #         E = E_known * cond_mask_E + E_unknown * (1 - cond_mask_E)
        #     else:
        #         E = E_known * e_mask1 * e_mask2 + E_unknown * (1 - e_mask1 * e_mask2)
                
        # else:
        #     z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        #     X, E, y = z_T.X, z_T.E, z_T.y

        # assert number_chain_steps < self.T
        # chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        # chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        # chain_X = torch.zeros(chain_X_size)
        # chain_E = torch.zeros(chain_E_size)

        if self.conditional:
            X_origin, E_origin, y_origin, inpaint_mask = cond_data.X, cond_data.E, cond_data.y, cond_data.cond_mask

            if self.cfg.general.control_strategy == 'community':
                control_num = inpaint_mask.clone().squeeze(-1).sum(-1)
                total_num = X_origin.clone().squeeze(-1).sum(-1)
                control_rate = control_num/total_num



            # n = X_origin.shape[0]
            # gt_mask = torch.ones(n, 1, dtype=torch.long).to(X_origin.device)
            gt_mask = X_origin.clone().long()
            gt_mask_padded = F.pad(gt_mask, (0,0,0,n_max - gt_mask.shape[1]), 'constant', 0).to(torch.bool)
            gt_mask = gt_mask.to(torch.bool)
            

            inpaint_mask_padded = F.pad(inpaint_mask, (0,0,0,n_max - inpaint_mask.shape[1]), 'constant', 0).to(torch.bool)

            # e_gt_mask1 = inpaint_mask_padded.squeeze(-1).unsqueeze(-1).unsqueeze(2)
            # e_gt_mask2 = inpaint_mask_padded.squeeze(-1).unsqueeze(-1).unsqueeze(1)

            
            e_gt_mask1 = gt_mask_padded.squeeze(-1).unsqueeze(-1).unsqueeze(2)
            e_gt_mask2 = gt_mask_padded.squeeze(-1).unsqueeze(-1).unsqueeze(1)
            e_gt_mask = e_gt_mask1 * e_gt_mask2

            e_gt_mask = e_gt_mask.expand(e_gt_mask.shape[0], e_gt_mask.shape[1], e_gt_mask.shape[2], 2)
            padding_feature = diffusion_utils.sample_padding_feature(padding_dist=self.padding_dist, node_mask=node_mask)
            X_padding, E_padding, y_padding = padding_feature.X, padding_feature.E, padding_feature.y
            # assert False
            X_padding[gt_mask_padded] = X_origin[gt_mask]
            # assert False
            e_gt_mask_origin = (gt_mask.unsqueeze(2) * gt_mask.unsqueeze(1))
            e_gt_mask_origin = e_gt_mask_origin.expand(e_gt_mask_origin.shape[0], e_gt_mask_origin.shape[1], e_gt_mask_origin.shape[2], 2)
            E_padding[e_gt_mask] = E_origin[e_gt_mask_origin]
            X_origin = X_padding.clone()
            E_origin = E_padding.clone()
            
            X = X_origin.clone()
            E = E_origin.clone()
            y = torch.zeros([gt_mask_padded.shape[0], 0]).type_as(X_origin)


            # if self.cfg.general.to_all:
                # cond_mask_E = 
            e_inpaint_mask1 = inpaint_mask_padded.squeeze(-1).unsqueeze(-1).unsqueeze(2)
            e_inpaint_mask2 = inpaint_mask_padded.squeeze(-1).unsqueeze(-1).unsqueeze(1)


            
            cond_mask_E = e_inpaint_mask1 + e_inpaint_mask2

            cond_mask_E[cond_mask_E > 0] = 1

            # else:
            #     cond_mask_E = inpaint_mask_padded.unsqueeze(2) & inpaint_mask_padded.unsqueeze(1)

            

            cond_mask_E = cond_mask_E.expand(cond_mask_E.shape[0], cond_mask_E.shape[1], cond_mask_E.shape[2], \
                                            2)
            



            # cond_mask_E = torch.full((E_origin.size(0), E_origin.size(1), E_origin.size(2), 1), False, device=E_origin.device)
            # is_last_vector_01 = torch.all(torch.eq(E_origin, torch.tensor([0, 1], device=E_origin.device)), dim=-1).unsqueeze(-1)
            # is_last_vector_00 = torch.all(torch.eq(E_origin, torch.tensor([0, 0], device=E_origin.device)), dim=-1).unsqueeze(-1)
            # cond_mask_E[is_last_vector_01] = True
            # cond_mask_E[is_last_vector_00] = True




            cond_mask_E = cond_mask_E + 0


            e_gt_mask = e_gt_mask + 0
            # E_view = torch.argmax(E_origin,dim=-1)
            # plt.matshow(E_view[0, :, :].cpu().numpy())
            # plt.savefig('./E_view.png')
            # plt.matshow(cond_mask_E[0, :, :, 0].cpu().numpy())
            # plt.savefig('./cond_mask_E.png')
            # assert False

            # cond_mask_E = F.pad(cond_mask_E, (0,0,0,n_max - cond_mask_E.shape[1], 0, n_max - cond_mask_E.shape[2]), 'constant', 0)
            # X_origin = F.pad(X_origin, (0,0,0,n_max - X_origin.shape[1]), 'constant', 0)
            # E_origin = F.pad(E_origin, (0,0,0,n_max- E_origin.shape[1], 0, n_max - E_origin.shape[2]), 'constant', 0)
            # ori_mask = ((X_origin == 1) + 0).squeeze(-1)
            # e_mask1 = ori_mask.unsqueeze(-1).unsqueeze(2)
            # e_mask2 = ori_mask.unsqueeze(-1).unsqueeze(1)


            # z_T = self.apply_noise_for_known(X, E, y, node_mask, self.T)
            # X, E, y = z_T.X, z_T.E, z_T.y

            # X_T = X_origin * ori_mask.unsqueeze(-1) + X_T * (1 - ori_mask.unsqueeze(-1))
            # E_T = E_origin * e_mask1 * e_mask2 + E_T * (1 - e_mask1 * e_mask2)
            
            # known_T = self.apply_noise_for_known(X_origin, E_origin, y, gt_mask_padded.type_as(node_mask), self.T)
            # X_known, E_known, y_known = known_T.X, known_T.E, known_T.y
            if self.cfg.model.resample:

                image_after_step = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)

                times = self.get_schedule_jump(**self.cfg.model.resample_params)

                time_pairs = list(zip(times[:-1], times[1:]))
                
                for t_last, t_cur in tqdm(time_pairs):
                    t_last_t = t_last * torch.ones((batch_size, 1)).type_as(y)
                    t_cur_t = t_cur * torch.ones((batch_size, 1)).type_as(y)
                    t_last_norm = t_last_t / self.T
                    t_cur_norm = t_cur_t / self.T
                    
                    if t_cur < t_last:
                        with torch.no_grad():
                            # known_part = self.apply_noise_for_known(X_origin, E_origin, y, gt_mask_padded.type_as(node_mask), t_last)
                            # X_known, E_known, y_known = known_part.X, known_part.E, known_part.y
                            X_known, E_known, y_known = X_origin, E_origin, torch.zeros([gt_mask_padded.shape[0], 0]).type_as(X_origin)
                            X_unknown, E_unknown, y = image_after_step.X, image_after_step.E, image_after_step.y
                            # X = X_known * gt_mask_padded.unsqueeze(-1) + X_unknown * (1 - gt_mask_padded.unsqueeze(-1))
                            X = X_unknown

                            if self.cfg.general.ne:
                                E = E_known * cond_mask_E + E_unknown * (1 - cond_mask_E)
                            else:
                                E = E_known * e_gt_mask + E_unknown * (1 - e_gt_mask)
                            
                            image_after_step, discrete_image_after_step = self.sample_p_zs_given_zt(t_cur_norm, t_last_norm, X, E, y, node_mask)
                    else:
                        t_shift = 1
                        image_after_step = self.apply_noise_single_step(image_after_step.X, image_after_step.E, image_after_step.y, node_mask, t_last + t_shift)
                
                image_after_step = image_after_step.mask(node_mask, collapse=True)
                X, E, y = image_after_step.X, image_after_step.E, image_after_step.y
                
                
            else:
                X_known, E_known, y_known = X_origin.clone(), E_origin.clone(), torch.zeros([gt_mask_padded.shape[0], 0]).type_as(X_origin)
                z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)

                X_unknown, E_unknown, y = z_T.X, z_T.E, z_T.y
                X = X_unknown

                if self.cfg.general.ne:
                    # E = E_known * cond_mask_E + E_unknown * (1 - cond_mask_E)
                    E = E_known * (1 - cond_mask_E) + E_unknown * (cond_mask_E)
                else:
                    # E = E_known * e_mask1 * e_mask2 + E_unknown * (1 - e_mask1 * e_mask2)
                    E = E_known * e_gt_mask + E_unknown * (1 - e_gt_mask)


                for s_int in tqdm(reversed(range(0, self.T)), total=len(range(0, self.T))):
                    s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
                    t_array = s_array + 1
                    s_norm = s_array / self.T
                    t_norm = t_array / self.T

                    # Sample z_s
                    sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)

                    

                    ### original code
                    # X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
                    

                        # z_t = self.apply_noise_for_known(X_origin, E_origin, y, ori_mask.type_as(node_mask), s_int)
                        # X_known, E_known, y = z_t.X, z_t.E, z_t.y
                        # X = gt_mask * X_known + (1 - gt_mask) * sampled_s.X
                        # E = cond_mask_E * E_known + (1 - cond_mask_E) * sampled_s.E
                        # X = X_known * ori_mask.unsqueeze(-1) + sampled_s.X * (1 - ori_mask.unsqueeze(-1))
                    X = X_unknown
                    if self.cfg.general.ne:
                        # E = E_known * cond_mask_E + sampled_s.E * (1 - cond_mask_E)
                        E = E_known * (1 - cond_mask_E) + sampled_s.E * (cond_mask_E)
                    else:
                        E = E_known * e_gt_mask + sampled_s.E * (1 - e_gt_mask)
                    y = sampled_s.y

                    # Save the first keep_chain graphs
                    # write_index = (s_int * number_chain_steps) // self.T
                    # chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                    # chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        
                    # if s_int % 25 == 0:
                    #     sampled_s_Save = sampled_s.mask(node_mask, collapse=True)
                    #     X_Save, E_Save, y_Save = sampled_s_Save.X, sampled_s_Save.E, sampled_s_Save.y
                    #     molecule_list = []
                    #     for i in range(batch_size):
                    #         n = n_nodes[i]
                    #         atom_types = X_Save[i, :n].cpu()
                    #         edge_types = E_Save[i, :n, :n].cpu()
                    #         if self.cfg.general.control_strategy == 'community':
                    #             rt = control_rate[i].cpu().item()
                    #             molecule_list.append([atom_types, edge_types, rt])
                    #         else:
                    #             molecule_list.append([atom_types, edge_types])
                    #     torch.save(molecule_list, f'./sampled_network_{batch_id}_{s_int}.pt')
                    if s_int == 0:
                        sampled_s.E = E
                # Sample
                sampled_s = sampled_s.mask(node_mask, collapse=True)
                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y




        else:
            
            z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
            X, E, y = z_T.X, z_T.E, z_T.y
            # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
            for s_int in reversed(range(0, self.T)):
                s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
                t_array = s_array + 1
                s_norm = s_array / self.T
                t_norm = t_array / self.T

                # Sample z_s
                sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)

                

                ### original code
                # X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
                X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

                # Save the first keep_chain graphs
                # write_index = (s_int * number_chain_steps) // self.T
                # chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                # chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

            # Sample
            sampled_s = sampled_s.mask(node_mask, collapse=True)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y



        # Prepare the chain for saving
        # if keep_chain > 0:
        #     final_X_chain = X[:keep_chain]
        #     final_E_chain = E[:keep_chain]

        #     chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
        #     chain_E[0] = final_E_chain

        #     chain_X = diffusion_utils.reverse_tensor(chain_X)
        #     chain_E = diffusion_utils.reverse_tensor(chain_E)

        #     # Repeat last frame to see final sample better
        #     chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
        #     chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
        #     assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            if self.cfg.general.control_strategy == 'community':
                rt = control_rate[i].cpu().item()
                molecule_list.append([atom_types, edge_types, rt])
            else:
                molecule_list.append([atom_types, edge_types])

        predicted_graph_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            if self.cfg.general.control_strategy == 'community':
                rt = control_rate[i].cpu().item()
                predicted_graph_list.append([atom_types, edge_types,rt])
            else:
                predicted_graph_list.append([atom_types, edge_types])

        # Visualize chains
        # if self.visualization_tools is not None:
        #     print('Visualizing chains...')
        #     current_path = os.getcwd()
        #     num_molecules = chain_X.size(1)       # number of molecules
        #     for i in range(num_molecules):
        #         result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
        #                                                  f'epoch{self.current_epoch}/'
        #                                                  f'chains/molecule_{batch_id + i}')
        #         if not os.path.exists(result_path):
        #             os.makedirs(result_path)
        #             _ = self.visualization_tools.visualize_chain(result_path,
        #                                                          chain_X[:, i, :].numpy(),
        #                                                          chain_E[:, i, :].numpy())
        #         print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
        #     print('\nVisualizing molecules...')

        #     # Visualize the final molecules
        #     current_path = os.getcwd()
        #     result_path = os.path.join(current_path,
        #                                f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
        #     self.visualization_tools.visualize(result_path, molecule_list, save_final)
        #     self.visualization_tools.visualize(result_path, predicted_graph_list, save_final, log='predicted')
        #     print("Done.")

        return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(X_t), out_discrete.mask(node_mask, collapse=True).type_as(X_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
