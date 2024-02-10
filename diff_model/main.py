import os
import sys

sys.path.append('..')
# from rdkit import Chem
try:
    # import graph_tool
    pass
except ModuleNotFoundError:
    pass

import os
import pathlib
import warnings
import torch
# import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import pytorch_lightning as pl
import utils
import inspect

from datasets.spectre_dataset import SBMDataModule, Comm20DataModule, PlanarDataModule, ResiDataModule, SpectreDatasetInfos, ResiPairDatasetInfos
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics, ResiSamplingMetrics, ResiBipartiteSamplingMetrics


from analysis.visualization import MolecularVisualization, NonMolecularVisualization
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures


import setproctitle
import random
import numpy as np

setproctitle.setproctitle('GDiff@liuchang')

warnings.filterwarnings("ignore", category=PossibleUserWarning)

def gpumap(gpu):
    if gpu == 0:
        return 4
    elif gpu == 1:
        return 5
    elif gpu == 2:
        return 0
    elif gpu == 3:
        return 1
    elif gpu == 4:
        return 2
    elif gpu == 5:
        return 3
    

@hydra.main(version_base='1.1', config_path='./configs', config_name='config')
def main(cfg: DictConfig):
    torch.set_num_threads(1)
    if cfg.general.guidance_model_type == 'static':

        # from guidance.graphode_data_modify import GraphODEClassifier
        from guidance.graphode_classifier import GraphODEClassifier
        
        # cfg.general.guidance_model_path = cfg.general.guidance_static_model_path

    else:
        raise NotImplementedError

    print('Target beta:', cfg.dataset.beta_c)
    print('Target beta:', cfg.dataset.beta_c)
    print('Target beta:', cfg.dataset.beta_c)
    torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.general.gpu_ids)
    # if cfg.general.gpu_map:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpumap(cfg.general.gpu_ids))
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.general.gpu_ids)
    dataset_config = cfg["dataset"]
    print("beta_c", cfg['dataset']['beta_c'])
    tb_logger = utils.create_folders(cfg)

    if dataset_config["name"] in ['sbm', 'comm-20', 'planar', 'resi']:
        
        if dataset_config['name'] == 'resi':
            # datamodule = ResiDataModule(cfg, dataset_config['mech'])
            print(cfg.dataset.mech)
            datamodule = ResiDataModule(cfg, mech=cfg.dataset.mech)
        
            sampling_metrics = ResiSamplingMetrics(datamodule.dataloaders, dataset_config['beta_c'],datamodule.dataloaders_cond)
        else:
            raise NotImplementedError
        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)
        
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))
    

    # cfg = setup_wandb(cfg)
    if cfg.general.use_guidance:
        from guidance_diffusion_model_discrete_huri import DiscreteDenoisingDiffusion
        print('Use Model: Discrete_huri_guidance')
    else:
        from diffusion_model_discrete_huri import DiscreteDenoisingDiffusion
        print('Use Model: Discrete_huri')
    
    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'test':
        print("[WARNING]: Run is called 'test' -- it will run in debug mode on 20 batches. ")
    elif name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=[0] if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=True,
                      callbacks=callbacks,
                      logger=tb_logger)
    def remove_unexpected_keys(state_dict, keys_to_remove):
        """
        Removes specified keys from the state dictionary.

        Parameters:
        - state_dict (dict): The original state dictionary loaded from the checkpoint.
        - keys_to_remove (list of str): Keys to be removed from the state dictionary.

        Returns:
        - dict: A new state dictionary with the specified keys removed.
        """
        # Create a new dictionary for the cleaned state dict
        cleaned_state_dict = {}
        
        for key, value in state_dict.items():
            if key not in keys_to_remove:
                cleaned_state_dict[key] = value

        return cleaned_state_dict
    def load_model(model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
        key_to_remove = {}
        
        cleaned_state_dict = remove_unexpected_keys(checkpoint['state_dict'], key_to_remove)
        
        model.load_state_dict(cleaned_state_dict)
        return model
    # gpu_tracker.track()
    
    if not cfg.general.test_only:
        if cfg.general.train:
            trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        # gpu_tracker.track()
        if cfg.general.name not in ['debug', 'test']:
            if cfg.general.load:

                loadpath = '../../../ckpts/' + cfg.dataset.dyna + '/uncond-5.pth'

                # model = load_model(model, cfg.general.loadpath)
                model = load_model(model, loadpath)
            
            if cfg.general.use_guidance:

                training_kwargs = {"lr": cfg.train.lr, "input_dim": cfg.train.input_dim, "output_dim": cfg.train.output_dim, "stable_coef": cfg.train.stable_coef, "resi_coef": cfg.train.resi_coef}
                guidance_model = GraphODEClassifier(cfg, dataset_infos, extra_features, domain_features, **training_kwargs)

            

                
                guidance_model_path = '../../../../ode/ckpts/' + cfg.dataset.dyna + '-5/' + 'finetune.ckpt'

                print('Load guidance model!!!!')
                # print('Model path:', cfg.general.guidance_model_path)
                print('Model path:', guidance_model_path)

                # guidance_model = load_model(guidance_model, cfg.general.guidance_model_path)
                guidance_model = load_model(guidance_model, guidance_model_path)
                model.guidance_model = guidance_model

            trainer.test(model, datamodule=datamodule)

    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    # setup_wandb(cfg)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    main()
