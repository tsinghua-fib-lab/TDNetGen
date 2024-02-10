import torch
import hydra
import omegaconf
from omegaconf import DictConfig
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import warnings
from setproctitle import setproctitle
import os
import pathlib
import sys
sys.path.append('../data_enhance')
import utils
from datasets.spectre_dataset_semi import SpectreDatasetInfos,ResiGuideDataModule
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from setproctitle import setproctitle
import logging

# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
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


@hydra.main(config_path='./configs_guide', config_name='config')
def main(cfg: DictConfig):

    

    logger = logging.getLogger(__name__)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.general.gpu_ids)
    dataset_config = cfg["dataset"]
    
    tb_logger = utils.create_folders(cfg)

    assert dataset_config.name == 'resi'
    print(cfg.dataset.mech)

    datamodule_semi = ResiGuideDataModule(cfg, type='ode_train', mech=cfg.dataset.mech)

    datamodule_full = ResiGuideDataModule(cfg, type='resinf_train', mech=cfg.dataset.mech)

    datamodule_finetune = ResiGuideDataModule(cfg, type='finetune', mech=cfg.dataset.mech)

    datamodule_test = ResiGuideDataModule(cfg, type='test', mech=cfg.dataset.mech)
    
    dataset_infos_semi = SpectreDatasetInfos(datamodule_semi, dataset_config)
    dataset_infos_full = SpectreDatasetInfos(datamodule_full, dataset_config)

    dataset_infos_finetune = SpectreDatasetInfos(datamodule_finetune, dataset_config)
    

    if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos_full)
    else:
        extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()
    
    dataset_infos_full.compute_input_output_dims(datamodule=datamodule_full, extra_features=extra_features,
                                                domain_features=domain_features)
    
    training_kwargs = {"lr": cfg.train.lr, "input_dim": cfg.train.input_dim, "output_dim": cfg.train.output_dim, "stable_coef": cfg.train.stable_coef, "resi_coef": cfg.train.resi_coef, "sys_logger": logger}


    from graphode_four_stage_check import GraphODEClassifier

    model = GraphODEClassifier(cfg,dataset_infos_full,extra_features,domain_features, **training_kwargs)

    name = cfg.general.name
    


    callbacks_semi = []
    if cfg.train.early_stopping:
        callbacks_semi.append(EarlyStopping(monitor='val_loss', mode='min', patience=cfg.train.patience))
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='semi_{epoch}',
                                              monitor='val_loss',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='semi_last', every_n_epochs=1)
        callbacks_semi.append(last_ckpt_save)
        callbacks_semi.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks_semi.append(ema_callback)

    callbacks_full = []

    if cfg.train.early_stopping:
        callbacks_full.append(EarlyStopping(monitor='val_loss', mode='min', patience=cfg.train.patience))
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='full_{epoch}',
                                              monitor='val_resi_f1',
                                              auto_insert_metric_name=True,
                                              save_top_k=20,
                                              mode='max',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='full_last', every_n_epochs=1)
        callbacks_full.append(last_ckpt_save)
        callbacks_full.append(checkpoint_callback)
    
    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks_full.append(ema_callback)

    callbacks_finetune = []

    if cfg.train.early_stopping:
        callbacks_finetune.append(EarlyStopping(monitor='val_loss', mode='min', patience=cfg.train.patience))
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='finetune_{epoch}',
                                              monitor='val_resi_f1',
                                              auto_insert_metric_name=True,
                                              save_top_k=20,
                                              mode='max',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='finetune_last', every_n_epochs=1)
        callbacks_finetune.append(last_ckpt_save)
        callbacks_finetune.append(checkpoint_callback)
    
    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks_finetune.append(ema_callback)

    callbacks_enhancement = []

    if cfg.train.early_stopping:
        callbacks_enhancement.append(EarlyStopping(monitor='val_loss', mode='min', patience=cfg.train.patience))
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='enhance_{epoch}',
                                              monitor='val_resi_f1',
                                              auto_insert_metric_name=True,
                                              save_top_k=20,
                                              mode='max',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='enhance_last', every_n_epochs=1)
        callbacks_enhancement.append(last_ckpt_save)
        callbacks_enhancement.append(checkpoint_callback)
    
    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks_enhancement.append(ema_callback)



    trainer_semi = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=[0] if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_semi_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=True,
                      callbacks=callbacks_semi,
                      logger=tb_logger)

    trainer_full = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=[0] if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_full_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=True,
                      callbacks=callbacks_full,
                      logger=tb_logger)
    
    trainer_finetune = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=[0] if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_finetune_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=True,
                      callbacks=callbacks_finetune,
                      logger=tb_logger)
    
    trainer_enhancement = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=[0] if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_enhancement_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=True,
                      callbacks=callbacks_enhancement,
                      logger=tb_logger)
    

    
    if not cfg.general.test_only:

        model.type = 'ode_train'


        if cfg.dataset.load_semi:

            model = load_model(model, cfg.dataset.semi_load_data_path)

            print('Load semi data from file. Start training......')

        else:
            
            trainer_semi.fit(model, datamodule=datamodule_semi)

    
        model.type = 'resinf_train'
        
        trainer_full.fit(model, datamodule=datamodule_full)
        

        model.type = 'finetune'

        trainer_finetune.fit(model, datamodule=datamodule_finetune)
        

    else:

    
        path = cfg.dataset.loadpath
        
        model = load_model(model, path)

        model.type = 'resinf_train'
        

        if cfg.dataset.self_training:
            
            datamodule_selftraining = ResiGuideDataModule(cfg, type='self_training', mech=cfg.dataset.mech)

            trainer_full.test(model, datamodule=datamodule_selftraining)
            

        else:       

            trainer_full.test(model, datamodule=datamodule_test)
        

        print('Test end. Training with enhanced data................')

        if cfg.general.re_train:
            
            datamodule_retrain = ResiGuideDataModule(cfg, type='retrain', mech=cfg.dataset.mech)

            model.type = 'retrain'

            trainer_enhancement.fit(model, datamodule=datamodule_retrain)
            
            model.type = 'resinf_train'

            trainer_full.test(model, datamodule=datamodule_test)

            

if __name__ == '__main__':

   
    
    main()

