import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album

sys.path.append('../src')
import loss
import models
import metrics
import validation
from utils import reduce_mem_usage, DataHandler
from dataset.custom_dataset import CustomDataset
from models.custom_model import CustomModel

dh = DataHandler()


def get_fold(cfg, df):
    df_ = df.copy()

    for col in [cfg.split.y, cfg.split.groups]:
        if col and col not in df_.columns:
            feat = dh.load(f'../features/{col}.feather')
            df_[col] = feat[col]

    fold_df = pd.DataFrame(index=range(len(df_)))

    if len(cfg.weight) == 1:
        weight_list = [cfg.weight[0] for i in range(cfg.params.n_splits)]
    else:
        weight_list = cfg.weight

    fold = getattr(validation, cfg.name)(cfg)
    for fold_, (trn_idx, val_idx) in enumerate(fold.split(df_)):
        fold_df[f'fold_{fold_}'] = 0
        fold_df.loc[val_idx, f'fold_{fold_}'] = weight_list[fold_]

    return fold_df


def get_model(cfg):
    model = getattr(models, cfg.name)(cfg=cfg)
    return model


def get_metrics(cfg):
    evaluator = getattr(metrics, cfg)
    return evaluator


def fill_dropped(dropped_array, drop_idx):
    filled_array = np.zeros((len(dropped_array) + len(drop_idx), dropped_array.shape[1]))
    idx_array = np.arange(len(filled_array))
    use_idx = np.delete(idx_array, drop_idx)
    filled_array[use_idx, :] = dropped_array
    return filled_array


def get_features(features, cfg):
    dfs = [dh.load(f'../features/{f}_{cfg.data_type}.feather') for f in features if f is not None]
    df = pd.concat(dfs, axis=1)
    if cfg.reduce:
        df = reduce_mem_usage(df)
    return df


def get_result(log_name, cfg):
    log_dir = Path(f'../logs/{log_name}')

    model_oof = dh.load(log_dir / 'oof.npy')
    model_cfg = dh.load(log_dir / 'config.yml')

    if model_cfg.common.drop:
        drop_name_list = []
        for drop_name in model_cfg.common.drop:
            if 'exploded' not in drop_name:
                drop_name = f'exploded_{drop_name}'
            drop_name_list.append(drop_name)

        drop_idxs = get_drop_idx(drop_name_list)
        model_oof = fill_dropped(model_oof, drop_idxs)

    model_preds = dh.load(log_dir / 'raw_preds.npy')

    return model_oof, model_preds


def get_target(cfg):
    target = pd.read_feather(f'../features/{cfg.name}.feather')
    if cfg.convert_type is not None:
        target = getattr(np, cfg.convert_type)(target)
    return target


def get_drop_idx(cfg):
    drop_idx_list = []
    for drop_name in cfg:
        drop_idx = np.load(f'../pickle/{drop_name}.npy')
        drop_idx_list.append(drop_idx)
    all_drop_idx = np.unique(np.concatenate(drop_idx_list))
    return all_drop_idx


def get_ad(cfg, train_df, test_df):
    whole_df = pd.concat([train_df, test_df], axis=0, sort=False).reset_index(drop=True)
    target = np.concatenate([np.zeros(len(train_df)), np.ones(len(test_df))])
    target_df = pd.DataFrame({f'{cfg.data.target.name}': target.astype(int)})
    return whole_df, target_df


def get_nn_model(cfg, is_train=True):
    model = CustomModel(cfg)

    if cfg.model.multi_gpu and is_train:
        model = nn.DataParallel(model)

    return model


def get_loss(cfg):
    if hasattr(nn, cfg.loss.name):
        loss_ = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    elif hasattr(loss, cfg.loss.name):
        loss_ = getattr(loss, cfg.loss.name)(**cfg.loss.params)
    return loss_


def get_dataloader(df, cfg):
    dataset = CustomDataset(df, cfg)
    loader = DataLoader(dataset, **cfg.loader)
    return loader


def get_optim(cfg, parameters):
    optim = getattr(torch.optim, cfg.optimizer.name)(params=parameters, **cfg.optimizer.params)
    return optim


def get_scheduler(cfg, optimizer):
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **cfg.scheduler.params,
        )
    else:
        scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.name)(
            optimizer,
            **cfg.scheduler.params,
        )
    return scheduler


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(album, transform.name):
            return getattr(album, transform.name)
        else:
            return eval(transform.name)
    if cfg.transforms:
        transforms = [get_object(transform)(**transform.params) for name, transform in cfg.transforms.items()]
        return album.Compose(transforms)
    else:
        return None
