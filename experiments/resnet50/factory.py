from typing import List, Dict

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import const, loss, metrics, validation
from src.utils import reduce_mem_usage, DataHandler

from .dataset import CustomDataset
from .model import CustomModel

dh = DataHandler()


def get_fold(cfg: Dict, df: pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()

    for col in [cfg.split.y, cfg.split.groups]:
        if col and col not in df_.columns:
            if cfg.name != "MultilabelStratifiedKFold":
                feat = dh.load(const.FEATURE_DIR / f"{col}.feather")
                df_[col] = feat[col]

            elif cfg.name == "MultilabelStratifiedKFold":
                col = getattr(const, col)
                for c in col:
                    feat = dh.load(const.FEATURE_DIR / f"{c}.feather")
                    df_[c] = feat[c]

    fold_df = pd.DataFrame(index=range(len(df_)))

    weight_list = get_fold_weights(cfg.params.n_splits, cfg.weight)

    fold = getattr(validation, cfg.name)(cfg)
    for fold_, (trn_idx, val_idx) in enumerate(fold.split(df_)):
        fold_df[f"fold_{fold_}"] = 0
        fold_df.loc[val_idx, f"fold_{fold_}"] = weight_list[fold_]
        if cfg.name == "GroupTimeSeriesKFold":
            fold_df.loc[val_idx[-1] + 1 :, f"fold_{fold_}"] = -1

    return fold_df


def get_fold_weights(n_splits: int, weight_type: str) -> List[float]:
    if weight_type == "average":
        weight_list = [1 / n_splits for i in range(n_splits)]

    elif weight_type == "accum_weight":
        sum_ = sum([i + 1 for i in range(n_splits)])
        weight_list = [(i + 1) / sum_ for i in range(n_splits)]

    assert len(weight_list) == n_splits

    return weight_list


def get_metrics(cfg):
    evaluator = getattr(metrics, cfg)
    return evaluator


def fill_dropped(dropped_array, drop_idx):
    filled_array = np.zeros(
        (len(dropped_array) + len(drop_idx), dropped_array.shape[1])
    )
    idx_array = np.arange(len(filled_array))
    use_idx = np.delete(idx_array, drop_idx)
    filled_array[use_idx, :] = dropped_array
    return filled_array


def get_features(features, cfg):
    dfs = [
        dh.load(const.FEATURE_DIR / f"{f}_{cfg.data_type}.feather")
        for f in features
        if f is not None
    ]
    df = pd.concat(dfs, axis=1)
    if cfg.reduce:
        df = reduce_mem_usage(df)
    return df


def get_result(log_name, cfg):
    log_dir = Path(const.LOG_DIR / log_name)

    model_oof = dh.load(log_dir / "oof.npy")
    model_cfg = dh.load(log_dir / "config.yml")

    if model_cfg.common.drop:
        drop_name_list = []
        for drop_name in model_cfg.common.drop:
            if "exploded" not in drop_name:
                drop_name = f"exploded_{drop_name}"
            drop_name_list.append(drop_name)

        drop_idxs = get_drop_idx(drop_name_list)
        model_oof = fill_dropped(model_oof, drop_idxs)

    model_preds = dh.load(log_dir / "raw_preds.npy")

    return model_oof, model_preds


def get_target(cfg):
    target = pd.read_feather(const.FEATURE_DIR / f"{cfg.name}.feather")
    if cfg.convert_type is not None:
        target = getattr(np, cfg.convert_type)(target)
    return target


def get_drop_idx(cfg):
    drop_idx_list = []
    for drop_name in cfg:
        drop_idx = np.load(const.PROCESSED_DATA_DIR / f"{drop_name}.npy")
        drop_idx_list.append(drop_idx)
    all_drop_idx = np.unique(np.concatenate(drop_idx_list))
    return all_drop_idx


def get_nn_model(cfg: Dict, multi_gpu: bool = False, is_train: bool = True):
    model = CustomModel(model_name=cfg.backbone, n_classes=cfg.n_classes, **cfg.params,)

    if multi_gpu and is_train:
        model = nn.DataParallel(model)

    return model


def get_loss(cfg):
    if hasattr(nn, cfg.loss.name):
        loss_ = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    elif hasattr(loss, cfg.loss.name):
        loss_ = getattr(loss, cfg.loss.name)(**cfg.loss.params)
    return loss_


def get_dataloader(df, target_df, cfg):
    dataset = CustomDataset(df=df, target_df=target_df, cfg=cfg)

    loader = DataLoader(dataset, **cfg.loader)
    return loader


def get_optim(cfg, parameters):
    optim = getattr(torch.optim, cfg.optimizer.name)(
        params=parameters, **cfg.optimizer.params
    )
    return optim


def get_scheduler(cfg, optimizer):
    if cfg.scheduler.name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **cfg.scheduler.params,
        )
    else:
        scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.name)(
            optimizer, **cfg.scheduler.params,
        )
    return scheduler
