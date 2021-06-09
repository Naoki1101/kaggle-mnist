import dataclasses
from abc import ABCMeta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from iterstrat import ml_stratifiers

from src import const


@dataclasses.dataclass
class _BaseKFold(metaclass=ABCMeta):
    cfg: Dict

    def __post_init__(self):
        self.fold = None
        if self.cfg.split.y:
            self.y = (lambda x: x if type(x) == str else str(x))(self.cfg.split.y)
        if self.cfg.split.groups:
            self.groups = (lambda x: x if type(x) == str else str(x))(
                self.cfg.split.groups
            )

    def split(self, df: pd.DataFrame):
        pass


@dataclasses.dataclass
class KFold(_BaseKFold):
    cfg: Dict

    def __post_init__(self):
        super(KFold, self).__post_init__()
        self.fold = model_selection.KFold(**self.cfg.params)

    def split(self, df: pd.DataFrame):
        y = (lambda x: df[x] if hasattr(df, x) else None)(self.y)
        return self.fold.split(df, y=y)


@dataclasses.dataclass
class StratifiedKFold(_BaseKFold):
    cfg: Dict

    def __post_init__(self):
        super(StratifiedKFold, self).__post_init__()
        self.fold = model_selection.StratifiedKFold(**self.cfg.params)

    def split(self, df: pd.DataFrame):
        y = (lambda x: df[x] if hasattr(df, x) else None)(self.y)
        return self.fold.split(df, y=y)


@dataclasses.dataclass
class GroupKFold(_BaseKFold):
    cfg: Dict

    def __post_init__(self):
        super(GroupKFold, self).__post_init__()
        self.fold = model_selection.GroupKFold(**self.cfg.params)

    def split(self, df: pd.DataFrame):
        groups = (lambda x: df[x] if hasattr(df, x) else None)(self.groups)
        return self.fold.split(df, groups=groups)


@dataclasses.dataclass
class MultilabelStratifiedKFold(_BaseKFold):
    cfg: Dict

    def __post_init__(self):
        super(MultilabelStratifiedKFold, self).__post_init__()
        self.y = getattr(const, self.y)
        self.fold = ml_stratifiers.MultilabelStratifiedKFold(**self.cfg.params)

    def split(self, df):
        y = df[self.y]
        return self.fold.split(df, y=y)


@dataclasses.dataclass
class StratifiedGroupKFold(_BaseKFold):
    cfg: Dict

    def __post_init__(self):
        super(StratifiedGroupKFold, self).__post_init__()
        self.y_dist = None
        self.all_label = None
        self.train_idx_list = []
        self.valid_idx_list = []

    def split(self, X: pd.DataFrame) -> Tuple[np.array, np.array]:
        y_value_counts = X[self.y].value_counts().sort_index()
        self.all_label, self.y_dist = y_value_counts.index, y_value_counts.values
        df = pd.concat([X[[self.y, self.groups]]], axis=1)
        df.columns = ["y", "groups"]
        count_y_each_group = df.pivot_table(
            index="groups", columns="y", fill_value=0, aggfunc=len
        )
        order = np.argsort(np.sum(count_y_each_group.values, axis=1))[::-1]
        count_y_each_group_sorted = count_y_each_group.iloc[order]

        group_arr = np.zeros(len(count_y_each_group_sorted))
        fold_id_arr = np.zeros(len(count_y_each_group_sorted))
        count_y_each_fold = [
            np.zeros(len(self.all_label)) for i in range(self.cfg.params.n_splits)
        ]

        for i, (g, c) in enumerate(count_y_each_group_sorted.iterrows()):
            best_fold = -1
            min_eval = None
            for fold_id in range(self.cfg.params.n_splits):
                fold_eval = self._eval_y_counts_per_fold(
                    count_y_each_fold, c.values, fold_id
                )
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = fold_id
            count_y_each_fold[best_fold] += c.values
            group_arr[i] = g
            fold_id_arr[i] = best_fold

        for fold_ in range(self.cfg.params.n_splits):
            trn_fold_idx, val_fold_idx = np.array([]), np.array([])
            group_idx = np.where(fold_id_arr == fold_)[0]
            for g in np.sort(group_arr[group_idx]):
                val_fold_idx = np.append(
                    val_fold_idx, X[X[self.groups] == g].index.values
                )

            yield trn_fold_idx, val_fold_idx

    def _eval_y_counts_per_fold(
        self, count_y_each_fold, y_counts: pd.DataFrame, fold_id: str
    ) -> np.array:
        count_y_each_fold[fold_id] += y_counts
        std_per_label = []
        for label_id, label in enumerate(self.all_label):
            label_std = np.std(
                [
                    count_y_each_fold[k][label_id] / self.y_dist[label_id]
                    for k in range(self.cfg.params.n_splits)
                ]
            )
            std_per_label.append(label_std)
        count_y_each_fold[fold_id] -= y_counts
        return np.mean(std_per_label)


# https://stackoverflow.com/questions/51963713/cross-validation-for-grouped-time-series-panel-data
@dataclasses.dataclass
class GroupTimeSeriesKFold(_BaseKFold):
    """
    Time Series cross-validator for a variable number of observations within the time
    unit. In the kth split, it returns first k folds as train set and the (k+1)th fold
    as test set. Indices can be grouped so that they enter the CV fold together.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    """

    cfg: Dict

    def __post_init__(self):
        super(GroupTimeSeriesKFold, self).__post_init__()

    def split(self, df):
        """
        Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is
            the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
            Most often just a time feature.
        Yields
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_splits = self.cfg.params.n_splits
        # y = (lambda x: df[x] if hasattr(df, x) else None)(self.y)
        groups = (lambda x: df[x] if hasattr(df, x) else None)(self.groups)

        # X, y, groups = indexable(df, self.y, self.groups)
        X, groups = indexable(df, groups)
        n_samples = _num_samples(df)
        n_folds = n_splits + 1
        indices = np.arange(n_samples)
        group_counts = np.unique(groups, return_counts=True)[1]
        groups = np.split(indices, np.cumsum(group_counts)[:-1])
        n_groups = _num_samples(groups)
        if n_folds > n_groups:
            raise ValueError(
                (
                    "Cannot have number of folds ={0} greater"
                    " than the number of groups: {1}."
                ).format(n_folds, n_groups)
            )
        test_size = n_groups // n_folds
        test_starts = range(test_size + n_groups % n_folds, n_groups, test_size)
        for test_start in test_starts:
            if self.cfg.params.max_train_size:
                train_start = np.searchsorted(
                    np.cumsum(group_counts[:test_start][::-1])[::-1]
                    < self.cfg.params.max_train_size + 1,
                    True,
                )
                yield (
                    np.concatenate(groups[train_start:test_start]),
                    np.concatenate(groups[test_start : test_start + test_size]),
                )
            else:
                yield (
                    np.concatenate(groups[:test_start]),
                    np.concatenate(groups[test_start : test_start + test_size]),
                )
