from abc import ABCMeta

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split


class SingleFold:
    def __init__(self, cfg):
        self.cfg = cfg
        self.params = cfg.params
        self.params['test_size'] = 1 / cfg.params['n_splits']
        self.y = (lambda x: x if type(x) == str else str(x))(cfg.split.y)
        self.groups = (lambda x: x if type(x) == str else str(x))(cfg.split.groups)
        del self.params['n_splits']

    def split(self, df):
        trn_idx_list, val_idx_list = train_test_split(range(len(df)),
                                                      stratify=(lambda x: df[x] if hasattr(df, x) else None)(self.y),
                                                      **self.params)
        yield trn_idx_list, val_idx_list


class _BaseKFold(metaclass=ABCMeta):
    def __init__(self, cfg):
        self.cfg = cfg
        self.fold = None
        self.y = (lambda x: x if type(x) == str else str(x))(cfg.split.y)
        self.groups = (lambda x: x if type(x) == str else str(x))(cfg.split.groups)

    def split(self, df):
        y = (lambda x: df[x] if hasattr(df, x) else None)(self.y)
        groups = (lambda x: df[x] if hasattr(df, x) else None)(self.groups)
        return self.fold.split(df, y=y, groups=groups)


class KFold(_BaseKFold):
    def __init__(self, cfg):
        super(KFold, self).__init__(cfg)
        self.fold = model_selection.KFold(**cfg.params)


class StratifiedKFold(_BaseKFold):
    def __init__(self, cfg):
        super(StratifiedKFold, self).__init__(cfg)
        self.fold = model_selection.StratifiedKFold(**cfg.params)


class GroupKFold(_BaseKFold):
    def __init__(self, cfg):
        super(GroupKFold, self).__init__(cfg)
        self.fold = model_selection.GroupKFold(**cfg.params)


class StratifiedGroupKFold:
    def __init__(self, cfg):
        self.cfg = cfg
        self.y = cfg.split.y
        self.groups = cfg.split.groups
        self.y_dist = None
        self.all_label = None
        self.train_idx_list = []
        self.valid_idx_list = []

    def split(self, X):
        y_value_counts = X[self.y].value_counts().sort_index()
        self.all_label, self.y_dist = y_value_counts.index, y_value_counts.values
        df = pd.concat([X[[self.y, self.groups]]], axis=1)
        df.columns = ['y', 'groups']
        count_y_each_group = df.pivot_table(index='groups', columns='y', fill_value=0, aggfunc=len)
        order = np.argsort(np.sum(count_y_each_group.values, axis=1))[::-1]
        count_y_each_group_sorted = count_y_each_group.iloc[order]

        group_arr = np.zeros(len(count_y_each_group_sorted))
        fold_id_arr = np.zeros(len(count_y_each_group_sorted))
        count_y_each_fold = [np.zeros(len(self.all_label)) for i in range(self.cfg.params.n_splits)]

        for i, (g, c) in enumerate(count_y_each_group_sorted.iterrows()):
            best_fold = -1
            min_eval = None
            for fold_id in range(self.cfg.params.n_splits):
                fold_eval = self._eval_y_counts_per_fold(count_y_each_fold, c.values, fold_id)
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
                val_fold_idx = np.append(val_fold_idx, X[X[self.groups] == g].index.values)

            yield trn_fold_idx, val_fold_idx

    def _eval_y_counts_per_fold(self, count_y_each_fold, y_counts, fold_id):
        count_y_each_fold[fold_id] += y_counts
        std_per_label = []
        for label_id, label in enumerate(self.all_label):
            label_std = np.std([count_y_each_fold[k][label_id] / self.y_dist[label_id] for k in range(self.cfg.params.n_splits)])
            std_per_label.append(label_std)
        count_y_each_fold[fold_id] -= y_counts
        return np.mean(std_per_label)
