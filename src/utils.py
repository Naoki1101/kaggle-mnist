import random
import os
import json
import time
import yaml
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import torch
import joblib
import requests
import dropbox
from notion.client import NotionClient, CollectionRowBlock
from notion.collection import NotionDate
from collections import OrderedDict
from easydict import EasyDict as edict


class Timer:

    def __init__(self):
        self.processing_time = 0

    @contextmanager
    def timer(self, name):
        t0 = time.time()
        yield
        t1 = time.time()
        processing_time = t1 - t0
        self.processing_time += round(processing_time, 2)
        if self.processing_time < 60:
            print(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time:.2f} sec)')
        elif self.processing_time < 3600:
            print(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 60:.2f} min)')
        else:
            print(f'[{name}] done in {processing_time:.0f} s (Total: {self.processing_time / 3600:.2f} hour)')

    def get_processing_time(self):
        return round(self.processing_time / 60, 2)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# =============================================================================
# Data Processor
# =============================================================================
class DataProcessor(metaclass=ABCMeta):

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def save(self, path, data):
        pass


class YmlPrrocessor(DataProcessor):

    def load(self, path):
        # yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        #                      lambda loader,
        #                      node: OrderedDict(loader.construct_pairs(node)))

        with open(path, 'r') as yf:
            yaml_file = yaml.load(yf, Loader=yaml.SafeLoader)
        return edict(yaml_file)

    def save(self, path, data):
        def represent_odict(dumper, instance):
            return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())

        yaml.add_representer(OrderedDict, represent_odict)
        yaml.add_representer(edict, represent_odict)

        with open(path, 'w') as yf:
            yf.write(yaml.dump(OrderedDict(data), default_flow_style=False))


class CsvProcessor(DataProcessor):

    def __init__(self, sep):
        self.sep = sep

    def load(self, path, sep=','):
        data = pd.read_csv(path, sep=sep)
        return data

    def save(self, path, data):
        data.to_csv(path, index=False)


class FeatherProcessor(DataProcessor):

    def load(self, path):
        data = pd.read_feather(path)
        return data

    def save(self, path, data):
        data.to_feather(path)


class PickleProcessor(DataProcessor):

    def load(self, path):
        data = joblib.load(path)
        return data

    def save(self, path, data):
        joblib.dump(data, path, compress=True)


class NpyProcessor(DataProcessor):

    def load(self, path):
        data = np.load(path)
        return data

    def save(self, path, data):
        np.save(path, data)


class JsonProcessor(DataProcessor):

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
        return data

    def save(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)


class DataHandler:

    def __init__(self):
        self.data_encoder = {
            '.yml': YmlPrrocessor(),
            '.csv': CsvProcessor(sep=','),
            '.tsv': CsvProcessor(sep='\t'),
            '.feather': FeatherProcessor(),
            '.pkl': PickleProcessor(),
            '.npy': NpyProcessor(),
            '.json': JsonProcessor(),
        }

    def load(self, path):
        extension = self._extract_extension(path)
        data = self.data_encoder[extension].load(path)
        return data

    def save(self, path, data):
        extension = self._extract_extension(path)
        self.data_encoder[extension].save(path, data)

    def _extract_extension(self, path):
        return os.path.splitext(path)[1]


def make_submission(y_pred, target_name, sample_path, output_path, comp=False):
    df_sub = pd.read_feather(sample_path)
    df_sub[[f'F{i + 1}' for i in range(28)]] = y_pred.reshape(60980, 28)
    if comp:
        output_path += '.gz'
        df_sub.to_csv(output_path, index=False, compression='gzip')
    else:
        df_sub.to_csv(output_path, index=False)


# =============================================================================
# Kaggle API
# =============================================================================
class Kaggle:

    def __init__(self, compe_name):
        self.compe_name = compe_name

    def submit(self, data_path, run_name, cv):
        cmd = f'kaggle competitions submit -c {self.compe_name} -f {data_path}  -m "{run_name}_{cv:.4f}"'
        self._run(cmd)
        print(f'\n\nhttps://www.kaggle.com/c/{self.compe_name}/submissions\n\n')

    def download_data(self, to_path):
        cmd = f'kaggle competitions download -c {self.compe_name} -p {to_path}'
        self._run(cmd)

    def create_dataset(self, data_path):
        cmd = f'kaggle datasets create -p {data_path}'
        self._run(cmd)

    def _run(self, cmd):
        os.system(cmd)


# =============================================================================
# Notification
# =============================================================================
def send_line(line_token, message):
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)


class Notion:
    def __init__(self, token):
        self.client = NotionClient(token_v2=token)
        self.url = None

    def set_url(self, url):
        self.url = url

    def get_table(self, dropna=False):
        table = self.client.get_collection_view(self.url)

        rows = []
        for row in table.collection.get_rows():
            rows.append(self._get_row_item(row))

        table_df = pd.DataFrame(rows, columns=list(row.get_all_properties().keys()))
        if dropna:
            table_df = table_df.dropna().reset_index(drop=True)
        return table_df

    def _get_row_item(self, row):
        items = []
        for col, item in row.get_all_properties().items():
            type_ = type(item)
            item = row.get_property(identifier=col)
            if type_ not in [list, NotionDate]:
                items.append(item)
            elif type_ == list:
                items.append(' '.join(item))
            elif type_ == NotionDate:
                items.append(item.__dict__['start'])
        return items

    def insert_rows(self, item_dict):
        table = self.client.get_collection_view(self.url)
        row = self._create_new_record(table)

        for col_name, value in item_dict.items():
            row.set_property(identifier=col_name, val=value)

    def _create_new_record(self, table):
        row_id = self.client.create_record('block', parent=table.collection, type='page')
        row = CollectionRowBlock(self.client, row_id)

        with self.client.as_atomic_transaction():
            for view in self.client.get_block(table.get("parent_id")).views:
                view.set("page_sort", view.get("page_sort", []) + [row_id])

        return row


def transfar_dropbox(input_path, output_path, token):
    dbx = dropbox.Dropbox(token)
    dbx.users_get_current_account()
    with open(input_path, 'rb') as f:
        dbx.files_upload(f.read(), output_path)