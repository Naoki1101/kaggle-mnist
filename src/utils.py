import random
import os
import re
import git
import json
import time
import yaml
import shutil
import datetime
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import joblib
import requests
import dropbox
from notion.client import NotionClient, CollectionRowBlock
from notion.collection import NotionDate
from collections import OrderedDict
from easydict import EasyDict as edict

import const


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


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    # print("column = ", len(df.columns))
    for i, col in enumerate(df.columns):
        try:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)
        except:
            continue

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


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


def make_submission(run_name: str, y_pred: np.array, target_name: str, comp: bool = False):
    output_path = f'../data/output/{run_name}.csv'

    df_sub = pd.read_csv(const.SAMPLE_PATH)
    df_sub[target_name] = y_pred

    if comp:
        output_path += '.gz'
        df_sub.to_csv(output_path, index=False, compression='gzip')
    else:
        df_sub.to_csv(output_path, index=False)


# =============================================================================
# Kaggle API
# =============================================================================
class Kaggle:

    def __init__(self, compe_name, run_name):
        self.compe_name = compe_name
        self.run_name = run_name
        self.slug = re.sub('[_.]', '-', run_name)
        self.log_dir = f'../logs/{run_name}'
        self.notebook_dir = f'../notebooks/{run_name}'
        self.username = 'narimatsu'

    def submit(self, comment):
        cmd = f'kaggle competitions submit -c {self.compe_name} -f ../data/output/{self.run_name}.csv  -m "{comment}"'
        self._run(cmd)
        print(f'\n\nhttps://www.kaggle.com/c/{self.compe_name}/submissions\n\n')

    def create_dataset(self):
        cmd_init = f'kaggle datasets init -p {self.log_dir}'
        cmd_create = f'kaggle datasets create -p {self.log_dir} -q'
        self._run(cmd_init)
        self._insert_dataset_metadata()
        self._run(cmd_create)

    def push_notebook(self):
        time.sleep(20)
        self._prepare_notebook_dir()
        cmd_init = f'kaggle kernels init -p {self.notebook_dir}'
        cmd_push = f'kaggle kernels push -p {self.notebook_dir}'
        self._run(cmd_init)
        self._insert_kernel_metadata()
        self._run(cmd_push)

    def _run(self, cmd):
        os.system(cmd)

    def _insert_dataset_metadata(self):
        jh = JsonProcessor()

        metadata_path = f'{self.log_dir}/dataset-metadata.json'
        meta = jh.load(metadata_path)
        meta['title'] = self.run_name
        meta['id'] = re.sub('INSERT_SLUG_HERE', f'sub-{self.slug}', meta['id'])
        jh.save(metadata_path, meta)

    def _insert_kernel_metadata(self):
        jh = JsonProcessor()

        metadata_path = f'{self.notebook_dir}/kernel-metadata.json'
        meta = jh.load(metadata_path)
        meta['title'] = self.run_name
        meta['id'] = re.sub('INSERT_KERNEL_SLUG_HERE', self.slug, meta['id'])
        meta['code_file'] = f'sub_{self.run_name}.ipynb'
        meta['language'] = 'python'
        meta['kernel_type'] = 'notebook'
        meta['is_private'] = 'true'
        meta['enable_gpu'] = 'true'
        meta['enable_internet'] = 'false'
        meta['dataset_sources'] = [f'{self.username}/sub-{self.slug}']
        meta['competition_sources'] = [f'{self.compe_name}']
        meta['kernel_sources'] = []

        jh.save(metadata_path, meta)

    def _prepare_notebook_dir(self):
        Path(f'../notebooks/{self.run_name}').mkdir(exist_ok=True)
        if 'resnet' in self.run_name:
            shutil.copy(
                '../notebooks/resnet_for_inference.ipynb',
                f'{self.notebook_dir}/sub_{self.run_name}.ipynb'
            )
        elif 'se_resnext' in self.run_name:
            shutil.copy(
                '../notebooks/se_resnext_for_inference.ipynb',
                f'{self.notebook_dir}/sub_{self.run_name}.ipynb'
            )
        elif 'efficientnet' in self.run_name:
            shutil.copy(
                '../notebooks/efficientnet_for_inference.ipynb',
                f'{self.notebook_dir}/sub_{self.run_name}.ipynb'
            )
        elif 'resnest' in self.run_name:
            shutil.copy(
                '../notebooks/resnest_for_inference.ipynb',
                f'{self.notebook_dir}/sub_{self.run_name}.ipynb'
            )


# =============================================================================
# Notification
# =============================================================================
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


class Notificator:
    def __init__(self, run_name, model_name, cv, process_time, comment, params):
        self.run_name = run_name
        self.model_name = model_name
        self.cv = cv
        self.process_time = process_time
        self.comment = comment
        self.params = params

    def send_line(self):
        if self.params.line.token:
            endpoint = 'https://notify-api.line.me/api/notify'
            message = f'''\n{self.model_name}\ncv: {self.cv}\ntime: {self.process_time}[min]'''
            payload = {'message': message}
            headers = {'Authorization': 'Bearer {}'.format(self.params.line.token)}
            requests.post(endpoint, data=payload, headers=headers)

    def send_notion(self):
        if os.environ.get('NOTION_TOKEN'):
            notion = Notion(token=os.environ.get('NOTION_TOKEN'))
            notion.set_url(url=self.params.notion.url)
            notion.insert_rows({
                'name': self.run_name,
                'created': datetime.datetime.now(),
                'model': self.model_name,
                'local_cv': self.cv,
                'time': self.process_time,
                'comment': self.comment
            })


class Git:
    def __init__(self):
        os.chdir('../')
        self.repo = git.Repo()

    def push(self, comment):
        self.repo.git.add('.')
        self.repo.git.commit('-m', f'{comment}')
        origin = self.repo.remote(name='origin')
        origin.push()


def transfar_dropbox(input_path, output_path, token):
    dbx = dropbox.Dropbox(token)
    dbx.users_get_current_account()
    with open(input_path, 'rb') as f:
        dbx.files_upload(f.read(), output_path)
