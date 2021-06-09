import os
import json
import yaml
from abc import ABCMeta, abstractmethod
from typing import Dict

import dataclasses
import numpy as np
import pandas as pd
import joblib
from collections import OrderedDict
from easydict import EasyDict as edict


class DataProcessor(metaclass=ABCMeta):
    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def save(self, path: str, data) -> None:
        pass


@dataclasses.dataclass
class YmlPrrocessor(DataProcessor):
    def load(self, path: str) -> edict:
        with open(path, "r") as yf:
            yaml_file = yaml.load(yf, Loader=yaml.SafeLoader)
        yaml_file = edict(yaml_file)
        return yaml_file

    def save(self, path: str, data: edict) -> None:
        def represent_odict(dumper, instance):
            return dumper.represent_mapping("tag:yaml.org,2002:map", instance.items())

        yaml.add_representer(OrderedDict, represent_odict)
        yaml.add_representer(edict, represent_odict)

        with open(path, "w") as yf:
            yf.write(yaml.dump(OrderedDict(data), default_flow_style=False))


@dataclasses.dataclass
class CsvProcessor(DataProcessor):
    sep: str = ","

    def load(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path, sep=self.sep)
        return data

    def save(self, path: str, data: pd.DataFrame) -> None:
        data.to_csv(path, index=False)


@dataclasses.dataclass
class FeatherProcessor(DataProcessor):
    def load(self, path: str) -> pd.DataFrame:
        data = pd.read_feather(path)
        return data

    def save(self, path: str, data: pd.DataFrame):
        data.to_feather(path)


@dataclasses.dataclass
class PickleProcessor(DataProcessor):
    def load(self, path: str):
        data = joblib.load(path)
        return data

    def save(self, path: str, data) -> None:
        joblib.dump(data, path, compress=True)


@dataclasses.dataclass
class NpyProcessor(DataProcessor):
    def load(self, path: str) -> np.array:
        data = np.load(path)
        return data

    def save(self, path: str, data: np.array) -> None:
        np.save(path, data)


@dataclasses.dataclass
class JsonProcessor(DataProcessor):
    def load(self, path: str) -> OrderedDict:
        with open(path, "r") as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
        return data

    def save(self, path: str, data: Dict) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)


@dataclasses.dataclass
class SqlProcessor(DataProcessor):
    def load(self, path: str) -> str:
        with open(path, "r") as f:
            query = f.read()

        return query

    def save(self, path: str, data: str) -> None:
        pass


@dataclasses.dataclass
class DataHandler:
    def __post_init__(self):
        self.data_encoder = {
            ".yml": YmlPrrocessor(),
            ".csv": CsvProcessor(sep=","),
            ".tsv": CsvProcessor(sep="\t"),
            ".feather": FeatherProcessor(),
            ".pkl": PickleProcessor(),
            ".npy": NpyProcessor(),
            ".json": JsonProcessor(),
            ".sql": SqlProcessor(),
        }

    def load(self, path: str):
        extension = self._extract_extension(path)
        data = self.data_encoder[extension].load(path)
        return data

    def save(self, path: str, data) -> None:
        extension = self._extract_extension(path)
        self.data_encoder[extension].save(path, data)

    def _extract_extension(self, path: str) -> str:
        extention = os.path.splitext(path)[1]
        return extention


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024 ** 2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    # print("column = ", len(df.columns))
    for i, col in enumerate(df.columns):
        try:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)

                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)

                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)

                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int32)

                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float32)

                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)

                    else:
                        df[col] = df[col].astype(np.float32)

        except Exception as e:
            print(e.args)
            continue

    end_mem = df.memory_usage().sum() / 1024 ** 2
    decreased_mem = 100 * (start_mem - end_mem) / start_mem
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(decreased_mem))

    return df
