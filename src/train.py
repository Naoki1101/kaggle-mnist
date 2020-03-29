import gc
import os
import argparse
import datetime
from datetime import date
from collections import Counter, defaultdict
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
from sklearn.model_selection import train_test_split
import joblib
import torch

from utils import Timer, seed_everything, DataHandler
from utils import send_line, Notion
from trainer import train_cnn, save_png

import warnings
warnings.filterwarnings('ignore')


# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument('--common', default='../configs/common/compe.yml')
parser.add_argument('--notify', default='../configs/common/notify.yml')
parser.add_argument('-m', '--model')
parser.add_argument('-c', '--comment')
options = parser.parse_args()

dh = DataHandler()
cfg = dh.load(options.common)
cfg.update(dh.load(f'../configs/exp/{options.model}.yml'))


# ===============
# Constants
# ===============
comment = options.comment
now = datetime.datetime.now()
model_name = options.model
run_name = f'{model_name}_{now:%Y%m%d%H%M%S}'
notify_params = dh.load(options.notify)

logger_path = Path(f'../logs/{run_name}')


# ===============
# Main
# ===============
def main():
    t = Timer()
    seed_everything(cfg.common.seed)

    logger_path.mkdir()
    logging.basicConfig(filename=logger_path / 'train.log', level=logging.DEBUG)

    dh.save(logger_path / 'config.yml', cfg)

    with t.timer('load data'):
        root = Path(cfg.common.input_root)
        train_df = dh.load(root / cfg.common.img_file)

    with t.timer('drop several rows'):
        if cfg.common.drop_fname is not None:
            drop_idx = dh.load(f'../pickle/{cfg.common.drop_fname}.npy')
            train_df = train_df.drop(drop_idx, axis=0).reset_index(drop=True)

    with t.timer('make folds'):
        train_x_all = train_df.drop('label', axis=1)
        train_y_all = train_df['label']
        trn_x, val_x, trn_y, val_y = train_test_split(train_x_all,
                                                      train_y_all,
                                                      test_size=0.2, 
                                                      shuffle=True, 
                                                      random_state=cfg.common.seed)

    with t.timer('train model'):
        result = train_cnn(trn_x, val_x, trn_y, val_y, cfg)

        np.save(f'../logs/{run_name}/oof.npy', result['best_valid_preds'])

        torch.save(result['best_model'], f'../logs/{run_name}/weight_best.pt')
        save_png(run_name, cfg, result['train_loss'], result['val_loss'], result['val_score'])

    logging.disable(logging.FATAL)
    logger_path.rename(f'../logs/{run_name}_{result["best_val_score"]:.3f}')

    process_minutes = t.get_processing_time()

    with t.timer('notify'):
        message = f'''{model_name}\ncv: {result["best_val_score"]:.3f}\ntime: {process_minutes:.2f}[h]'''
        send_line(notify_params.line.token, message)

        # send_notion(token_v2=notify_params.notion.token_v2,
        #             url=notify_params.notion.url,
        #             name=run_name,
        #             created=now,
        #             model=cfg.model.name,
        #             local_cv=round(result['best_score'], 4),
        #             time_=process_minutes,
        #             comment=comment)



if __name__ == '__main__':
    main()