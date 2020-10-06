import argparse
import datetime
import logging
import warnings
import pandas as pd
from pathlib import Path

import factory
import const
from trainer import NNTrainer
from utils import (DataHandler, Timer, seed_everything, 
                   Kaggle, make_submission)

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

notify_params = dh.load(options.notify)

comment = options.comment
model_name = options.model
now = datetime.datetime.now()
run_name = f'{model_name}_{now:%Y%m%d%H%M%S}'

logger_path = Path(f'../logs/{run_name}')


# ===============
# Main
# ===============
def main():
    t = Timer()
    seed_everything(cfg.common.seed)

    logger_path.mkdir(exist_ok=True)
    logging.basicConfig(filename=logger_path / 'train.log', level=logging.DEBUG)

    dh.save(logger_path / 'config.yml', cfg)

    with t.timer('load data'):
        train_df = pd.read_csv(const.TRAIN_PATH)
        test_df = pd.read_csv(const.TEST_PATH)

    with t.timer('make folds'):
        fold_df = factory.get_fold(cfg.validation, train_df)
        if cfg.validation.single:
            fold_df = fold_df[['fold_0']]
            fold_df /= fold_df['fold_0'].max()

    with t.timer('drop index'):
        if cfg.common.drop is not None:
            drop_idx = factory.get_drop_idx(cfg.common.drop)
            train_df = train_df.drop(drop_idx, axis=0).reset_index(drop=True)
            fold_df = fold_df.drop(drop_idx, axis=0).reset_index(drop=True)

    with t.timer('train model'):
        trainer = NNTrainer(run_name, fold_df, cfg)
        cv = trainer.train(train_df=train_df, target_df=train_df[const.TARGET_COL])
        preds = trainer.predict(test_df)
        trainer.save()

        run_name_cv = f'{run_name}_{cv:.3f}'
        logger_path.rename(f'../logs/{run_name_cv}')
        logging.disable(logging.FATAL)

    with t.timer('make submission'):
        make_submission(run_name=run_name_cv,
                        y_pred=preds,
                        target_name='Label',
                        comp=False)
        if cfg.common.kaggle.submit:
            kaggle = Kaggle(cfg.compe.name, run_name_cv)
            kaggle.submit(comment)


if __name__ == '__main__':
    main()
