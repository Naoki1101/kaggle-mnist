import argparse
import datetime
import logging
import warnings
import pandas as pd
from pathlib import Path

from .trainer import NNTrainer
from .factory import get_drop_idx, get_fold
from src import const
from src.utils import DataHandler, Timer, seed_everything

warnings.filterwarnings("ignore")

# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument("--notify", default=const.CONFIG_DIR / "notify.yml")
parser.add_argument("--debug", action="store_true")
parser.add_argument("-m", "--multigpu", action="store_true")
parser.add_argument("-c", "--comment")
options = parser.parse_args()

exp_dir = Path(__file__).resolve().parents[0]
exp_id = str(exp_dir).split("/")[-1]

dh = DataHandler()
cfg = dh.load(exp_dir / "config.yml")

notify_params = dh.load(options.notify)

comment = options.comment
model_name = cfg.model.backbone
now = datetime.datetime.now()
run_name = f"{exp_id}_{now:%Y%m%d%H%M%S}"

logger_path = Path(const.LOG_DIR / run_name)


# ===============
# Main
# ===============
def main():
    t = Timer()
    seed_everything(cfg.seed)

    logger_path.mkdir(exist_ok=True)
    logging.basicConfig(filename=logger_path / "train.log", level=logging.DEBUG)

    dh.save(logger_path / "config.yml", cfg)

    with t.timer("load data"):
        train_df = pd.read_csv(const.INPUT_DATA_DIR / "train.csv")
        # test_df = pd.read_csv(const.INPUT_DATA_DIR / "test.csv")

    with t.timer("make folds"):
        fold_df = get_fold(cfg.validation, train_df)
        if cfg.validation.single:
            fold_df = fold_df[["fold_0"]]
            fold_df /= fold_df["fold_0"].max()

    with t.timer("drop rows"):
        if cfg.drop is not None:
            drop_idx = get_drop_idx(cfg.drop)
            train_df = train_df.drop(drop_idx, axis=0).reset_index(drop=True)
            fold_df = fold_df.drop(drop_idx, axis=0).reset_index(drop=True)

    with t.timer("train model"):
        trainer = NNTrainer(cfg, run_name, options.multigpu, options.debug, comment)
        cv = trainer.train(
            train_df=train_df, target_df=train_df[[const.TARGET_COL]], fold_df=fold_df
        )
        trainer.save()
        # preds = trainer.predict(test_df)

        run_name_cv = f"{run_name}_{cv:.3f}"
        logger_path.rename(const.LOG_DIR / run_name_cv)
        logging.disable(logging.FATAL)

    # with t.timer("make submission"):
    #     make_submission(
    #         run_name=run_name_cv, y_pred=preds, target_name="Label", comp=False
    #     )
    #     if cfg.common.kaggle.submit:
    #         kaggle = Kaggle(cfg.compe.name, run_name_cv)
    #         kaggle.submit(comment)


if __name__ == "__main__":
    main()
