import gc
import sys
import time
import logging
import matplotlib.pyplot as plt

import optuna
import numpy as np
from pathlib import Path
from fastprogress import master_bar, progress_bar
import torch

sys.path.append('../src')
import factory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NNTrainer:
    def __init__(self, run_name, fold_df, cfg):
        self.run_name = run_name
        self.cfg = cfg
        self.fold_df = fold_df
        self.oof = None
        self.raw_preds = None

    def train(self, train_df, target_df):
        oof = np.zeros((len(train_df), self.cfg.model.n_classes))
        cv = 0

        for fold_, col in enumerate(self.fold_df.columns):
            print(f'\n========================== FOLD {fold_} ... ==========================\n')
            logging.debug(f'\n========================== FOLD {fold_} ... ==========================\n')

            trn_x, val_x = train_df[self.fold_df[col] == 0], train_df[self.fold_df[col] > 0]
            val_y = target_df[self.fold_df[col] > 0].values

            train_loader = factory.get_dataloader(trn_x, self.cfg.data.train)
            valid_loader = factory.get_dataloader(val_x, self.cfg.data.valid)

            model = factory.get_nn_model(self.cfg).to(device)

            criterion = factory.get_loss(self.cfg)
            optimizer = factory.get_optim(self.cfg, model.parameters())
            scheduler = factory.get_scheduler(self.cfg, optimizer)

            best_epoch = -1
            best_val_score = -np.inf
            mb = master_bar(range(self.cfg.model.epochs))

            train_loss_list = []
            val_loss_list = []
            val_score_list = []

            for epoch in mb:
                start_time = time.time()

                model, avg_loss = self._train_epoch(model, train_loader, criterion, optimizer, mb)

                valid_preds, avg_val_loss = self._val_epoch(model, valid_loader, criterion)

                val_score = factory.get_metrics(self.cfg.common.metrics.name)(val_y, valid_preds)

                train_loss_list.append(avg_loss)
                val_loss_list.append(avg_val_loss)
                val_score_list.append(val_score)

                if self.cfg.scheduler.name != 'ReduceLROnPlateau':
                    scheduler.step()
                elif self.cfg.scheduler.name == 'ReduceLROnPlateau':
                    scheduler.step(avg_val_loss)

                elapsed = time.time() - start_time
                mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} val_score: {val_score:.4f} time: {elapsed:.0f}s')
                logging.debug(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} val_score: {val_score:.4f} time: {elapsed:.0f}s')

                if val_score > best_val_score:
                    best_epoch = epoch + 1
                    best_val_score = val_score
                    best_valid_preds = valid_preds
                    if self.cfg.model.multi_gpu:
                        best_model = model.module.state_dict()
                    else:
                        best_model = model.state_dict()

            oof[val_x.index, :] = best_valid_preds
            cv += best_val_score * self.fold_df[col].max()

            torch.save(best_model, f'../logs/{self.run_name}/weight_best_{fold_}.pt')
            self._save_loss_png(train_loss_list, val_loss_list, val_score_list, fold_)

            print(f'\nEpoch {best_epoch} - val_score: {best_val_score:.4f}')
            logging.debug(f'\nEpoch {best_epoch} - val_score: {best_val_score:.4f}')

        print('\n\n===================================\n')
        print(f'CV: {cv:.6f}')
        logging.debug(f'\n\nCV: {cv:.6f}')
        print('\n===================================\n\n')

        self.oof = oof.reshape(-1, 5)

        return cv

    def predict(self, test_df):
        all_preds = np.zeros((len(test_df), self.cfg.model.n_classes * len(self.fold_df.columns)))
        result_preds = np.zeros((len(test_df), self.cfg.model.n_classes))

        for fold_num, col in enumerate(self.fold_df.columns):
            preds = self._predict_fold(fold_num, test_df)
            all_preds[:, fold_num * self.cfg.model.n_classes: (fold_num + 1) * self.cfg.model.n_classes] = preds

        for i in range(self.cfg.model.n_classes):
            preds_col_idx = [i + self.cfg.model.n_classes * j for j in range(len(self.fold_df.columns))]
            result_preds[:, i] = np.mean(all_preds[:, preds_col_idx], axis=1)

        self.raw_preds = result_preds

        result_preds_class = np.argmax(result_preds, axis=1)

        return result_preds_class

    def save(self):
        log_dir = Path(f'../logs/{self.run_name}')
        np.save(log_dir / 'oof.npy', self.oof)
        np.save(log_dir / 'raw_preds.npy', self.raw_preds)

    def _train_epoch(self, model, train_loader, criterion, optimizer, mb):
        model.train()
        avg_loss = 0.

        for images, labels in progress_bar(train_loader, parent=mb):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images.float())
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        del images, labels; gc.collect()
        return model, avg_loss

    def _val_epoch(self, model, valid_loader, criterion):
        model.eval()
        valid_preds = np.zeros((len(valid_loader.dataset),
                                self.cfg.model.n_classes * self.cfg.data.valid.tta.iter_num))
        valid_preds_tta = np.zeros((len(valid_preds), self.cfg.model.n_classes))

        avg_val_loss = 0.
        valid_batch_size = valid_loader.batch_size

        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)

                preds = model(images.float())

                loss = criterion(preds, labels)
                valid_preds[i * valid_batch_size: (i + 1) * valid_batch_size, :] = preds.cpu().detach().numpy()
                avg_val_loss += loss.item() / len(valid_loader)

        for i in range(self.cfg.model.n_classes):
            preds_col_idx = [i + self.cfg.model.n_classes * j for j in range(self.cfg.data.valid.tta.iter_num)]
            valid_preds_tta[:, i] = np.mean(valid_preds[:, preds_col_idx], axis=1).reshape(-1)

        return valid_preds_tta, avg_val_loss

    def _predict_fold(self, fold_num, test_df):
        test_loader = factory.get_dataloader(test_df, self.cfg.data.test)

        test_preds = np.zeros((len(test_loader.dataset),
                               self.cfg.model.n_classes * self.cfg.data.test.tta.iter_num))
        test_preds_tta = np.zeros((len(test_preds), self.cfg.model.n_classes))

        test_batch_size = test_loader.batch_size

        model = factory.get_nn_model(self.cfg, is_train=False).to(device)
        model.load_state_dict(torch.load(f'../logs/{self.run_name}/weight_best_{fold_num}.pt'))

        model.eval()
        for t in range(self.cfg.data.test.tta.iter_num):
            with torch.no_grad():
                for i, images in enumerate(test_loader):
                    images = images.to(device)

                    preds = model(images.float())
                    test_preds[i * test_batch_size: (i + 1) * test_batch_size, t * self.cfg.model.n_classes: (t + 1) * self.cfg.model.n_classes] = preds.cpu().detach().numpy()

        for i in range(self.cfg.model.n_classes):
            preds_col_idx = [i + self.cfg.model.n_classes * j for j in range(self.cfg.data.test.tta.iter_num)]
            test_preds_tta[:, i] = np.mean(test_preds[:, preds_col_idx], axis=1).reshape(-1)

        return test_preds_tta

    def _save_loss_png(self, train_loss_list, val_loss_list, val_score_list, fold_num):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        ax1.plot(range(len(train_loss_list)), train_loss_list, color='blue', linestyle='-', label='train_loss')
        ax1.plot(range(len(val_loss_list)), val_loss_list, color='green', linestyle='-', label='val_loss')
        ax1.legend()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        ax1.set_title(f'Training and validation {self.cfg.loss.name}')
        ax1.grid()

        ax2.plot(range(len(val_score_list)), val_score_list, color='blue', linestyle='-', label='val_score')
        ax2.legend()
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('score')
        ax2.set_title('Training and validation score')
        ax2.grid()

        plt.savefig(f'../logs/{self.run_name}/learning_curve_{fold_num}.png')


def opt_ensemble_weight(cfg, y_true, oof_list, metric):
    def objective(trial):
        p_list = [0 for i in range(len(oof_list))]
        for i in range(len(oof_list) - 1):
            p_list[i] = trial.suggest_discrete_uniform(f'p{i}', 0.0, 1.0 - sum(p_list), 0.01)
        p_list[-1] = round(1 - sum(p_list[:-1]), 2)

        y_pred = np.zeros(len(y_true))
        for i in range(len(oof_list)):
            y_pred += oof_list[i] * p_list[i]

        return metric(y_true, y_pred)

    study = optuna.create_study(direction=cfg.opt_params.direction)
    study.optimize(objective, timeout=cfg.opt_params.timeout)
    best_params = list(study.best_params.values())
    best_weight = best_params + [round(1 - sum(best_params), 2)]

    return best_weight
