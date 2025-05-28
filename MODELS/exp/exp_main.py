import logging
from sklearn.model_selection import TimeSeriesSplit
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
from data.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.vis import plot_loss, visual_interval
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.losses import R2Loss

warnings.filterwarnings('ignore')

# This is the main experiment class for time series forecasting models.
class Exp_Main(Exp_Basic):
    """
    Main class for time series forecasting experiments.
    This class extends the Exp_Basic class and implements methods for building the model,
    getting data, selecting the optimizer and criterion, training, validating, and testing the model.
    It supports both k-fold cross-validation and standard training.
    Attributes:
        args: Argument parser containing model parameters.
        model_dict: Dictionary mapping model names to their respective classes.
        device: The device (CPU or GPU) on which the model will run.
        model: The initialized model instance.
    """
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-2)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        # criterion = R2Loss()    
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        """
        Validate the model on the validation dataset.
        Args:
            vali_data: Validation dataset.
            vali_loader: DataLoader for the validation dataset.
            criterion: Loss function to evaluate the model's performance.
        Returns:
            avg_loss: Average validation loss.
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                f_dim = -2 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                pred = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]


                loss_lower = criterion(pred[:, :, 0], true[:, :, 0])
                loss_upper = criterion(pred[:, :, 1], true[:, :, 1])
                loss = loss_lower + loss_upper
                
                total_loss.append(loss.item())

        avg_loss = np.average(total_loss)
        self.model.train()

        return avg_loss

    def train(self, setting, plot=True):
        """
        Train the model based on the specified setting.
        Args:
            setting: A string indicating the training setting (e.g., 'kfold', 'standard').
            plot: A boolean indicating whether to plot the training and validation losses.
        Returns:
            val_losses: A list of validation losses for each fold if k-fold is used, or the final validation loss if standard training is used.
        """
        if self.args.kfold > 0:
            return self.train_kfold(setting, plot=plot)
        elif self.args.kfold == 0:
            return self.train_standard(setting, plot=plot)


    def train_kfold(self, setting, plot=True, checkpoint_base="./checkpoints/tmp"):
        """
        Train the model using k-fold cross-validation.
        Args:
            setting: A string indicating the training setting (e.g., 'kfold').
            plot: A boolean indicating whether to plot the training and validation losses.
            checkpoint_base: Base directory for saving checkpoints.
        Returns:
            val_losses: A list of validation losses for each fold.
        """
        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"[MODEL] Trainable parameters: {model_params:,}")
        
        # Get training, validation, and test data loaders
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        dataset = train_data
        indices = np.arange(len(dataset))
        tscv = TimeSeriesSplit(n_splits=self.args.kfold)

        val_losses = []
        
        # If using k-fold, we need to split the dataset into k folds
        for fold, (train_idx, val_idx) in enumerate(tscv.split(indices)):
            
            fold_start = time.time()  # <-- Đo thời gian riêng từng fold
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()


            print(f"{'='*50} Fold {fold+1}/{self.args.kfold} {'='*50}")

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            vali_data = torch.utils.data.Subset(dataset, val_idx)
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=self.args.batch_size, shuffle=False)
            vali_loader = torch.utils.data.DataLoader(vali_data, batch_size=self.args.batch_size, shuffle=False)

            # print(f"[INFO] Dataset Shapes:")
            # print(f"  ➤ Train set  : {len(train_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in train_loader][0]}")
            # print(f"  ➤ Val set    : {len(vali_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in vali_loader][0]}")
            # print(f"  ➤ Test set   : {len(test_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in test_loader][0]}")

            time_now = time.time()

            train_steps = len(train_loader)

            if checkpoint_base:
                best_model_path = os.path.join(checkpoint_base, f'{setting}', f'fold{fold}.pth')
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            else:
                path = os.path.join(self.args.checkpoints, f'{setting}', f'fold{fold}')
                os.makedirs(path, exist_ok=True)
                best_model_path = os.path.join(path, 'checkpoint.pth')
            
            # Early stopping to prevent overfitting
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
            
            # Select optimizer and learning rate scheduler
            model_optim = self._select_optimizer()
            if self.args.lradj == 'type3':
                scheduler = ReduceLROnPlateau(
                    model_optim,
                    mode='min',
                    factor=0.3,
                    patience=5,
                    threshold=1e-3,
                    min_lr=1e-8, #
                    cooldown=0,
                    threshold_mode='abs',    
                    )
            else:
                scheduler = None

            criterion = self._select_criterion()

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            train_losses_this_fold = []
            val_losses_this_fold = []
            
            # Training loop for each fold
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -2 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                            pred = outputs[:, -self.args.pred_len:, :]
                            true = batch_y[:, -self.args.pred_len:, :]


                            loss_lower = criterion(pred[:, :, 0], true[:, :, 0])
                            loss_upper = criterion(pred[:, :, 1], true[:, :, 1])
                            loss = loss_lower + loss_upper
                            train_loss.append(loss.item())
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -2 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                        pred = outputs[:, -self.args.pred_len:, :]
                        true = batch_y[:, -self.args.pred_len:, :]


                        loss_lower = criterion(pred[:, :, 0], true[:, :, 0])
                        loss_upper = criterion(pred[:, :, 1], true[:, :, 1])
                        loss = loss_lower + loss_upper
                        train_loss.append(loss.item())

                    # Log training progress
                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        # Backward and optimize
                        loss.backward()
                        model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
                
                train_losses_this_fold.append(train_loss)
                val_losses_this_fold.append(vali_loss)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Vali Loss: {3:.7f} | Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

                # Early stopping based on validation loss
                early_stopping(vali_loss, self.model, best_model_path)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, vali_loss)
            
            self.model.load_state_dict(torch.load(best_model_path))

            if plot:
                self.model.load_state_dict(torch.load(best_model_path))
                folder_path = os.path.join('./test_results', setting, f'fold{fold}')
                os.makedirs(folder_path, exist_ok=True)
                plot_loss(train_losses_this_fold, val_losses_this_fold, name=os.path.join(folder_path, 'loss.pdf'))

            fold_time = time.time() - fold_start
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MiB
            logging.info(f"[TIME] Fold {fold+1} training time: {fold_time:.2f}s, Peak GPU memory used: {peak_mem:.2f} MB")

            val_losses.append(val_losses_this_fold[-1])

        return val_losses


    def train_standard(self, setting, plot=True,  checkpoint_base="./checkpoints/tmp"):
        """
        Train the model using standard training procedure.
        Args:
            setting: A string indicating the training setting (e.g., 'standard').
            plot: A boolean indicating whether to plot the training and validation losses.
            checkpoint_base: Base directory for saving checkpoints.
        Returns:
            val_loss: The final validation loss after training.
        """
        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"[MODEL] Trainable parameters: {model_params:,}")
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # print(f"[INFO] Dataset Shapes:")
        # print(f"  ➤ Train set  : {len(train_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in train_loader][0]}")
        # print(f"  ➤ Val set    : {len(vali_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in vali_loader][0]}")
        # print(f"  ➤ Test set   : {len(test_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in test_loader][0]}")

        time_now = time.time()

        train_steps = len(train_loader)

        if checkpoint_base:
            best_model_path = os.path.join(checkpoint_base, f'{setting}', f'checkpoint.pth')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        else:
            path = os.path.join(self.args.checkpoints, f'{setting}', f'checkpoint')
            os.makedirs(path, exist_ok=True)
            best_model_path = os.path.join(path, 'checkpoint.pth')
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Select optimizer and learning rate scheduler
        model_optim = self._select_optimizer()
        if self.args.lradj == 'type3':
            scheduler = ReduceLROnPlateau(
                model_optim,
                mode='min',
                factor=0.3,
                patience=5,
                threshold=1e-3,
                min_lr=1e-8, #
                cooldown=0,
                threshold_mode='abs',    
                )
        else:
            scheduler = None


        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_losses = []
        val_losses, val_mses = [], []

        # Training loop
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -2 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                        pred = outputs[:, -self.args.pred_len:, :]
                        true = batch_y[:, -self.args.pred_len:, :]


                        loss_lower = criterion(pred[:, :, 0], true[:, :, 0])
                        loss_upper = criterion(pred[:, :, 1], true[:, :, 1])
                        loss = loss_lower + loss_upper
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -2 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                    pred = outputs[:, -self.args.pred_len:, :]
                    true = batch_y[:, -self.args.pred_len:, :]


                    loss_lower = criterion(pred[:, :, 0], true[:, :, 0])
                    loss_upper = criterion(pred[:, :, 1], true[:, :, 1])
                    loss = loss_lower + loss_upper
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # Backward pass and optimization
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(vali_loss)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, vali_loss)

        self.model.load_state_dict(torch.load(best_model_path))
        
        if plot:
            self.model.load_state_dict(torch.load(best_model_path))
            folder_path = './test_results/' + setting + '/'
            os.makedirs(folder_path, exist_ok=True)
            plot_loss(train_losses, val_losses, name=os.path.join(folder_path, 'loss.pdf'))
        
        time_train = time.time() - time_now
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MiB
        logging.info(f"[TIME] Training time: {time_train:.2f}s, Peak GPU memory used: {peak_mem:.2f} MB")

        # return self.model
        return vali_loss



    def test(self, setting, test=0, plot=True):
        """
        Test the model on the test dataset.
        Args:
            setting: A string indicating the testing setting (e.g., 'kfold', 'standard').
            test: An integer indicating whether to load a pre-trained model (1) or not (0).
            plot: A boolean indicating whether to plot the test results.
        Returns:
            None
        """
        start_time = time.time()

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]    # (B, pred_len, C)
                batch_y = batch_y[:, -self.args.pred_len:, :]    # (B, pred_len, C)

                # ONLY KEEP target columns (lower, upper)
                outputs = outputs[..., -2:]
                batch_y = batch_y[..., -2:]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    # inverse scaling
                    bs, pred_len, _ = outputs.shape
                    outputs_reshape = np.zeros((bs * pred_len, test_data.scaler.scale_.shape[0]))
                    outputs_reshape[:, -2:] = outputs.reshape(bs * pred_len, -1)
                    outputs = test_data.inverse_transform(outputs_reshape).reshape(bs, pred_len, -1)[:, :, -2:]

                    batch_y_reshape = np.zeros((bs * pred_len, test_data.scaler.scale_.shape[0]))
                    batch_y_reshape[:, -2:] = batch_y.reshape(bs * pred_len, -1)
                    batch_y = test_data.inverse_transform(batch_y_reshape).reshape(bs, pred_len, -1)[:, :, -2:]

                preds.append(outputs)
                trues.append(batch_y)

                if i % 12 == 0 and plot:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    gt = np.concatenate((input[0, :, -2:], batch_y[0, :, :]), axis=0)
                    pd = np.concatenate((input[0, :, -2:], outputs[0, :, :]), axis=0)

                    folder_path = './test_results/' + setting + '/'
                    os.makedirs(folder_path, exist_ok=True)
                    visual_interval(gt, pd, os.path.join(folder_path, f'{i}.pdf'), input_len=self.args.seq_len)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'
        
        # Save results
        if plot:
            folder_path = './results/' + setting + '/'
            os.makedirs(folder_path, exist_ok=True)

            mae, mse, rmse, mape, mspe, nse = metric(preds, trues)
            print(f'TESTING: mse:{mse:.4f} | mae:{mae:.4f} | dtw:{dtw} | nse:{nse:.4f}')
            
            np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, nse]))
            np.save(os.path.join(folder_path, 'pred.npy'), preds)
            np.save(os.path.join(folder_path, 'true.npy'), trues)

        elapsed_time = time.time() - start_time
        print(f"[TIME] Testing time: {elapsed_time:.2f} seconds")
        return