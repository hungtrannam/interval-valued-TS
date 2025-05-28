# -- coding: utf-8 --
"""
@file   : run_optun.py
@author : HUNG TRAN-NAM
@contact:

"""
import argparse
import optuna, optunahub
import yaml
import random
import torch
import numpy as np
from datetime import datetime
from exp.exp_main import Exp_Main
from utils.tools import set_seed
import os
from utils.params import suggest_model_specific
from utils.vis import *
import shutil
import time


# ========================
# Objective function for Optuna
# ========================

def objective(trial):
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--task_name', type=str, default='short_term_forecast')
    parser.add_argument('--model_id', type=str, default='optuna_tune')
    parser.add_argument('--model', type=str, default=os.environ.get("MODEL_NAME", "Autoformer"))
    parser.add_argument('--des', type=str, default='optuna')

    # Data
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='/home/hung-tran-nam/SWAT_AIv2v/dataset/DataSet_raw')
    parser.add_argument('--data_path', type=str, default='pre_ChiangSaen_spi_6.csv')
    parser.add_argument('--target', type=str, default='spi_6')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--freq', type=str, default='m')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    valid_seq_label_pairs = [(s, l) for s in [24, 36, 48, 60, 72, 84]
                                    for l in [6, 12, 24, 36, 48]
                                    if s >= l and (s + l + 12) <= 96]
    seq_len, label_len = trial.suggest_categorical("seq_label_len", valid_seq_label_pairs)
    parser.add_argument('--seq_len', type=int, default=seq_len)
    parser.add_argument('--label_len', type=int, default=label_len)
    parser.add_argument('--moving_avg', default=3)

    parser.add_argument('--pred_len', type=int, default=12)


    # Setting
    parser.add_argument(
        '--kfold', type=int, default=0,
        help='Number of folds for k-fold cross validation (0 = no k-fold, >=2 = use k-fold)'
    )
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')


    # Learning
    parser.add_argument('--dropout', type=float, default=trial.suggest_float('dropout', 0.0, 0.4))
    parser.add_argument('--lradj', type=str, default=trial.suggest_categorical('lradj', ['type1', 'type2', 'type3']), help='adjust learning rate')
    parser.add_argument('--learning_rate', type=float, default=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True))
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--activation', type=str, default=trial.suggest_categorical('activation', ['relu', 'silu', 'gelu']), help='activation function')

    # optimization
    parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--batch_size', type=int, default=trial.suggest_categorical('batch_size', [4, 8]))

    # For bi-LSTM
    parser.add_argument('--bidirectional', default=True, action="store_true", help="Bidirectional LSTM")  #NEED

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=True,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment") #NEED
    parser.add_argument('--seed', type=int, default=42, help="Randomization seed")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")  #NEED
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")  #NEED    
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")  #NEED
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")  #NEED
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")    



    # Fixed value
    args, _ = parser.parse_known_args()

    # Fixed param for model requirement
    if args.embed == 'timeF':
        args.enc_in = 2
        args.dec_in = 2
    elif args.embed == 'monthSine':
        args.enc_in = 2+2
        args.dec_in = 2+2
    args.c_out = 2

    args.use_amp = False
    args.inverse = True
    args.use_gpu = False
    args.gpu_type = 'cuda'
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0,1,2,3'

    # Random Seed
    set_seed(args.seed)
    args.__dict__.update(suggest_model_specific(trial, args.model))

    # Detect Device
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print('Using GPU')
    else:
        args.device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
        print('Using CPU or MPS')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '').split(',')
        args.device_ids = [int(id_) for id_ in args.devices]
        args.gpu = args.device_ids[0]
    
    # Create checkpoints directory
    exp = Exp_Main(args)
    setting = f"{args.model_id}_{args.data_path}_{args.model}_trial{trial.number}"

    # Create a unique directory for this trial
    if args.kfold > 0:
        checkpoint_tmp = os.path.join(args.checkpoints, "tmp")
        val_losses = exp.train_kfold(
            setting,
            plot=False,
            checkpoint_base=checkpoint_tmp
        )
        trial.set_user_attr("val_losses", val_losses)

        median_val_loss = np.median(val_losses)

        if not hasattr(objective, "best_val_loss") or median_val_loss < objective.best_val_loss:
            objective.best_val_loss = median_val_loss
            objective.best_trial_number = trial.number

        return median_val_loss

    else:
        checkpoint_tmp = os.path.join(args.checkpoints, "tmp")
        val_loss = exp.train_standard(setting, plot=False, checkpoint_base=checkpoint_tmp)
        return val_loss


# ========================
# Main function to run the optimization
# ========================
if __name__ == '__main__':
    # Set up the Optuna study
    sampler = optuna.samplers.TPESampler(
        seed=42,
        multivariate=True,
        group=True,
        n_startup_trials=5
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, 
        n_warmup_steps=0)


    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        study_name="optuna_best_search",
        load_if_exists=True
    )

    # NUM_TRIALS
    study.optimize(objective, n_trials=10, catch=(RuntimeError, ValueError, TypeError))
    print(">>>> FINISHED OPTIMIZATION <<<<")

    # Save the study
    best_trial = study.best_trial
    print("Best Trial Result:")
    print(study.best_trial)

    # ------------------ BUILD FULL ARGUMENTS ------------------ #
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='optuna_tune')
    parser.add_argument('--model', type=str, default=os.environ.get("MODEL_NAME", "Autoformer"))
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='/home/hung-tran-nam/SWAT_AIv2v/dataset/DataSet_raw')
    parser.add_argument('--data_path', type=str, default='pre_ChiangSaen_spi_6.csv')
    parser.add_argument('--target', type=str, default='spi_6')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--freq', type=str, default='m')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--des', type=str, default='optuna')
    parser.add_argument('--train_epochs', type=int, default=100,  help='train epochs')
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--seg_len', type=int, default=24, help='segment length for SegRNN')
    parser.add_argument('--bidirectional', default=True, action="store_true", help="Bidirectional LSTM")  #NEED
    parser.add_argument(
        '--kfold', type=int, default=0,
        help='Number of folds for k-fold cross validation (0 = no k-fold, >=2 = use k-fold)'
    )
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--moving_avg', default=3)
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--use_dtw', default=True)
    # TimesBlock / FEDformer / De-stationary models
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('embed', type=str)
    # Augment
    parser.add_argument('--seed', type=int, default=42)

    # Fixed parameters
    args, _ = parser.parse_known_args()
    
    # Update args with best trial parameters
    tmp_dir = os.path.join(args.checkpoints, "tmp")
    start_copy = time.time()
    if args.kfold > 0:
        for fold in range(args.kfold):
            tmp_file = os.path.join(
                tmp_dir,
                f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}', f'fold{fold}.pth'
            )
            final_path = os.path.join(
                args.checkpoints,
                f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}', f'fold{fold}', 'checkpoint.pth'
            )
            if os.path.exists(tmp_file):
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                print(f"[INFO] Copying fold {fold} checkpoint...")
                fold_start = time.time()
                shutil.copy(tmp_file, final_path)
                print(f"[INFO] Fold {fold} copied in {time.time() - fold_start:.2f}s")
    else:
        tmp_file = os.path.join(
            tmp_dir,
            f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}', f'checkpoint.pth'
        )
        final_path = os.path.join(
            args.checkpoints,
            f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}', 'checkpoint.pth'
        )
        if os.path.exists(tmp_file):
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            print("[INFO] Copying checkpoint...")
            shutil.copy(tmp_file, final_path)

    # copy and paste the best trial parameters
    start_rm = time.time()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[INFO] Removed tmp_dir in {time.time() - start_rm:.2f}s")


    # Update args with best trial parameters
    best_params = best_trial.params
    args.__dict__.update(best_trial.params)
    args.__dict__.update(suggest_model_specific(best_trial, args.model))
    args.seq_len, args.label_len = best_params['seq_label_len']

    if args.embed == 'timeF':
        args.enc_in = 2
        args.dec_in = 2
    elif args.embed == 'monthSine':
        args.enc_in = 2+2
        args.dec_in = 2+2
    args.c_out = 2
    args.patience = 3
    args.inverse = True
    args.use_gpu = False

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

    set_seed(args.seed)



    # =========================
    # Run the experiment with the best parameters
    # =========================
    print("Run experiment with best params and plot result")
    Exp = Exp_Main
    exp = Exp(args)


    # Check if the checkpoint directory exists, if not, create it
    if args.kfold > 0:
        setting = f"{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}"
        exp.train_kfold(setting, plot=True)
        for fold in range(args.kfold):
            # Test each fold
            print(f"{'>'*50} Testing/Predicting {setting} {'<'*50}")
            exp.test(os.path.join(setting, f'fold{fold}'), test=1, plot=True)

    elif args.kfold == 0:
        setting = f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}'
        exp.train_standard(setting, plot=True)
        print(f"{'>'*50} Testing/Predicting {setting} {'<'*50}")
        # Test the model
        exp.test(setting, test=1, plot=True)


    # Clean GPU Cache
    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()

    print(f"{'>'*50} Finish Experiments {setting} {'<'*50}")


    # ========================
    # Plotting results
    # ========================

    # plot_input(dataset, save_path=f'./test_results/{setting}/INplot.pdf')
    dataset, train_loader = exp._get_data(flag='train')
    # plot_metrics(study, './test_results/' + setting)
    # plot_hyperparameter_importance(study, './test_results/' + setting)
    plot_optimization_history(study, './test_results/' + setting)

    if args.kfold > 0:
        for fold in range(args.kfold):
            setting = f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}/fold{fold}'

            y_true = np.load(f'./results/{setting}/true.npy')
            y_pred = np.load(f'./results/{setting}/pred.npy')

            plot_scatter_truth_vs_pred(y_true, y_pred, save_path=f'./test_results/{setting}/PredScatter.pdf')
            # plot_residual_acf(y_true, y_pred, save_path=f'./test_results/{setting}/ACF.pdf')

            args_save_path = f'./test_results/{setting}/best.yaml'
            args_dict = vars(args)
            with open(args_save_path, 'w') as f:
                yaml.dump({
                    'args': vars(args)
                }, f, sort_keys=False)
    else:
        setting = f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}'

        y_true = np.load(f'./results/{setting}/true.npy')
        y_pred = np.load(f'./results/{setting}/pred.npy')

        plot_scatter_truth_vs_pred(y_true, y_pred, save_path=f'./test_results/{setting}/PredScatter.pdf')
        # plot_residual_acf(y_true, y_pred, save_path=f'./test_results/{setting}/ACF.pdf')

        args_save_path = f'./test_results/{setting}/best.yaml'
        args_dict = vars(args)
        with open(args_save_path, 'w') as f:
            yaml.dump({
                'args': vars(args)
            }, f, sort_keys=False)


