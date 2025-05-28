import argparse
import os
import torch
from exp.exp_stats import Exp_Main
import random
import numpy as np
from datetime import datetime


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='Not implemented')
parser.add_argument('--embed', type=str, default='timeF',
                    help='Not implemented')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='GBRT',
                    help='model name, options: [GBRT, Arima, SArima]')

# data loader
parser.add_argument('--data', type=str, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='spi_6', help='target feature in S or MS task')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--sample', type=float, default=1, help='Sampling percentage, the inference time of ARIMA and SARIMA is too long, you might sample 0.01')
parser.add_argument('--freq', type=str, default='m',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
                    
# forecasting task
parser.add_argument('--seq_len', type=int, default=196, help='input sequence length')
parser.add_argument('--label_len', type=int, default=36, help='start token length') # Just for reusing data loader
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='Not implemented')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
parser.add_argument('--des', type=str, default='test', help='exp description')

args = parser.parse_args()
args.use_gpu = False
print('Args in experiment:')
print(args)
Exp = Exp_Main

timestamp = datetime.now().strftime('%Y%m%d_%H%M')
setting = f'{args.model_id}_{args.data_path}_{args.model}_{timestamp}'

exp = Exp(args)  # set experiments
print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
exp.test(setting)
       