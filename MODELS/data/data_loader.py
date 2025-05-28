# -*- coding: utf-8 -*-
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from utils.timefeatures import time_features
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='iFile.csv',
                 target='Low,High', scale=True, timeenc=0, freq='m',
                 embed='timeF', kfold=0):
        # Default sequence length settings
        self.embed = embed 
        self.seq_len, self.label_len, self.pred_len = size
        self.kfold = kfold

        self.flag = flag
        if self.kfold > 0:
            assert self.flag in ['train', 'test']
            type_map = {'train': 0, 'test': 1}
        else:
            assert self.flag in ['train', 'val', 'test']
            type_map = {'train': 0, 'val': 1, 'test': 2}

        self.set_type = type_map[self.flag]

        self.features = features
        if isinstance(target, str):
            self.target = [t.strip() for t in target.split(',')]
            print(self.target)
        else:
            self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        file_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(file_path)

        cols = list(df_raw.columns)
        cols.remove('date')
        for t in self.target:
            if t in cols:
                cols.remove(t)
        df_raw = df_raw[['date'] + cols + self.target]
        self.df_target = df_raw[['date'] + self.target].copy()

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # kfold hoặc standard split
        if self.kfold == 0:
            n = len(df_raw)
            self.num_train = int(n * 0.8)
            self.num_test = int(n * 0.1)
            self.num_vali = n - self.num_train - self.num_test

            border1s = [0, self.num_train - self.seq_len, n - self.num_test - self.seq_len]
            border2s = [self.num_train, self.num_train + self.num_vali, n]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

        else:  # kfold > 0
            self.num_train = int(len(df_raw) * 0.8)
            self.num_test = len(df_raw) - self.num_train
            if self.flag == 'train':
                border1 = 0
                border2 = self.num_train
            else:  # 'test'
                border1 = len(df_raw) - self.num_test
                border2 = len(df_raw)

            if self.scale:
                train_data = df_data[:self.num_train]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

        # Normal cases: add data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.dt.year
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            self.data_stamp = df_stamp[['year', 'month', 'day']].values
        elif self.timeenc == 1:
            self.data_stamp = time_features(df_stamp['date'].values, freq=self.freq).transpose(1, 0)
        elif self.timeenc == 2:
            if 'month_sin' in df_raw.columns and 'month_cos' in df_raw.columns:
                self.data_stamp = df_raw[['month_sin', 'month_cos']].values[border1:border2]
            else:
                self.data_stamp = np.zeros((self.data_x.shape[0], 2))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='MS', data_path='iFile.csv',
                 target='Low,High', scale=True, inverse=False, timeenc=0, freq='m', cols=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag in ['pred']

        self.features = features
        if isinstance(target, str):
            self.target = [t.strip() for t in target.split(',')]
            print(self.target)
        else:
            self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])

        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.dt.year
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(df_stamp['date'].values, freq=self.freq).transpose(1, 0)
        elif self.timeenc == 2:
            data_stamp = df_stamp[['month_sin', 'month_cos']].values

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.df_target = df_raw[['date'] + self.target].copy()


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)