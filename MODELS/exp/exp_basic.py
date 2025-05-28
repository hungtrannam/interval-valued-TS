# -*- coding: utf-8 -*-
"""
@file   : exp_main.py
@author : HUNG TRAN-NAM
@contact:
"""

import os
import torch
from models import DLinear, FEDformer, LSTM,\
            PatchTST, TimeNet, Transformer, iTransformer,\
            GRU, Nonstationary_Transformer                


# This is the base class for experiments in time series forecasting models.
class Exp_Basic(object):
    """
    Base class for time series forecasting experiments.
    This class provides a framework for building, training, validating, and testing time series models.
    It initializes the model based on the provided arguments and sets the device for computation.
    Attributes:
        args: Argument parser containing model parameters.
        model_dict: Dictionary mapping model names to their respective classes.
        device: The device (CPU or GPU) on which the model will run.
        model: The initialized model instance.
    """
    
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimeNet,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'PatchTST': PatchTST,
            'LSTM': LSTM,
            'TimeNet': TimeNet,
            'iTransformer': iTransformer,
            'NLinear': Nonstationary_Transformer,
            'GRU': GRU

        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass