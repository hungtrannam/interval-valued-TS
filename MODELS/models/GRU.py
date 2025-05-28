

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.Model = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            dropout=configs.dropout,
            batch_first=True
        )
        self.fc = nn.Linear(configs.d_model, configs.c_out)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.fc_class = nn.Linear(configs.seq_len * configs.d_model, configs.num_class)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        x_enc: [B, L, D] Input sequence
        """
        # Forecasting task
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            out, _ = self.Model(x_enc)  # [B, L, H]
            out = self.fc(out)        # [B, L, C]
            return out[:, -self.pred_len:, :]  # [B, pred_len, C]

        # Imputation task
        if self.task_name == 'imputation':
            out, _ = self.Model(x_enc)
            out = self.fc(out)
            return out  # [B, L, C]

        # Anomaly detection task
        if self.task_name == 'anomaly_detection':
            out, _ = self.Model(x_enc)
            out = self.fc(out)
            return out  # [B, L, C]

        # Classification task
        if self.task_name == 'classification':
            out, _ = self.Model(x_enc)
            out = self.act(out)
            out = self.dropout(out)
            out = out.reshape(out.shape[0], -1)  # flatten
            out = self.fc_class(out)  # [B, num_class]
            return out

        return None
