import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from utils.activations import get_activation_fn

class Model(nn.Module):
    """
    LSTM-based Seq2Seq Model for Time Series Forecasting with MS mode (Multivariate ➝ Univariate)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.d_layers = configs.d_layers
        self.bidirectional = configs.bidirectional
        self.activation = get_activation_fn(configs.activation)

        # Embedding for encoder and decoder
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                           configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model,
                                           configs.embed, configs.freq, configs.dropout)

        # Encoder: many-to-many LSTM
        self.encoder = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            bidirectional=self.bidirectional,
            dropout=configs.dropout if configs.e_layers > 1 else 0,
            batch_first=True
        )

        # Decoder: many-to-many LSTM
        self.decoder = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=configs.d_layers,
            bidirectional=False,
            dropout=configs.dropout if configs.d_layers > 1 else 0,
            batch_first=True
        )

        # Final projection to univariate output (target)
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward pass for MS mode (multivariate input ➝ univariate output)

        Args:
            x_enc: [B, seq_len, enc_in]      # multivariate input
            x_dec: [B, label_len+pred_len, dec_in]
        Returns:
            output: [B, pred_len, c_out=1]   # univariate target forecast
        """
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, seq_len, d_model]
        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [B, label+pred, d_model]

        # Encoder
        _, (h_n, c_n) = self.encoder(enc_out)

        # Align hidden size if encoder and decoder have different layer counts
        if h_n.size(0) != self.d_layers:
            batch_size = h_n.size(1)
            device = h_n.device
            h_n = torch.zeros(self.d_layers, batch_size, self.d_model, device=device)
            c_n = torch.zeros(self.d_layers, batch_size, self.d_model, device=device)

        # Decoder
        dec_out, _ = self.decoder(dec_out, (h_n, c_n))

        # Final projection and activation
        output = self.projection(dec_out[:, -self.pred_len:, :])  # [B, pred_len, 1]
        output = self.activation(output)
        return output