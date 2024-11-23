import torch
import torch.nn as nn


class AFilter_FFT_Decomper(nn.Module):

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, X) -> tuple[torch.Tensor, torch.Tensor]:
        X_fft: torch.Tensor = torch.fft.fft(X, dim=-1)
        X_fft_Afiltered: torch.Tensor = X_fft.where(abs(X_fft) > self.threshold, 0)
        X_fft_Afiltered_ifft: torch.Tensor = torch.fft.ifft(X_fft_Afiltered).real
        return X_fft_Afiltered_ifft, X - X_fft_Afiltered_ifft


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.decomp = AFilter_FFT_Decomper(5)
        self.l_season = nn.Linear(configs.seq_len, configs.pred_len)
        self.l_trend = nn.Linear(configs.seq_len, configs.pred_len)

    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        X_batch = X_batch.permute(0, 2, 1)  # [B,P,C]->[B,C,P]
        X_trend, X_season = self.decomp(X_batch)
        Y_season: torch.Tensor = self.l_season(X_season)  # [B,C,P]@[P,F]=[B,C,F]
        Y_trend: torch.Tensor = self.l_trend(X_trend)  # [B,C,P]@[P,F]=[B,C,F]
        Y_pred: torch.Tensor = Y_season + Y_trend
        Y_pred = Y_pred.permute(0, 2, 1)  # [B,C,F]->[B,F,C]
        return Y_pred
