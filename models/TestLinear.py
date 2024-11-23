import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.P = configs.seq_len
        self.F = configs.pred_len
        self.C = configs.enc_in

        self.linear = nn.Linear(self.P, self.F)

    def forward(self, x: torch.Tensor):
        # [B,P,C]
        x = x.permute(0, 2, 1)  # [B,C,P]
        E = torch.mean(x, dim=-1).unsqueeze(-1).repeat([1, 1, x.shape[-1]])  # [B,C,P]
        S = torch.std(x, dim=-1).unsqueeze(-1).repeat([1, 1, x.shape[-1]])  # [B,C,P]
        x = (x - E) / S
        x = self.linear(x)  # [B,C,F]
        x = x * S + E
        x = x.permute(0, 2, 1)  # [B,F,C]
        return x
