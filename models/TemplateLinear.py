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

        # 过去影响未来
        self.l = nn.Linear(self.P, self.F)

    def forward(self, x: torch.Tensor):
        # [B,P,C]
        x = x.permute(0, 2, 1)  # [B,C,P]
        x = self.p(x)  # [B,C,P]@[P,F]=[B,C,F]
        x = x.permute(0, 2, 1)  # [B,F,C]
        return x
