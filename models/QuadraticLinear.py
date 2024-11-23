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
        self.R = self.P

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.C = configs.enc_in

        if configs.individual:
            raise NotImplementedError("Only one")

        # L1 is [P,R]
        self.Linear1 = nn.Linear(in_features=self.P, out_features=self.R, bias=True)
        # L1 is [P,R]
        self.Linear2 = nn.Linear(in_features=self.P, out_features=self.R, bias=True)
        # L3 is [R,F]
        self.Linear3 = nn.Linear(in_features=self.R, out_features=self.F, bias=True)

    def forward(self, x: torch.Tensor):
        # x: [B,P,C]

        x = x.permute(0, 2, 1)
        # x is [B,C,P]

        x1 = self.Linear1(x)
        # x1 is [B,C,P][P,R] is [B,C,R]

        x2 = self.Linear1(x)
        # x2 is [B,C,P][P,R] is [B,C,R]

        x3 = x1 * x2
        # x3 is [B,C,R] dots [B,C,R] is [B,C,R]

        x4 = self.Linear3(x3)
        # x4 is [B,C,R][R,F] is [B,C,F]

        y = x4.permute(0, 2, 1)
        # y is [B,F,C]

        return y
