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

        self.hidden_width = 4 * max(self.P, self.F)

        if configs.individual:
            raise NotImplementedError("Only non-individual is implemented")

        # P to F
        self.Layers = nn.Sequential(
            nn.Linear(self.P, self.hidden_width),
            nn.PReLU(),
            nn.Linear(self.hidden_width, self.F),
        )

    def forward(self, x: torch.Tensor):
        # x is [B,P,C]
        x = x.permute(0, 2, 1)
        # x is [B,C,P]

        y = self.Layers(x)

        # y is [B,C,F]
        y = y.permute(0, 2, 1)
        # y is [B,F,C]

        return y
