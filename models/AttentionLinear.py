import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.d_Q = out_features  # Attention输出特征和d_Q相同
        self.d_K = self.d_Q  # Attention需要QK^T
        self.d_V = out_features

        self.WK = nn.Linear(self.in_features, self.d_K)
        self.WQ = nn.Linear(self.in_features, self.d_Q)
        self.WV = nn.Linear(self.in_features, self.d_V)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_Q,
            num_heads=1,
            kdim=self.d_K,
            vdim=self.d_V,
        )

    def forward(self, x: torch.Tensor):
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        y, _ = self.attn(Q, K, V)
        # [B,C, d_Q]
        return y


class Model(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        print(F"{configs=}")

        # Layer 1

        self.P = configs.seq_len
        self.F = configs.pred_len
        self.C = configs.enc_in

        self.linear = nn.Linear(self.P, self.F)

        self.attn = nn.Linear(self.P, self.F)

        self.mix = nn.Linear(2 * self.C, self.C)

        # Layer 2

    def forward(self, x: torch.Tensor):
        # from [B,P,C] to [B,C,P]
        x = x.permute(0, 2, 1)

        # Process

        # z = self.mix(torch.cat([self.linear(x), self.attn(x)], dim=1).permute(0, 2, 1))
        z = self.linear(x) * self.attn(x)
        z = z.permute(0, 2, 1)
        # z = self.linear(x).permute(0, 2, 1)

        # Output
        return z
