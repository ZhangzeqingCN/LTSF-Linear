import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            # Linear : [P,F]*C
            self.Linear0 = nn.ModuleList()
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear0.append(nn.Linear(self.seq_len, self.seq_len))
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            # Linear : [P,F]
            self.Linear0 = nn.Linear(self.seq_len, self.seq_len)
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor):

        # x: [B,P,C]

        if self.individual:
            # output: [B,F,C]
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                # x[:,:,i]==[B,P]
                # [B,P][P,P]==[B,P]
                # [B,P][P,F]==[B,F]
                output[:, :, i] = self.Linear0[i](x[:, :, i])
                output[:, :, i] = self.Linear[i](output[:, :, i])
            x: torch.Tensor = output

        else:
            # [B,P,C].permute(0,2,1)==[B,C,P]
            # [B,C,P][P,P]==[B,C,P]
            # [B,C,P][P,F]==[B,C,F]
            # [B,C,F].permute(0,2,1)==[B,F,C]
            x = x.permute(0, 2, 1)
            x: torch.Tensor = self.Linear0(x)
            x: torch.Tensor = self.Linear(x)
            x = x.permute(0, 2, 1)

            # x: [B,F,C]

        return x