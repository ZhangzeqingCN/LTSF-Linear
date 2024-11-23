from datetime import datetime
from itertools import product
from typing import Type, Literal
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class Args:
    B = 16
    epochs = 10
    P = 96
    F = 96
    C = 12
    lstm_hidden = 233
    lstm_layers = 6
    kernel_size = 25


class MyDataset(Dataset):

    def __init__(self, D, args: Args) -> None:
        super().__init__()
        self.D = D  # [T,C]
        self.F = args.F
        self.P = args.P

    def __getitem__(self, i: int) -> torch.Tensor:
        X_s_i = self.D[i : i + self.P]
        Y_s_i = self.D[i + self.P : i + self.P + self.F]
        return X_s_i, Y_s_i

    def __len__(self):
        return len(self.D) - self.P - self.F + 1


class Linear(nn.Module):

    def __init__(self, args: Args) -> None:
        super().__init__()
        self.P = args.P
        self.F = args.F
        self.l = nn.Linear(args.P, args.F)

    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        X_batch = X_batch.permute(0, 2, 1)  # [B,P,C]->[B,C,P]
        Y_pred: torch.Tensor = self.l(X_batch)  # [B,C,P]@[P,F]=[B,C,F]
        Y_pred = Y_pred.permute(0, 2, 1)  # [B,C,F]->[B,F,C]
        return Y_pred


class NLinear(nn.Module):

    def __init__(self, args: Args) -> None:
        super().__init__()
        self.P = args.P
        self.F = args.F
        self.l = nn.Linear(args.P, args.F)

    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        R = X_batch[:, -1:, :].detach()  # Remove auto grad
        X_batch = X_batch - R
        X_batch = X_batch.permute(0, 2, 1)  # [B,P,C]->[B,C,P]
        Y_pred: torch.Tensor = self.l(X_batch)  # [B,C,P]@[P,F]=[B,C,F]
        Y_pred = Y_pred.permute(0, 2, 1)  # [B,C,F]->[B,F,C]
        Y_pred = Y_pred + R
        return Y_pred


class DLinear(nn.Module):

    def __init__(self, args: Args) -> None:
        super().__init__()
        self.P = args.P
        self.F = args.F
        self.l_season = nn.Linear(args.P, args.F)
        self.l_trend = nn.Linear(args.P, args.F)
        padding1 = (args.kernel_size - 1) // 2
        padding2 = (args.kernel_size - 1) - padding1
        self.padding = nn.ReplicationPad1d(padding=(padding1, padding2))
        self.avgpool = nn.AvgPool1d(kernel_size=args.kernel_size, stride=1)

    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        X_batch = X_batch.permute(0, 2, 1)  # [B,P,C]->[B,C,P]
        X_trend = self.avgpool(self.padding(X_batch))
        X_season = X_batch - X_trend
        Y_season: torch.Tensor = self.l_season(X_season)  # [B,C,P]@[P,F]=[B,C,F]
        Y_trend: torch.Tensor = self.l_trend(X_trend)  # [B,C,P]@[P,F]=[B,C,F]
        Y_pred: torch.Tensor = Y_season + Y_trend
        Y_pred = Y_pred.permute(0, 2, 1)  # [B,C,F]->[B,F,C]
        return Y_pred


class Lstm(nn.Module):

    def __init__(self, args: Args) -> None:
        super().__init__()
        self.P = args.P
        self.F = args.F
        self.lstm = nn.LSTM(args.C, args.lstm_hidden, args.lstm_layers)
        self.l = nn.Linear(args.lstm_hidden, args.C)

    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        # [B,P,C]
        X_iter = X_batch
        for _ in range(self.F):
            X_lstm1 = self.lstm(X_iter[:, -self.P :, :])[0][:, -1, :]
            X_lstm2 = self.l(X_lstm1).unsqueeze(-2)
            X_iter = torch.cat([X_iter, X_lstm2], dim=1)
        return X_iter[:, -self.F :, :]


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.min_loss = torch.inf
        self.early_stop = False

    def __call__(self, vali_loss):
        if vali_loss < self.min_loss:
            self.counter = 0
            self.min_loss = vali_loss
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True


def get_data(
    csv_path="ETTm1.csv",
    label: Literal["M", "S"] = "M",
) -> torch.Tensor:
    data_frame = pd.read_csv(csv_path)
    assert label in ["M", "S"]
    if label == "M":
        data_frame = data_frame.drop(columns=["date", "OT"])  # 取OT和date除外的
        Data = torch.tensor(data_frame.to_numpy())
    else:
        data_frame = data_frame["OT"]
        Data = torch.tensor(data_frame.to_numpy()).unsqueeze(1)
    return Data  # [N,C]


def split_data(
    Data: torch.Tensor,
    P: int,
    R_train: float = 0.7,
    R_test: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert R_train >= 0 and R_test >= 0 and 1 - R_train - R_test >= 0
    N = len(Data)
    N_train = int(R_train * N)
    N_test = int(R_test * N)
    N_vali = N - N_train - N_test
    Data_train = Data[0:N_train]
    Data_vali = Data[N_train - P : N_train + N_vali]
    Data_test = Data[N_test - P : N]
    return Data_train, Data_vali, Data_test


def get_device(use_cuda=True) -> torch.device:
    print(f"{torch.cuda.is_available()=}")
    print(f"{use_cuda=}")
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"{device=}")


# 暂时未加入Vali和EarlyStopping
def do_train(
    Data_train: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    args: Args,
):
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters())
    model.train(True)
    for _ in range(args.epochs):
        loss: torch.Tensor = None
        for X_batch, Y_batch in DataLoader(
            MyDataset(D=Data_train, args=args),
            batch_size=args.B,
            shuffle=True,
            drop_last=True,
        ):
            X_batch: torch.Tensor = X_batch.float().to(device=device)
            Y_batch: torch.Tensor = Y_batch.float().to(device=device)
            opt.zero_grad()
            Y_pred: torch.Tensor = model(X_batch)
            loss: torch.Tensor = crit(Y_batch, Y_pred)
            loss.backward()
            opt.step()


def mse(Y_true: torch.Tensor, Y_pred: torch.Tensor) -> float:
    return torch.mean((Y_pred - Y_true) ** 2).item()


def mae(Y_true: torch.Tensor, Y_pred: torch.Tensor) -> float:
    return torch.mean(torch.abs(Y_pred - Y_true)).item()


def do_test(
    Data_test: torch.Tensor, model: nn.Module, device: torch.device, args: Args
) -> tuple[float, float]:
    model.eval()
    Y_pred = []
    Y_true = []
    for X_batch, Y_batch in DataLoader(
        MyDataset(D=Data_test, args=args),
        batch_size=args.B,
        shuffle=False,
        drop_last=False,
    ):
        X_batch: torch.Tensor = X_batch.float().to(device=device)
        Y_batch: torch.Tensor = Y_batch.float().to(device=device)
        Y_batch_pred: torch.Tensor = model(X_batch)
        Y_pred.append(Y_batch_pred.cpu())
        Y_true.append(Y_batch.cpu())
    Y_pred: torch.Tensor = torch.concat(Y_pred)
    Y_true: torch.Tensor = torch.concat(Y_true)
    return mse(Y_true=Y_true, Y_pred=Y_pred), mae(Y_true=Y_true, Y_pred=Y_pred)


def do_exp(
    Data: torch.Tensor,
    device: torch.device,
    ModelType: Type[nn.Module],
    args: Args,
):

    model = ModelType(args).to(device=device)
    Data_train, Data_vali, Data_test = split_data(Data=Data, P=args.P)
    do_train(
        Data_train=Data_train,
        model=model,
        device=device,
        args=args,
    )
    mse_loss, mae_loss = do_test(
        Data_test=Data_test,
        model=model,
        device=device,
        args=args,
    )
    print(f"{mse_loss=},{mae_loss=}")
    return mse_loss, mae_loss


def main():

    device = get_device()
    torch.set_default_dtype(torch.float)

    scaler = StandardScaler()
    args_m = Args()
    args_s = Args()

    # 多变量
    Data_M = get_data(label="M")
    scaler.fit(Data_M)
    Data_M = scaler.transform(Data_M)
    args_m.C = Data_M.shape[-1]
    # 单变量 OT
    Data_S = get_data(label="S")
    scaler.fit(Data_S)
    Data_S = scaler.transform(Data_S)
    args_s.C = 1

    P_values: list[int] = [96, 192, 336, 720]
    F_values: list[int] = [720]
    Model_types: list[Type[nn.Module]] = [Lstm]

    F: int
    Model_i: Type[nn.Module]

    for P, F, Model_i in product(P_values, F_values, Model_types):
        args_m.P = P
        args_m.F = F
        args_s.P = P
        args_s.F = F
        # 单变量 OT
        print(f"{Model_i.__name__}_S_P{P}_F{F}")
        mse_loss, mae_loss = do_exp(
            Data=Data_S,
            ModelType=Model_i,
            device=device,
            args=args_s,
        )
        print(f"{mse_loss=},{mae_loss=}")

        # 多变量
        print(f"{Model_i.__name__}_M_P{P}_F{F}")
        mse_loss, mae_loss = do_exp(
            Data=Data_M,
            device=device,
            ModelType=Model_i,
            args=args_m,
        )
        print(f"{mse_loss=},{mae_loss=}")


main()
