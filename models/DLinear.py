from loguru import logger
from paprika import *

import torch
import torch.nn as nn
import objprint

logger.remove()


@objprint.add_objprint
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor):
        # padding on the both ends of time series

        logger.info(F"Input {x.shape=}")

        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        logger.info(F"Output {x.shape=}")

        return x


@objprint.add_objprint
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        logger.info(F"Input {x.shape=}")

        moving_mean = self.moving_avg(x)
        res = x - moving_mean

        logger.info(F"Output {res.shape=} {moving_mean.shape=}")

        return res, moving_mean


@objprint.add_objprint
class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):

        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        logger.info(configs)
        logger.info(F"{self.Linear_Trend=}")
        logger.info(F"{self.Linear_Seasonal=}")

    def forward(self, x: torch.Tensor):

        # x_in: [Batch, Input length, Channel]
        # Batch (批次): 这个维度指的是送入模型进行训练或预测的数据样本数量。在这个例子中，Batch 大小为 16，意味着每次前向传播或反向传播过程中，模型会同时处理 16 个样本。
        # Input length (输入长度): 表示每个样本的时间序列或者序列数据的长度。这里的 Input length 是 96，意味着每个样本包含 96 个时间点的数据或者说是 96 个连续的特征向量。
        # Channel (通道): 在时间序列预测或处理一维信号的场景中，通道通常对应于不同特征的数目或是传感器的数量。如果有多个并行的数据流（如同一时间段内不同地点的温度测量），每个数据流可以视为一个通道。此处 Channel 数量为 321，说明每个时间点有 321 个特征或变量。

        # x_out: [Batch, Output length, Channel]
        # Batch (批次): 输出的批次大小也是 16，表示模型对同样数量的样本进行了处理，这在批处理训练或预测时保持不变。
        # Output length (输出长度): 在常规情况下，输出长度应当代表预测的未来时间点数，但此处假设输出长度与输入长度相同（96），这可能意味着模型不仅预测未来，还在每个输入时间点上都有一个对应的输出（例如，在某些场景下模型可能对输入序列进行某种转换或处理，而不仅仅是预测）。实际应用中，若模型目的是预测未来，输出长度应为预测的点数，比如预测接下来的12个小时、一周的气温等，这时Output length就会不同于Input length。
        # Channel (通道): 输出的通道数同样是 321，表明对于每个时间点，模型都预测了与输入相同数量的特征或变量。这在多变量预测任务中很常见，比如同时预测多个相关指标的未来值。

        logger.info(F"Input {x.shape=}")

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x: torch.Tensor = seasonal_output + trend_output
        x: torch.Tensor = x.permute(0, 2, 1)

        logger.info(F"Input {x.shape=}")

        return x


@to_string
@data
class Config:
    seq_len: int = 336
    pred_len: int = 96
    enc_in: int = 321  # Channel or Variates
    individual: bool = False


def main():
    use_cuda = False
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    x_in: torch.Tensor = torch.load("models.DLinear-Input", map_location=device)
    dlm = Model(Config())
    dlm(x_in)
    objprint.objprint(dlm)


if __name__ == '__main__':
    main()
