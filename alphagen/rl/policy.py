import gymnasium as gym
import math
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from alphagen.data.expression import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化位置编码模块。
        使用正弦和余弦函数生成位置编码矩阵，用于为序列模型提供位置信息。
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        "x: ([batch_size, ]seq_len, embedding_dim)"
        """
        前向传播。
        将预计算的位置编码叠加到输入特征上。
        """
        # 判断输入维度以获取序列长度（兼容 (seq_len, dim) 和 (batch, seq_len, dim)）
        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        return x + self._pe[:seq_len]  # type: ignore


class TransformerSharedNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_encoder_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        device: torch.device
    ):
        """
        初始化基于 Transformer 的共享特征提取网络。
        """
        super().__init__(observation_space, d_model)

        # 确保观察空间是 Box 类型
        assert isinstance(observation_space, gym.spaces.Box)
        # 获取动作空间的大小（即词汇表大小），用于 Embedding
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        self._transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head,
                dim_feedforward=d_ffn, dropout=dropout,
                activation=lambda x: F.leaky_relu(x),               # type: ignore
                batch_first=True, device=device
            ),
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(d_model, eps=1e-5, device=device)
        )

    def forward(self, obs: Tensor) -> Tensor:
        """
        前向传播。
        处理输入序列，通过 Embedding、位置编码和 Transformer 编码器。
        """
        bs, seqlen = obs.shape
        # 创建一个全为 BEG（开始）Token 的张量，形状为 (batch_size, 1)
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        # 将 BEG Token 拼接到观察序列的最前面
        obs = torch.cat((beg, obs.long()), dim=1)
        # 创建填充掩码，输入中为 0 的位置被视为填充（Padding）
        pad_mask = obs == 0
        src = self._pos_enc(self._token_emb(obs))
        res = self._transformer(src, src_key_padding_mask=pad_mask)
        # 对序列维度取平均，得到固定长度的特征向量
        return res.mean(dim=1)


class LSTMSharedNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_layers: int,
        d_model: int,
        dropout: float,
        device: torch.device
    ):
        """
        初始化基于 LSTM 的共享特征提取网络。
        """
        super().__init__(observation_space, d_model)

        # 确保观察空间是 Box 类型
        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        self._lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, obs: Tensor) -> Tensor:
        """
        前向传播。
        处理输入序列，通过 Embedding、位置编码和 LSTM。
        """
        bs, seqlen = obs.shape
        # 创建全为 BEG Token 的张量
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        # 将 BEG Token 拼接到观察序列前面
        obs = torch.cat((beg, obs.long()), dim=1)
        # 计算每个样本的实际长度（非填充部分的长度），用于优化计算（虽然这里并未用 pack_padded_sequence）
        real_len = (obs != 0).sum(1).max()
        src = self._pos_enc(self._token_emb(obs))
        # 截取到最大实际长度输入 LSTM
        res = self._lstm(src[:,:real_len])[0]
        # 对序列维度取平均
        return res.mean(dim=1)


class Decoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        device: torch.device
    ):
        """
        初始化解码器网络（结构上类似于 TransformerSharedNet，这里作为特征提取器使用）。
        """
        super().__init__(observation_space, d_model)

        # 确保观察空间是 Box 类型
        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        # Actually an encoder for now
        self._decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=d_ffn,
                dropout=dropout, batch_first=True, device=device
            ),
            n_layers,
            norm=nn.LayerNorm(d_model, device=device)
        )

    def forward(self, obs: Tensor) -> Tensor:
        """
        前向传播。
        """
        batch_size = obs.size(0)
        # 创建 BEG Token 张量
        begins = torch.full(size=(batch_size, 1), fill_value=self._n_actions,
                            dtype=torch.long, device=obs.device)
        # 拼接 BEG Token 到序列头部
        obs = torch.cat((begins, obs.type(torch.long)), dim=1)      # (bs, len)
        # 创建填充掩码
        pad_mask = obs == 0
        res = self._token_emb(obs)                                  # (bs, len, d_model)
        res = self._pos_enc(res)                                    # (bs, len, d_model)
        res = self._decoder(res, src_key_padding_mask=pad_mask)     # (bs, len, d_model)
        # 返回序列平均值
        return res.mean(dim=1)
