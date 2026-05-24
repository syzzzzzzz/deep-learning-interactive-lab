"""
Neural network Lego factory / central playground.

Run:
    streamlit run part6_universal_framework/neural_network_playground.py
or:
    python main.py part6/neural_network_playground
"""

from __future__ import annotations

import copy
import json
import math
import textwrap
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from plotly.subplots import make_subplots


PLAYGROUND_TARGET = "part6_universal_framework/neural_network_playground"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
PLOT_FONT = {"family": "Microsoft YaHei, SimHei, Segoe UI, sans-serif", "color": "#172026"}
PLAYGROUND_BATCH_SIZE = 16

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass


COMPONENT_REGISTRY: dict[str, dict[str, Any]] = {
    "Linear": {
        "name": "Linear",
        "type": "layer",
        "params": {"in_features": 784, "out_features": 128, "bias": True},
        "input_rule": "输入必须是一维特征向量，或二维序列表示；最后一维等于 in_features。",
        "output_rule": "输出最后一维变为 out_features。",
        "description": "全连接层，把输入特征映射到新的特征空间。",
        "pytorch_code": "nn.Linear(in_features, out_features, bias=True)",
        "related_topic": "神经网络基础 / MLP",
    },
    "Conv2d": {
        "name": "Conv2d",
        "type": "layer",
        "params": {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
        "input_rule": "输入必须是 (channels, height, width)，channels 等于 in_channels。",
        "output_rule": "输出为 (out_channels, H_out, W_out)。",
        "description": "二维卷积层，用局部感受野提取图像空间特征。",
        "pytorch_code": "nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)",
        "related_topic": "CNN / 卷积直觉",
    },
    "MaxPool2d": {
        "name": "MaxPool2d",
        "type": "layer",
        "params": {"kernel_size": 2, "stride": 2},
        "input_rule": "输入必须是 (channels, height, width)。",
        "output_rule": "通道数不变，空间尺寸按池化窗口缩小。",
        "description": "最大池化层，降低空间分辨率并保留强响应。",
        "pytorch_code": "nn.MaxPool2d(kernel_size, stride)",
        "related_topic": "CNN / 池化",
    },
    "ReLU": {
        "name": "ReLU",
        "type": "activation",
        "params": {},
        "input_rule": "任意张量形状。",
        "output_rule": "形状不变。",
        "description": "常用非线性激活函数，截断负值。",
        "pytorch_code": "nn.ReLU()",
        "related_topic": "激活函数",
    },
    "Sigmoid": {
        "name": "Sigmoid",
        "type": "activation",
        "params": {},
        "input_rule": "任意张量形状。",
        "output_rule": "形状不变。",
        "description": "把值压到 0 到 1 之间，常见于二分类概率输出。",
        "pytorch_code": "nn.Sigmoid()",
        "related_topic": "激活函数",
    },
    "Tanh": {
        "name": "Tanh",
        "type": "activation",
        "params": {},
        "input_rule": "任意张量形状。",
        "output_rule": "形状不变。",
        "description": "把值压到 -1 到 1 之间，早期序列模型常用。",
        "pytorch_code": "nn.Tanh()",
        "related_topic": "激活函数",
    },
    "Dropout": {
        "name": "Dropout",
        "type": "regularization",
        "params": {"p": 0.5},
        "input_rule": "任意张量形状。",
        "output_rule": "形状不变。",
        "description": "训练时随机置零部分激活，缓解过拟合。",
        "pytorch_code": "nn.Dropout(p)",
        "related_topic": "正则化",
    },
    "BatchNorm": {
        "name": "BatchNorm",
        "type": "normalization",
        "params": {"num_features": 128},
        "input_rule": "一维特征向量使用 BatchNorm1d；图像特征图使用 BatchNorm2d。",
        "output_rule": "形状不变。",
        "description": "批归一化层，稳定激活分布并加速训练。",
        "pytorch_code": "nn.BatchNorm1d(num_features) / nn.BatchNorm2d(num_features)",
        "related_topic": "归一化",
    },
    "LayerNorm": {
        "name": "LayerNorm",
        "type": "normalization",
        "params": {"normalized_shape": 64, "eps": 1e-5},
        "input_rule": "输入可以是一维特征或序列表示；最后一维等于 normalized_shape。",
        "output_rule": "形状不变，只对最后一维做归一化。",
        "description": "层归一化，不依赖 batch 统计，是 Transformer 中的默认归一化方式。",
        "pytorch_code": "nn.LayerNorm(normalized_shape, eps=1e-5)",
        "related_topic": "Transformer / LayerNorm",
    },
    "Flatten": {
        "name": "Flatten",
        "type": "reshape",
        "params": {"start_dim": 1, "end_dim": -1},
        "input_rule": "按 PyTorch 约定保留 batch 维，默认从第 1 维开始展平。",
        "output_rule": "把指定范围内的维度合并成一个特征维。",
        "description": "把卷积特征图展平成向量，通常接 Linear 层。",
        "pytorch_code": "nn.Flatten(start_dim=1, end_dim=-1)",
        "related_topic": "CNN / MLP 衔接",
    },
    "GELU": {
        "name": "GELU",
        "type": "activation",
        "params": {},
        "input_rule": "任意张量形状。",
        "output_rule": "形状不变。",
        "description": "Transformer 和现代 MLP 常用激活函数，比 ReLU 更平滑。",
        "pytorch_code": "nn.GELU()",
        "related_topic": "Transformer / 激活函数",
    },
    "ResidualBlock": {
        "name": "ResidualBlock",
        "type": "block",
        "params": {"dim": 128, "hidden_dim": 256, "dropout": 0.1},
        "input_rule": "输入最后一维必须等于 dim，可用于向量或序列 token 表示。",
        "output_rule": "输出形状与输入完全相同。",
        "description": "残差 MLP 块，学习 F(x) 后再和输入 x 相加，缓解深层网络退化和梯度传播困难。",
        "pytorch_code": "ResidualMLPBlock(dim, hidden_dim, dropout)",
        "related_topic": "ResNet / 残差连接",
    },
    "MultiheadAttention": {
        "name": "MultiheadAttention",
        "type": "attention",
        "params": {"embed_dim": 64, "num_heads": 4, "dropout": 0.1},
        "input_rule": "输入必须是 (seq_len, embed_dim)，即 batch 后的 token 序列表示。",
        "output_rule": "输出形状与输入相同。",
        "description": "自注意力块，让每个 token 根据上下文重新汇聚信息。",
        "pytorch_code": "SelfAttentionBlock(embed_dim, num_heads, dropout)",
        "related_topic": "多头注意力 / Transformer",
    },
    "TransformerEncoder": {
        "name": "TransformerEncoder",
        "type": "block",
        "params": {"d_model": 64, "nhead": 4, "dim_feedforward": 128, "num_layers": 1, "dropout": 0.1},
        "input_rule": "输入必须是 (seq_len, d_model)，且 d_model 能被 nhead 整除。",
        "output_rule": "输出形状与输入相同。",
        "description": "标准 Transformer Encoder 堆叠，内部包含多头注意力、前馈网络、残差和 LayerNorm。",
        "pytorch_code": "nn.TransformerEncoder(nn.TransformerEncoderLayer(...), num_layers)",
        "related_topic": "Transformer Encoder",
    },
}


LOSS_REGISTRY: dict[str, dict[str, str]] = {
    "MSE": {"name": "MSE", "pytorch_code": "nn.MSELoss", "description": "均方误差，常用于回归任务。"},
    "CrossEntropy": {
        "name": "CrossEntropy",
        "pytorch_code": "nn.CrossEntropyLoss",
        "description": "交叉熵损失，常用于多分类 logits。",
    },
}


OPTIMIZER_REGISTRY: dict[str, dict[str, Any]] = {
    "SGD": {
        "name": "SGD",
        "params": {"lr": 0.01, "momentum": 0.0},
        "description": "随机梯度下降，简单稳定，momentum 可加速。",
        "pytorch_code": "torch.optim.SGD",
    },
    "Adam": {
        "name": "Adam",
        "params": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8},
        "description": "自适应优化器，常作为深度学习默认起点。",
        "pytorch_code": "torch.optim.Adam",
    },
}


PRESETS: dict[str, dict[str, Any]] = {
    "mlp": {
        "title": "MLP 分类器（MNIST）",
        "input_shape": (1, 28, 28),
        "loss": "CrossEntropy",
        "optimizer": "Adam",
        "layers": [
            {"component": "Flatten", "params": {"start_dim": 1, "end_dim": -1}},
            {"component": "Linear", "params": {"in_features": 784, "out_features": 128, "bias": True}},
            {"component": "ReLU", "params": {}},
            {"component": "Linear", "params": {"in_features": 128, "out_features": 10, "bias": True}},
        ],
    },
    "cnn": {
        "title": "CNN 图像分类器",
        "input_shape": (1, 28, 28),
        "loss": "CrossEntropy",
        "optimizer": "Adam",
        "layers": [
            {"component": "Conv2d", "params": {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"component": "ReLU", "params": {}},
            {"component": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
            {"component": "Conv2d", "params": {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"component": "ReLU", "params": {}},
            {"component": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
            {"component": "Flatten", "params": {"start_dim": 1, "end_dim": -1}},
            {"component": "Linear", "params": {"in_features": 1568, "out_features": 128, "bias": True}},
            {"component": "ReLU", "params": {}},
            {"component": "Linear", "params": {"in_features": 128, "out_features": 10, "bias": True}},
        ],
    },
    "transformer": {
        "title": "简化 Transformer 示例",
        "input_shape": (16, 64),
        "loss": "MSE",
        "optimizer": "Adam",
        "layers": [
            {"component": "Linear", "params": {"in_features": 64, "out_features": 64, "bias": True}},
            {"component": "LayerNorm", "params": {"normalized_shape": 64, "eps": 1e-5}},
            {"component": "MultiheadAttention", "params": {"embed_dim": 64, "num_heads": 4, "dropout": 0.1}},
            {"component": "ResidualBlock", "params": {"dim": 64, "hidden_dim": 128, "dropout": 0.1}},
            {"component": "TransformerEncoder", "params": {"d_model": 64, "nhead": 4, "dim_feedforward": 128, "num_layers": 1, "dropout": 0.1}},
            {"component": "Linear", "params": {"in_features": 64, "out_features": 64, "bias": True}},
        ],
        "note": "这个预设展示序列输入里的 LayerNorm、多头注意力、残差 MLP 和 Transformer Encoder 如何保持 token 形状稳定。",
    },
    "residual_mlp": {
        "title": "残差 MLP 分类器",
        "input_shape": (784,),
        "loss": "CrossEntropy",
        "optimizer": "Adam",
        "layers": [
            {"component": "Linear", "params": {"in_features": 784, "out_features": 128, "bias": True}},
            {"component": "LayerNorm", "params": {"normalized_shape": 128, "eps": 1e-5}},
            {"component": "GELU", "params": {}},
            {"component": "ResidualBlock", "params": {"dim": 128, "hidden_dim": 256, "dropout": 0.1}},
            {"component": "Linear", "params": {"in_features": 128, "out_features": 10, "bias": True}},
        ],
        "note": "残差块要求输入输出维度一致。它不负责改变维度，而是让网络在原表示上学习一个修正量 F(x)。",
    },
}


@dataclass(frozen=True)
class ShapeStep:
    index: int
    layer: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...] | None
    message: str
    ok: bool
    code: str


@dataclass
class PlaygroundTrainingHistory:
    epochs: list[int]
    losses: list[float]
    accuracies: list[float | None]
    learning_rates: list[float]
    grad_norms: dict[str, list[float]]
    update_ratios: list[float]
    cnn_feature_maps: np.ndarray | None
    attention_heatmap: np.ndarray | None
    attention_tokens: list[str]
    attention_is_simulated: bool
    mode_notes: list[str]
    effective_loss: str


class PlaygroundResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PlaygroundSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        attended, weights = self.attn(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention_weights = weights.detach()
        return x + self.dropout(attended)


class PlaygroundBuiltModel(nn.Module):
    def __init__(self, modules: list[nn.Module], layer_labels: list[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(modules)
        self.layer_labels = layer_labels
        self.last_cnn_feature_maps: torch.Tensor | None = None
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_cnn_feature_maps = None
        self.last_attention_weights = None
        for module in self.layers:
            x = module(x)
            if self.last_cnn_feature_maps is None and isinstance(module, nn.Conv2d):
                self.last_cnn_feature_maps = x.detach()
            if isinstance(module, PlaygroundSelfAttentionBlock) and module.last_attention_weights is not None:
                self.last_attention_weights = module.last_attention_weights
        return x


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def go_home() -> None:
    st.query_params.clear()
    _rerun()


def open_playground(example: str) -> None:
    st.query_params["module"] = PLAYGROUND_TARGET
    st.query_params["example"] = example
    _rerun()


def parse_shape(text: str) -> tuple[int, ...]:
    cleaned = text.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    if not cleaned:
        raise ValueError("请输入形状，例如 1, 28, 28 或 784。")
    dims = tuple(int(item.strip()) for item in cleaned.split(",") if item.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("形状中的每个维度都必须是正整数。")
    return dims


def format_shape(shape: Iterable[int] | None) -> str:
    if shape is None:
        return "-"
    dims = tuple(shape)
    if len(dims) == 1:
        return f"({dims[0]},)"
    return "(" + ", ".join(str(dim) for dim in dims) + ")"


def clone_layers(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(layers)


def default_state() -> None:
    if "playground_layers" not in st.session_state:
        st.session_state.playground_layers = clone_layers(PRESETS["mlp"]["layers"])
    if "playground_input_shape" not in st.session_state:
        st.session_state.playground_input_shape = format_shape(PRESETS["mlp"]["input_shape"])
    if "playground_loss" not in st.session_state:
        st.session_state.playground_loss = "CrossEntropy"
    if "playground_optimizer" not in st.session_state:
        st.session_state.playground_optimizer = "Adam"
    if "playground_optimizer_params" not in st.session_state:
        st.session_state.playground_optimizer_params = {
            name: copy.deepcopy(item["params"]) for name, item in OPTIMIZER_REGISTRY.items()
        }
    if "playground_loaded_example" not in st.session_state:
        st.session_state.playground_loaded_example = ""
    if "playground_project_name" not in st.session_state:
        st.session_state.playground_project_name = "我的神经网络结构"
    if "playground_last_import_error" not in st.session_state:
        st.session_state.playground_last_import_error = ""


def load_preset(key: str) -> None:
    preset = PRESETS[key]
    st.session_state.playground_layers = clone_layers(preset["layers"])
    st.session_state.playground_input_shape = format_shape(preset["input_shape"])
    st.session_state.playground_loss = preset["loss"]
    st.session_state.playground_optimizer = preset["optimizer"]
    st.session_state.playground_loaded_example = key
    st.session_state.playground_project_name = preset["title"]


def query_example() -> str:
    value = st.query_params.get("example", "")
    if isinstance(value, list):
        value = value[0] if value else ""
    value = str(value).strip().lower()
    return value if value in PRESETS else ""


def ensure_query_preset_loaded() -> None:
    example = query_example()
    if example and st.session_state.playground_loaded_example != example:
        load_preset(example)


def infer_layer_shape(component: str, params: dict[str, Any], current: tuple[int, ...]) -> tuple[tuple[int, ...], str, str]:
    if component == "Linear":
        expected = int(params["in_features"])
        out_features = int(params["out_features"])
        if len(current) == 3:
            raise ValueError(
                f"Linear 层需要一维特征向量；上一层输出是 {format_shape(current)}，请先添加 Flatten。"
            )
        actual = current[-1]
        if actual != expected:
            previous = "上一层"
            raise ValueError(f"Linear 层的输入特征数 {expected} 与{previous}的输出 {actual} 不匹配。")
        if len(current) == 1:
            return (out_features,), "最后一维替换为 out_features。", layer_code(component, params, current)
        return (*current[:-1], out_features), "序列长度保持不变，最后一维替换为 out_features。", layer_code(component, params, current)

    if component == "Conv2d":
        if len(current) != 3:
            raise ValueError(f"Conv2d 层需要 (channels, height, width) 输入，当前输入是 {format_shape(current)}。")
        channels, height, width = current
        in_channels = int(params["in_channels"])
        if channels != in_channels:
            raise ValueError(f"Conv2d 层的输入通道数 {in_channels} 与上一层输出通道数 {channels} 不匹配。")
        kernel = int(params["kernel_size"])
        stride = int(params["stride"])
        padding = int(params["padding"])
        h_out = math.floor((height + 2 * padding - kernel) / stride + 1)
        w_out = math.floor((width + 2 * padding - kernel) / stride + 1)
        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"Conv2d 的 kernel_size={kernel}, stride={stride}, padding={padding} 会让空间尺寸变为 "
                f"{h_out}x{w_out}，请调小卷积核或增大 padding。"
            )
        return (int(params["out_channels"]), h_out, w_out), "按卷积公式计算空间尺寸。", layer_code(component, params, current)

    if component == "MaxPool2d":
        if len(current) != 3:
            raise ValueError(f"MaxPool2d 层需要 (channels, height, width) 输入，当前输入是 {format_shape(current)}。")
        channels, height, width = current
        kernel = int(params["kernel_size"])
        stride = int(params["stride"])
        h_out = math.floor((height - kernel) / stride + 1)
        w_out = math.floor((width - kernel) / stride + 1)
        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"MaxPool2d 的 kernel_size={kernel}, stride={stride} 会让空间尺寸变为 {h_out}x{w_out}。"
            )
        return (channels, h_out, w_out), "通道不变，空间尺寸按池化公式缩小。", layer_code(component, params, current)

    if component == "BatchNorm":
        if len(current) not in (1, 3):
            raise ValueError(f"BatchNorm 当前只支持一维特征或图像特征图，当前输入是 {format_shape(current)}。")
        expected = int(params["num_features"])
        actual = current[0]
        if actual != expected:
            raise ValueError(f"BatchNorm 的 num_features={expected} 与上一层输出特征/通道数 {actual} 不匹配。")
        code = layer_code(component, params, current)
        return current, "归一化不改变形状。", code

    if component == "LayerNorm":
        expected = int(params["normalized_shape"])
        actual = current[-1]
        if actual != expected:
            raise ValueError(
                f"LayerNorm 的 normalized_shape={expected} 必须等于输入最后一维 {actual}。"
            )
        return current, "LayerNorm 只归一化最后一维，形状不变。", layer_code(component, params, current)

    if component == "Flatten":
        start_dim = int(params["start_dim"])
        end_dim = int(params["end_dim"])
        if start_dim < 1:
            raise ValueError("控制台按 batch 之后的形状推导，Flatten 的 start_dim 请设为 1 或更大。")
        relative_start = start_dim - 1
        relative_end = len(current) - 1 if end_dim == -1 else end_dim - 1
        if relative_start >= len(current) or relative_end >= len(current) or relative_start > relative_end:
            raise ValueError(
                f"Flatten(start_dim={start_dim}, end_dim={end_dim}) 超出当前形状 {format_shape(current)} 的范围。"
            )
        flattened = math.prod(current[relative_start : relative_end + 1])
        output = (*current[:relative_start], flattened, *current[relative_end + 1 :])
        return output, "指定范围内的维度被合并。", layer_code(component, params, current)

    if component == "ResidualBlock":
        dim = int(params["dim"])
        hidden_dim = int(params["hidden_dim"])
        dropout = float(params["dropout"])
        if current[-1] != dim:
            raise ValueError(
                f"ResidualBlock 要求输入最后一维等于 dim={dim}，但上一层输出最后一维是 {current[-1]}。"
                "请先用 Linear 对齐维度。"
            )
        if hidden_dim < dim:
            raise ValueError("ResidualBlock 的 hidden_dim 建议大于等于 dim，否则表达能力会被压窄。")
        if not 0.0 <= dropout <= 0.9:
            raise ValueError("ResidualBlock 的 dropout 建议在 0.0 到 0.9 之间。")
        return current, "残差块输出与输入相加，因此形状必须保持不变。", layer_code(component, params, current)

    if component == "MultiheadAttention":
        if len(current) != 2:
            raise ValueError(
                f"MultiheadAttention 需要 (seq_len, embed_dim) 输入，当前是 {format_shape(current)}。"
                "图像特征请先展平成序列，向量输入请增加序列维。"
            )
        embed_dim = int(params["embed_dim"])
        num_heads = int(params["num_heads"])
        if current[-1] != embed_dim:
            raise ValueError(f"embed_dim={embed_dim} 必须等于输入最后一维 {current[-1]}。")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} 必须能被 num_heads={num_heads} 整除。")
        return current, "自注意力重新混合 token 信息，但保持序列长度和维度不变。", layer_code(component, params, current)

    if component == "TransformerEncoder":
        if len(current) != 2:
            raise ValueError(
                f"TransformerEncoder 需要 (seq_len, d_model) 输入，当前是 {format_shape(current)}。"
            )
        d_model = int(params["d_model"])
        nhead = int(params["nhead"])
        if current[-1] != d_model:
            raise ValueError(f"d_model={d_model} 必须等于输入最后一维 {current[-1]}。")
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} 必须能被 nhead={nhead} 整除。")
        if int(params["num_layers"]) < 1:
            raise ValueError("TransformerEncoder 的 num_layers 至少为 1。")
        return current, "Encoder 层内部包含注意力、前馈、残差和归一化，整体形状不变。", layer_code(component, params, current)

    if component in {"ReLU", "Sigmoid", "Tanh", "Dropout", "GELU"}:
        return current, "该层不改变张量形状。", layer_code(component, params, current)

    raise ValueError(f"未知组件：{component}")


def infer_shapes(input_shape: tuple[int, ...], layers: list[dict[str, Any]]) -> list[ShapeStep]:
    current = input_shape
    steps: list[ShapeStep] = []
    for index, layer in enumerate(layers, 1):
        component = layer["component"]
        params = layer.get("params", {})
        try:
            output, message, code = infer_layer_shape(component, params, current)
            steps.append(ShapeStep(index, component, current, output, message, True, code))
            current = output
        except Exception as exc:
            steps.append(ShapeStep(index, component, current, None, str(exc), False, layer_code(component, params, current)))
            break
    return steps


def param_repr(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, tuple):
        return "(" + ", ".join(param_repr(item) for item in value) + ")"
    if isinstance(value, list):
        return "(" + ", ".join(param_repr(item) for item in value) + ")"
    return repr(value)


def layer_code(component: str, params: dict[str, Any], current_shape: tuple[int, ...] | None = None) -> str:
    if component == "Linear":
        return f"nn.Linear({params['in_features']}, {params['out_features']}, bias={param_repr(params['bias'])})"
    if component == "Conv2d":
        return (
            f"nn.Conv2d({params['in_channels']}, {params['out_channels']}, "
            f"kernel_size={params['kernel_size']}, stride={params['stride']}, padding={params['padding']})"
        )
    if component == "MaxPool2d":
        return f"nn.MaxPool2d(kernel_size={params['kernel_size']}, stride={params['stride']})"
    if component == "Dropout":
        return f"nn.Dropout(p={params['p']})"
    if component == "BatchNorm":
        cls = "nn.BatchNorm2d" if current_shape and len(current_shape) == 3 else "nn.BatchNorm1d"
        return f"{cls}({params['num_features']})"
    if component == "LayerNorm":
        return f"nn.LayerNorm({params['normalized_shape']}, eps={params['eps']})"
    if component == "Flatten":
        return f"nn.Flatten(start_dim={params['start_dim']}, end_dim={params['end_dim']})"
    if component == "GELU":
        return "nn.GELU()"
    if component == "ResidualBlock":
        return f"ResidualMLPBlock({params['dim']}, {params['hidden_dim']}, dropout={params['dropout']})"
    if component == "MultiheadAttention":
        return f"SelfAttentionBlock({params['embed_dim']}, {params['num_heads']}, dropout={params['dropout']})"
    if component == "TransformerEncoder":
        return (
            "nn.TransformerEncoder("
            f"nn.TransformerEncoderLayer(d_model={params['d_model']}, nhead={params['nhead']}, "
            f"dim_feedforward={params['dim_feedforward']}, dropout={params['dropout']}, batch_first=True), "
            f"num_layers={params['num_layers']})"
        )
    return f"nn.{component}()"


def optimizer_param_code(name: str, params: dict[str, Any]) -> str:
    if name == "SGD":
        return f"lr={params['lr']}, momentum={params['momentum']}"
    if name == "Adam":
        return f"lr={params['lr']}, betas={param_repr(tuple(params['betas']))}, eps={params['eps']}"
    return ""


def generate_code(
    input_shape: tuple[int, ...],
    layers: list[dict[str, Any]],
    steps: list[ShapeStep],
    loss_name: str,
    optimizer_name: str,
    optimizer_params: dict[str, Any],
) -> str:
    valid_codes = [step.code for step in steps if step.ok]
    layer_block = ",\n".join(f"            {code}" for code in valid_codes) or "            nn.Identity()"
    criterion_code = f"{LOSS_REGISTRY[loss_name]['pytorch_code']}()"
    optimizer_code = (
        f"{OPTIMIZER_REGISTRY[optimizer_name]['pytorch_code']}"
        f"(model.parameters(), {optimizer_param_code(optimizer_name, optimizer_params)})"
    )
    used_components = {layer["component"] for layer in layers}
    helper_blocks: list[str] = []
    if "ResidualBlock" in used_components:
        helper_blocks.append(
            """
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)
"""
        )
    if "MultiheadAttention" in used_components:
        helper_blocks.append(
            """
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normalized = self.norm(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        return x + self.dropout(attended)
"""
        )
    helper_code = "\n\n".join(textwrap.dedent(block).strip() for block in helper_blocks)
    sections = [
        "import torch",
        "import torch.nn as nn",
    ]
    if helper_code:
        sections.append(helper_code)
    sections.append(
        textwrap.dedent(
            f"""
            class BuiltModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = nn.Sequential(
            {layer_block}
                    )

                def forward(self, x):
                    return self.model(x)


            input_shape = {format_shape(input_shape)}
            batch_size = 4

            model = BuiltModel()
            criterion = {criterion_code}
            optimizer = {optimizer_code}

            x = torch.randn(batch_size, *input_shape)
            output = model(x)
            print("output shape:", tuple(output.shape))
            """
        ).strip()
    )
    return "\n\n".join(sections)


def build_torch_layer(component: str, params: dict[str, Any], current_shape: tuple[int, ...]) -> nn.Module:
    if component == "Linear":
        return nn.Linear(int(params["in_features"]), int(params["out_features"]), bias=bool(params["bias"]))
    if component == "Conv2d":
        return nn.Conv2d(
            int(params["in_channels"]),
            int(params["out_channels"]),
            kernel_size=int(params["kernel_size"]),
            stride=int(params["stride"]),
            padding=int(params["padding"]),
        )
    if component == "MaxPool2d":
        return nn.MaxPool2d(kernel_size=int(params["kernel_size"]), stride=int(params["stride"]))
    if component == "ReLU":
        return nn.ReLU()
    if component == "Sigmoid":
        return nn.Sigmoid()
    if component == "Tanh":
        return nn.Tanh()
    if component == "Dropout":
        return nn.Dropout(p=float(params["p"]))
    if component == "BatchNorm":
        cls = nn.BatchNorm2d if len(current_shape) == 3 else nn.BatchNorm1d
        return cls(int(params["num_features"]))
    if component == "LayerNorm":
        return nn.LayerNorm(int(params["normalized_shape"]), eps=float(params["eps"]))
    if component == "Flatten":
        return nn.Flatten(start_dim=int(params["start_dim"]), end_dim=int(params["end_dim"]))
    if component == "GELU":
        return nn.GELU()
    if component == "ResidualBlock":
        return PlaygroundResidualMLPBlock(
            int(params["dim"]),
            int(params["hidden_dim"]),
            dropout=float(params["dropout"]),
        )
    if component == "MultiheadAttention":
        return PlaygroundSelfAttentionBlock(
            int(params["embed_dim"]),
            int(params["num_heads"]),
            dropout=float(params["dropout"]),
        )
    if component == "TransformerEncoder":
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(params["d_model"]),
            nhead=int(params["nhead"]),
            dim_feedforward=int(params["dim_feedforward"]),
            dropout=float(params["dropout"]),
            batch_first=True,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=int(params["num_layers"]))
    raise ValueError(f"暂不支持训练联动组件：{component}")


def build_playground_model(
    input_shape: tuple[int, ...],
    layers: list[dict[str, Any]],
) -> tuple[PlaygroundBuiltModel, tuple[int, ...]]:
    modules: list[nn.Module] = []
    labels: list[str] = []
    current_shape = input_shape
    for index, layer in enumerate(layers, 1):
        component = layer["component"]
        params = layer.get("params", {})
        modules.append(build_torch_layer(component, params, current_shape))
        labels.append(f"{index}. {component}")
        current_shape, _, _ = infer_layer_shape(component, params, current_shape)
    return PlaygroundBuiltModel(modules, labels), current_shape


def make_playground_inputs(input_shape: tuple[int, ...], batch_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, *input_shape, generator=generator) * 0.35
    if len(input_shape) == 3:
        channels, height, width = input_shape
        yy = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
        xx = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        x[:, :1] += xx + yy
        if channels > 1:
            x[:, 1:2] += xx - yy
    elif len(input_shape) == 2:
        seq_len, dim = input_shape
        positions = torch.linspace(0.0, 1.0, seq_len).view(1, seq_len, 1)
        frequencies = torch.linspace(1.0, 3.0, dim).view(1, 1, dim)
        x += torch.sin(positions * frequencies * math.pi)
    elif len(input_shape) == 1:
        x += torch.linspace(-1.0, 1.0, input_shape[0]).view(1, input_shape[0])
    return x.float()


def cross_entropy_layout(final_shape: tuple[int, ...]) -> tuple[str, int] | None:
    if len(final_shape) == 1 and final_shape[0] >= 2:
        return "vector", final_shape[0]
    if len(final_shape) == 2 and final_shape[-1] >= 2:
        return "sequence", final_shape[-1]
    if len(final_shape) == 3 and final_shape[0] >= 2:
        return "spatial", final_shape[0]
    return None


def make_playground_target(
    final_shape: tuple[int, ...],
    requested_loss: str,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, str, list[str]]:
    generator = torch.Generator().manual_seed(seed + 1009)
    notes: list[str] = []
    if requested_loss == "CrossEntropy":
        layout = cross_entropy_layout(final_shape)
        if layout:
            kind, class_count = layout
            if kind == "vector":
                return torch.randint(class_count, (batch_size,), generator=generator), "CrossEntropy", notes
            if kind == "sequence":
                return (
                    torch.randint(class_count, (batch_size, final_shape[0]), generator=generator),
                    "CrossEntropy",
                    notes,
                )
            return (
                torch.randint(class_count, (batch_size, final_shape[1], final_shape[2]), generator=generator),
                "CrossEntropy",
                notes,
            )
        notes.append("当前输出形状不适合交叉熵分类，联动区已自动切换为 MSE 回归目标。")

    target = torch.tanh(torch.randn(batch_size, *final_shape, generator=generator))
    return target.float(), "MSE", notes


def compute_playground_loss(
    criterion: nn.Module,
    output: torch.Tensor,
    target: torch.Tensor,
    effective_loss: str,
) -> torch.Tensor:
    if effective_loss == "CrossEntropy":
        if output.ndim == 2:
            return criterion(output, target)
        if output.ndim == 3:
            return criterion(output.reshape(-1, output.shape[-1]), target.reshape(-1))
        if output.ndim == 4:
            return criterion(output, target)
    return criterion(output, target)


def compute_playground_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    effective_loss: str,
) -> float | None:
    if effective_loss != "CrossEntropy":
        return None
    if output.ndim == 2:
        prediction = output.argmax(dim=-1)
    elif output.ndim == 3:
        prediction = output.argmax(dim=-1)
    elif output.ndim == 4:
        prediction = output.argmax(dim=1)
    else:
        return None
    return float((prediction == target).float().mean().item())


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_playground_optimizer(
    model: nn.Module,
    optimizer_name: str,
    optimizer_params: dict[str, Any],
    learning_rate: float,
) -> torch.optim.Optimizer:
    parameters = trainable_parameters(model)
    if not parameters:
        raise ValueError("当前结构没有可训练参数，请至少加入 Linear、Conv2d、注意力或 Transformer 层。")
    if optimizer_name == "SGD":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=float(optimizer_params.get("momentum", 0.0)))
    if optimizer_name == "Adam":
        betas = tuple(optimizer_params.get("betas", (0.9, 0.999)))
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(float(betas[0]), float(betas[1])),
            eps=float(optimizer_params.get("eps", 1e-8)),
        )
    raise ValueError(f"暂不支持优化器：{optimizer_name}")


def collect_playground_grad_norms(model: PlaygroundBuiltModel) -> dict[str, float]:
    norms: dict[str, float] = {}
    for label, module in zip(model.layer_labels, model.layers, strict=True):
        squared = 0.0
        for parameter in module.parameters(recurse=True):
            if parameter.grad is None:
                continue
            squared += float(parameter.grad.detach().norm(2).item() ** 2)
        norms[label] = float(squared**0.5)
    return norms


def parameter_update_ratio(before: list[torch.Tensor], parameters: list[nn.Parameter]) -> float:
    update_squared = 0.0
    param_squared = 0.0
    for old, parameter in zip(before, parameters, strict=True):
        current = parameter.detach()
        update_squared += float((current - old).norm(2).item() ** 2)
        param_squared += float(old.norm(2).item() ** 2)
    return float((update_squared**0.5) / ((param_squared**0.5) + 1e-12))


def extract_cnn_feature_maps(feature_maps: torch.Tensor | None) -> np.ndarray | None:
    if feature_maps is None or feature_maps.ndim != 4:
        return None
    maps = feature_maps.detach().cpu()[0]
    maps = maps[: min(6, maps.shape[0])]
    array = maps.numpy()
    normalized: list[np.ndarray] = []
    for channel in array:
        spread = float(channel.max() - channel.min())
        if spread < 1e-12:
            normalized.append(np.zeros_like(channel))
        else:
            normalized.append((channel - channel.min()) / spread)
    return np.stack(normalized)


def extract_attention_heatmap(weights: torch.Tensor | None) -> np.ndarray | None:
    if weights is None:
        return None
    array = weights.detach().cpu()
    if array.ndim == 4:
        array = array[0].mean(dim=0)
    elif array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        return None
    return array.numpy()


def make_similarity_attention_heatmap(x: torch.Tensor) -> np.ndarray | None:
    if x.ndim != 3:
        return None
    tokens = x.detach()[0]
    tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scores = tokens @ tokens.T
    weights = torch.softmax(scores * 3.0, dim=-1)
    return weights.cpu().numpy()


def attention_token_labels(length: int) -> list[str]:
    return [f"token {index}" for index in range(length)]


def run_playground_training(
    input_shape: tuple[int, ...],
    layers: list[dict[str, Any]],
    steps: list[ShapeStep],
    loss_name: str,
    optimizer_name: str,
    optimizer_params: dict[str, Any],
    epochs: int = 8,
    learning_rate: float | None = None,
    seed: int = 7,
    batch_size: int = PLAYGROUND_BATCH_SIZE,
) -> PlaygroundTrainingHistory:
    if not steps or any(not step.ok for step in steps):
        raise ValueError("形状推导没有通过，暂时不能运行联动训练。")
    final_shape = steps[-1].output_shape
    if final_shape is None:
        raise ValueError("缺少最终输出形状，暂时不能运行联动训练。")

    torch.manual_seed(seed)
    model, inferred_final_shape = build_playground_model(input_shape, layers)
    x = make_playground_inputs(input_shape, batch_size=batch_size, seed=seed)
    target, effective_loss, target_notes = make_playground_target(
        inferred_final_shape,
        loss_name,
        batch_size=batch_size,
        seed=seed,
    )
    criterion: nn.Module = nn.CrossEntropyLoss() if effective_loss == "CrossEntropy" else nn.MSELoss()
    lr = float(learning_rate if learning_rate is not None else optimizer_params.get("lr", 0.001))
    if lr <= 0:
        raise ValueError("联动训练学习率必须大于 0。")
    optimizer = build_playground_optimizer(model, optimizer_name, optimizer_params, lr)

    epoch_values: list[int] = []
    losses: list[float] = []
    accuracies: list[float | None] = []
    learning_rates: list[float] = []
    update_ratios: list[float] = []
    tracked_grad_labels = [
        label
        for label, module in zip(model.layer_labels, model.layers, strict=True)
        if any(parameter.requires_grad for parameter in module.parameters(recurse=True))
    ]
    grad_history: dict[str, list[float]] = {label: [] for label in tracked_grad_labels}

    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        parameters = trainable_parameters(model)
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = compute_playground_loss(criterion, output, target, effective_loss)
        loss.backward()
        grad_snapshot = collect_playground_grad_norms(model)
        optimizer.step()
        ratio = parameter_update_ratio(before, parameters)

        model.eval()
        with torch.no_grad():
            eval_output = model(x)
            eval_loss = compute_playground_loss(criterion, eval_output, target, effective_loss)
            eval_accuracy = compute_playground_accuracy(eval_output, target, effective_loss)

        epoch_values.append(epoch)
        losses.append(float(eval_loss.item()))
        accuracies.append(eval_accuracy)
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        update_ratios.append(ratio)
        for label in tracked_grad_labels:
            grad_history[label].append(float(grad_snapshot.get(label, 0.0)))

    cnn_maps = extract_cnn_feature_maps(model.last_cnn_feature_maps)
    attention_heatmap = extract_attention_heatmap(model.last_attention_weights)
    attention_is_simulated = False
    components = {layer["component"] for layer in layers}
    if attention_heatmap is None and components & {"MultiheadAttention", "TransformerEncoder"}:
        attention_heatmap = make_similarity_attention_heatmap(x)
        attention_is_simulated = attention_heatmap is not None

    token_count = int(attention_heatmap.shape[0]) if attention_heatmap is not None else 0
    mode_notes = [
        "真实轻量训练：这里确实执行了 PyTorch forward、backward 和 optimizer.step；数据是为了浏览器课堂即时反馈生成的小批量，不等价于正式数据集训练。",
        "损失曲线、梯度流和参数更新幅度来自真实张量计算；它们可以用来判断结构是否能跑、梯度是否能传、更新是否过大或过小。",
    ]
    mode_notes.extend(target_notes)
    if cnn_maps is not None:
        mode_notes.append("CNN 特征图来自当前结构中第一层 Conv2d 的真实激活。")
    if attention_heatmap is not None and attention_is_simulated:
        mode_notes.append("教学模拟：当前 Transformer Encoder 不直接暴露内部注意力权重，所以热力图用 token 相似度近似展示“谁更像会看向谁”。")
    elif attention_heatmap is not None:
        mode_notes.append("注意力热力图来自 MultiheadAttention 的真实权重，并对多个 head 做了平均。")

    return PlaygroundTrainingHistory(
        epochs=epoch_values,
        losses=losses,
        accuracies=accuracies,
        learning_rates=learning_rates,
        grad_norms=grad_history,
        update_ratios=update_ratios,
        cnn_feature_maps=cnn_maps,
        attention_heatmap=attention_heatmap,
        attention_tokens=attention_token_labels(token_count),
        attention_is_simulated=attention_is_simulated,
        mode_notes=mode_notes,
        effective_loss=effective_loss,
    )


def make_loss_curve(history: PlaygroundTrainingHistory) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.epochs,
            y=history.losses,
            mode="lines+markers",
            line={"color": "#0f8b8d", "width": 3},
            marker={"size": 7},
            name="loss",
        )
    )
    fig.update_layout(
        title="损失曲线：真实轻量训练是否正在变好",
        height=320,
        margin={"l": 42, "r": 18, "t": 52, "b": 38},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.84)",
        font=PLOT_FONT,
        xaxis_title="Epoch",
        yaxis_title="Loss",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e1e7ec")
    fig.update_yaxes(showgrid=True, gridcolor="#e1e7ec")
    return fig


def make_gradient_flow_chart(history: PlaygroundTrainingHistory) -> go.Figure:
    fig = go.Figure()
    colors = ["#0f8b8d", "#3268a8", "#bf3f5b", "#c4871f", "#46535d", "#6f5da8"]
    for index, (layer, values) in enumerate(history.grad_norms.items()):
        fig.add_trace(
            go.Scatter(
                x=history.epochs,
                y=values,
                mode="lines+markers",
                line={"color": colors[index % len(colors)], "width": 2.4},
                marker={"size": 5},
                name=layer,
            )
        )
    fig.update_layout(
        title="梯度流：每一层到底有没有收到学习信号",
        height=360,
        margin={"l": 42, "r": 18, "t": 52, "b": 68},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.84)",
        font=PLOT_FONT,
        xaxis_title="Epoch",
        yaxis_title="Gradient L2 norm",
        legend={"orientation": "h", "y": -0.28},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e1e7ec")
    fig.update_yaxes(showgrid=True, gridcolor="#e1e7ec")
    return fig


def make_update_ratio_chart(history: PlaygroundTrainingHistory) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=history.epochs,
            y=[value * 100 for value in history.update_ratios],
            marker_color="#bf3f5b",
            name="参数更新幅度比",
        )
    )
    fig.update_layout(
        title="参数更新动画：每轮参数移动了多少",
        height=320,
        margin={"l": 42, "r": 18, "t": 52, "b": 38},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.84)",
        font=PLOT_FONT,
        xaxis_title="Epoch",
        yaxis_title="更新 / 参数范数（%）",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e1e7ec")
    fig.update_yaxes(showgrid=True, gridcolor="#e1e7ec")
    return fig


def make_cnn_feature_map(feature_maps: np.ndarray) -> go.Figure:
    channel_count = int(feature_maps.shape[0])
    cols = min(3, channel_count)
    rows = int(math.ceil(channel_count / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"通道 {index}" for index in range(channel_count)],
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )
    for index, channel in enumerate(feature_maps):
        row = index // cols + 1
        col = index % cols + 1
        fig.add_trace(
            go.Heatmap(z=channel, colorscale="Viridis", showscale=index == 0),
            row=row,
            col=col,
        )
    fig.update_layout(
        title="CNN 特征图：卷积层把输入图像改写成哪些响应",
        height=max(300, 210 * rows),
        margin={"l": 28, "r": 20, "t": 72, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def make_attention_heatmap(history: PlaygroundTrainingHistory) -> go.Figure:
    heatmap = history.attention_heatmap
    if heatmap is None:
        raise ValueError("缺少注意力热力图。")
    tokens = history.attention_tokens or attention_token_labels(int(heatmap.shape[0]))
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=heatmap,
                x=tokens,
                y=tokens,
                colorscale="YlGnBu",
                colorbar={"title": "权重"},
            )
        ]
    )
    title = "注意力热力图：query token 正在看向谁"
    if history.attention_is_simulated:
        title += "（教学模拟）"
    fig.update_layout(
        title=title,
        height=420,
        margin={"l": 78, "r": 22, "t": 58, "b": 78},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.84)",
        font=PLOT_FONT,
        xaxis_title="被看的 key token",
        yaxis_title="当前 query token",
    )
    return fig


def render_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #fbfcfb 0%, #eef5f2 100%);
            color: #172026;
        }
        .block-container { padding-top: 1.2rem; padding-bottom: 2.4rem; }
        .playground-hero {
            border-bottom: 1px solid #d7dde1;
            padding-bottom: 0.9rem;
            margin-bottom: 1rem;
        }
        .playground-hero h1 {
            margin: 0;
            font-size: clamp(2rem, 3vw, 3rem);
            letter-spacing: 0;
        }
        .playground-hero p {
            color: #58646d;
            max-width: 980px;
            line-height: 1.7;
            margin: 0.45rem 0 0;
        }
        .mini-note {
            border-left: 4px solid #0f8b8d;
            background: rgba(255,255,255,0.78);
            border-radius: 0 8px 8px 0;
            padding: 0.7rem 0.9rem;
            line-height: 1.65;
            color: #26343b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def project_config() -> dict[str, Any]:
    return {
        "version": 2,
        "name": st.session_state.playground_project_name,
        "input_shape": st.session_state.playground_input_shape,
        "loss": st.session_state.playground_loss,
        "optimizer": st.session_state.playground_optimizer,
        "optimizer_params": copy.deepcopy(st.session_state.playground_optimizer_params),
        "layers": clone_layers(st.session_state.playground_layers),
    }


def export_project_config() -> str:
    """Serialize the current playground model to a stable JSON document."""

    return json.dumps(project_config(), ensure_ascii=False, indent=2)


def import_project_config(raw_json: str) -> None:
    """Load a saved playground model config after validating public fields."""

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 无法解析：{exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("配置文件必须是 JSON object。")
    layers = payload.get("layers")
    if not isinstance(layers, list):
        raise ValueError("配置缺少 layers 数组。")
    normalized_layers: list[dict[str, Any]] = []
    for index, layer in enumerate(layers, 1):
        if not isinstance(layer, dict):
            raise ValueError(f"第 {index} 层不是 object。")
        component = layer.get("component")
        if component not in COMPONENT_REGISTRY:
            raise ValueError(f"第 {index} 层组件 {component!r} 不在注册表中。")
        params = layer.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"第 {index} 层 params 必须是 object。")
        defaults = COMPONENT_REGISTRY[component]["params"]
        merged = copy.deepcopy(defaults)
        for key, value in params.items():
            if key in defaults:
                merged[key] = value
        normalized_layers.append({"component": component, "params": merged})

    shape_text = str(payload.get("input_shape", "(1, 28, 28)"))
    parse_shape(shape_text)
    loss = str(payload.get("loss", "CrossEntropy"))
    optimizer = str(payload.get("optimizer", "Adam"))
    if loss not in LOSS_REGISTRY:
        raise ValueError(f"未知损失函数：{loss}")
    if optimizer not in OPTIMIZER_REGISTRY:
        raise ValueError(f"未知优化器：{optimizer}")

    st.session_state.playground_project_name = str(payload.get("name", "导入的神经网络结构"))[:80]
    st.session_state.playground_input_shape = shape_text
    st.session_state.playground_layers = normalized_layers
    st.session_state.playground_loss = loss
    st.session_state.playground_optimizer = optimizer
    optimizer_params = payload.get("optimizer_params")
    if isinstance(optimizer_params, dict):
        current = copy.deepcopy(st.session_state.playground_optimizer_params)
        for name, params in optimizer_params.items():
            if name in current and isinstance(params, dict):
                current[name].update(params)
        st.session_state.playground_optimizer_params = current
    st.session_state.playground_loaded_example = ""


def render_project_io() -> None:
    st.subheader("保存 / 加载结构")
    st.session_state.playground_project_name = st.text_input(
        "结构名称",
        value=st.session_state.playground_project_name,
        max_chars=80,
    )
    json_text = export_project_config()
    st.download_button(
        "下载模型结构 JSON",
        data=json_text.encode("utf-8"),
        file_name="neural_network_playground_config.json",
        mime="application/json",
        width="stretch",
    )
    with st.expander("查看 / 粘贴 JSON", expanded=False):
        edited = st.text_area("模型结构 JSON", value=json_text, height=260, key="playground_config_json")
        if st.button("从上方 JSON 加载", width="stretch"):
            try:
                import_project_config(edited)
                st.session_state.playground_last_import_error = ""
                st.success("模型结构已加载。")
                _rerun()
            except Exception as exc:
                st.session_state.playground_last_import_error = str(exc)
                st.error(f"加载失败：{exc}")
    uploaded = st.file_uploader("上传结构 JSON", type=["json"], accept_multiple_files=False)
    if uploaded is not None and st.button("加载上传文件", width="stretch"):
        try:
            import_project_config(uploaded.getvalue().decode("utf-8"))
            st.session_state.playground_last_import_error = ""
            st.success("上传结构已加载。")
            _rerun()
        except Exception as exc:
            st.session_state.playground_last_import_error = str(exc)
            st.error(f"上传文件加载失败：{exc}")


def module_url(target: str, **params: str) -> str:
    query = {"module": target}
    query.update(params)
    parts = [f"{quote(str(key), safe='')}={quote(str(value), safe='')}" for key, value in query.items()]
    return "/?" + "&".join(parts)


def render_integration_panel(input_shape: tuple[int, ...] | None, steps: list[ShapeStep]) -> None:
    final_shape = steps[-1].output_shape if steps and steps[-1].ok else None
    st.subheader("训练与章节联动")
    st.markdown(
        """
        <div class="mini-note">
        当前控制台已经能输出结构、shape 和代码。下一步要理解训练行为时，请顺着下面几个入口看：
        损失曲线回答“有没有学会”，梯度监控回答“为什么学不动”，特征图/注意力回答“中间表示在看什么”。
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].link_button("训练过程可视化", module_url("part6_universal_framework/training_demo"), width="stretch")
    cols[1].link_button("梯度监控", module_url("part5_toolbox/02_gradient_monitor"), width="stretch")
    if final_shape and len(final_shape) == 3:
        cols[2].link_button("CNN 特征图", module_url("part2_cnn/02_feature_maps"), width="stretch")
    else:
        cols[2].link_button("训练动态", module_url("part5_toolbox/03_training_dynamics"), width="stretch")
    if final_shape and len(final_shape) == 2:
        cols[3].link_button("注意力机制", module_url("part4_transformer/transformer_models"), width="stretch")
    else:
        cols[3].link_button("超参搜索", module_url("part5_toolbox/04_hyperparam_search"), width="stretch")

    if input_shape and final_shape:
        st.caption(f"当前结构：输入 {format_shape(input_shape)} → 输出 {format_shape(final_shape)}。")


def render_linked_training_panel(
    input_shape: tuple[int, ...] | None,
    layers: list[dict[str, Any]],
    steps: list[ShapeStep],
    loss_name: str,
    optimizer_name: str,
    optimizer_params: dict[str, Any],
) -> None:
    st.subheader("联动训练演示")
    st.markdown(
        """
        <div class="mini-note">
        这里把中央控制台搭出来的结构真正跑一小段训练：损失曲线回答“有没有学到东西”，
        梯度流回答“学习信号有没有传到每一层”，参数更新动画回答“每轮参数到底动了多少”，
        CNN 特征图和注意力热力图则把中间表示拉回到对应章节里看。
        </div>
        """,
        unsafe_allow_html=True,
    )
    if input_shape is None or not steps or any(not step.ok for step in steps):
        st.info("请先修复上方 shape 推导错误；只有结构能真实前向传播，联动训练才会启动。")
        return

    default_lr = float(optimizer_params.get("lr", 0.001))
    left, middle, right = st.columns(3)
    epochs = int(left.slider("联动训练 epoch", min_value=1, max_value=20, value=6, step=1))
    learning_rate = float(
        middle.number_input(
            "联动训练学习率",
            min_value=0.000001,
            max_value=1.0,
            value=max(default_lr, 0.000001),
            step=0.0005,
            format="%.6f",
        )
    )
    seed = int(right.number_input("联动训练随机种子", min_value=0, max_value=9999, value=7, step=1))

    config_key = json.dumps(
        {
            "input_shape": input_shape,
            "layers": layers,
            "loss": loss_name,
            "optimizer": optimizer_name,
            "optimizer_params": optimizer_params,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    if st.button("运行联动演示", width="stretch"):
        try:
            with st.spinner("正在执行真实轻量训练，并收集损失、梯度、特征图和注意力权重..."):
                history = run_playground_training(
                    input_shape,
                    layers,
                    steps,
                    loss_name,
                    optimizer_name,
                    optimizer_params,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    seed=seed,
                )
            st.session_state.playground_training_history = history
            st.session_state.playground_training_key = config_key
            st.success("联动训练完成。下面的图表已经和当前结构对齐。")
        except Exception as exc:
            st.session_state.playground_training_history = None
            st.session_state.playground_training_key = ""
            st.error(f"联动训练没有跑通：{exc}")
            with st.expander("错误详情", expanded=False):
                st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), language="text")

    history = st.session_state.get("playground_training_history")
    if st.session_state.get("playground_training_key") != config_key:
        history = None
    if history is None:
        st.caption("点击“运行联动演示”后，这里会出现损失、梯度流、参数更新、CNN 特征图和注意力热力图。")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("训练类型", "真实轻量训练")
    metric_cols[1].metric("可视化补充", "教学模拟" if history.attention_is_simulated else "真实权重/激活")
    metric_cols[2].metric("损失函数", history.effective_loss)
    metric_cols[3].metric("最终损失", f"{history.losses[-1]:.4f}")
    for note in history.mode_notes:
        st.caption(f"• {note}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(make_loss_curve(history), use_container_width=True, config=PLOT_CONFIG)
        st.markdown(
            "> 请观察损失曲线是否整体下降；如果它完全不动，通常说明结构无法表达目标、学习率太小，或某一层把梯度截断了。"
        )
    with chart_right:
        st.plotly_chart(make_update_ratio_chart(history), use_container_width=True, config=PLOT_CONFIG)
        frame = st.slider(
            "参数更新动画帧",
            min_value=1,
            max_value=len(history.epochs),
            value=len(history.epochs),
            step=1,
        )
        ratio = history.update_ratios[frame - 1]
        st.progress(min(100, max(0, int(ratio * 20000))))
        st.markdown(
            f"> 第 {frame} 轮的真实参数更新幅度比是 **{ratio * 100:.4f}%**。如果这条柱长期接近 0，模型几乎没动；如果突然很高，学习率可能过激。"
        )

    st.plotly_chart(make_gradient_flow_chart(history), use_container_width=True, config=PLOT_CONFIG)
    st.markdown(
        "> 梯度流里每条线对应一层。前面层长期接近 0 是梯度消失的信号；某一层突然暴涨，则要检查学习率、归一化和残差连接。"
    )

    visual_left, visual_right = st.columns(2)
    with visual_left:
        if history.cnn_feature_maps is not None:
            st.plotly_chart(make_cnn_feature_map(history.cnn_feature_maps), use_container_width=True, config=PLOT_CONFIG)
            st.markdown(
                "> CNN 特征图的亮区表示该卷积核强烈响应的位置。请切换到 CNN 预设再运行，观察浅层卷积通常更像边缘、纹理和方向探测器。"
            )
        else:
            st.info("当前结构没有 Conv2d 层，所以不会生成 CNN 特征图。加载 CNN 预设后，这里会显示第一层卷积的真实激活。")
    with visual_right:
        if history.attention_heatmap is not None:
            st.plotly_chart(make_attention_heatmap(history), use_container_width=True, config=PLOT_CONFIG)
            st.markdown(
                "> 热力图每一行是一个 query token，每一列是被看的 key token。颜色越深，表示这一行 token 在汇聚信息时越依赖那一列 token。"
            )
        else:
            st.info("当前结构没有注意力或 Transformer 组件，所以不会生成注意力热力图。加载 Transformer 预设后再运行即可观察。")


def render_component_docs() -> None:
    rows = [
        {
            "组件": item["name"],
            "类型": item["type"],
            "默认参数": ", ".join(f"{key}={value}" for key, value in item["params"].items()) or "-",
            "输入规则": item["input_rule"],
            "输出规则": item["output_rule"],
            "相关主题": item["related_topic"],
        }
        for item in COMPONENT_REGISTRY.values()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_layer_form() -> None:
    with st.form("add-layer-form", clear_on_submit=False):
        left, right = st.columns([0.9, 1.4])
        with left:
            component = st.selectbox("组件类型", list(COMPONENT_REGISTRY), index=0)
            info = COMPONENT_REGISTRY[component]
            st.caption(info["description"])
            st.caption(f"PyTorch: `{info['pytorch_code']}`")
        with right:
            params: dict[str, Any] = {}
            defaults = COMPONENT_REGISTRY[component]["params"]
            if not defaults:
                st.info("该组件没有可配置参数。")
            for name, default in defaults.items():
                key = f"new-{component}-{name}"
                if isinstance(default, bool):
                    params[name] = st.checkbox(name, value=default, key=key)
                elif isinstance(default, int):
                    if component == "Flatten" and name == "end_dim":
                        min_value = -1
                    elif name == "padding":
                        min_value = 0
                    else:
                        min_value = 1
                    params[name] = int(st.number_input(name, min_value=min_value, value=default, step=1, key=key))
                elif isinstance(default, float):
                    max_value = 0.9 if name in {"p", "dropout"} else None
                    step = 0.01 if name not in {"eps"} else 1e-6
                    params[name] = float(
                        st.number_input(
                            name,
                            min_value=0.0,
                            max_value=max_value,
                            value=default,
                            step=step,
                            format="%.8f",
                            key=key,
                        )
                    )
                else:
                    params[name] = st.text_input(name, value=str(default), key=key)
        submitted = st.form_submit_button("添加层", width="stretch")
        if submitted:
            st.session_state.playground_layers.append({"component": component, "params": params})
            st.success(f"已添加 {component}")
            _rerun()


def render_layer_list() -> None:
    layers = st.session_state.playground_layers
    if not layers:
        st.info("还没有添加任何层。可以先加载一个预设，或从上方表单添加。")
        return

    for index, layer in enumerate(layers):
        params = layer.get("params", {})
        params_text = ", ".join(f"{name}={value}" for name, value in params.items()) or "无参数"
        cols = st.columns([0.14, 0.52, 0.22, 0.12])
        cols[0].markdown(f"**{index + 1}**")
        cols[1].markdown(f"**{layer['component']}**  `{params_text}`")
        cols[2].caption(COMPONENT_REGISTRY[layer["component"]]["related_topic"])
        if cols[3].button("删除", key=f"delete-layer-{index}", width="stretch"):
            del st.session_state.playground_layers[index]
            _rerun()


def render_optimizer_controls() -> tuple[str, dict[str, Any]]:
    loss = st.selectbox(
        "损失函数",
        list(LOSS_REGISTRY),
        index=list(LOSS_REGISTRY).index(st.session_state.playground_loss),
    )
    st.session_state.playground_loss = loss

    optimizer = st.selectbox(
        "优化器",
        list(OPTIMIZER_REGISTRY),
        index=list(OPTIMIZER_REGISTRY).index(st.session_state.playground_optimizer),
    )
    st.session_state.playground_optimizer = optimizer

    params = st.session_state.playground_optimizer_params[optimizer]
    st.caption(OPTIMIZER_REGISTRY[optimizer]["description"])
    if optimizer == "SGD":
        params["lr"] = float(st.number_input("lr", min_value=0.0, value=float(params["lr"]), step=0.001, format="%.6f"))
        params["momentum"] = float(
            st.number_input("momentum", min_value=0.0, max_value=1.0, value=float(params["momentum"]), step=0.05)
        )
    elif optimizer == "Adam":
        params["lr"] = float(st.number_input("lr", min_value=0.0, value=float(params["lr"]), step=0.0005, format="%.6f"))
        beta1, beta2 = tuple(params["betas"])
        b1_col, b2_col = st.columns(2)
        params["betas"] = (
            float(b1_col.number_input("beta1", min_value=0.0, max_value=0.9999, value=float(beta1), step=0.01)),
            float(b2_col.number_input("beta2", min_value=0.0, max_value=0.9999, value=float(beta2), step=0.001, format="%.4f")),
        )
        params["eps"] = float(st.number_input("eps", min_value=0.0, value=float(params["eps"]), step=1e-8, format="%.10f"))
    return optimizer, params


def render_shape_table(steps: list[ShapeStep]) -> None:
    if not steps:
        st.info("添加层后会在这里显示逐层形状推导。")
        return
    rows = [
        {
            "#": step.index,
            "层": step.layer,
            "输入形状": format_shape(step.input_shape),
            "输出形状": format_shape(step.output_shape),
            "状态": "通过" if step.ok else "错误",
            "说明": step.message,
        }
        for step in steps
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    failed = next((step for step in steps if not step.ok), None)
    if failed:
        st.error(f"第 {failed.index} 层连接非法：{failed.message}")
    else:
        st.success(f"形状推导通过，最终输出形状为 {format_shape(steps[-1].output_shape)}。")


def render_preset_controls() -> None:
    st.subheader("示例模型")
    preset_keys = ("mlp", "cnn", "transformer", "residual_mlp")
    cols = st.columns(len(preset_keys))
    for col, key in zip(cols, preset_keys, strict=True):
        preset = PRESETS[key]
        if col.button(preset["title"], key=f"load-{key}", width="stretch"):
            load_preset(key)
            st.query_params["module"] = PLAYGROUND_TARGET
            st.query_params["example"] = key
            _rerun()
    current = st.session_state.playground_loaded_example
    if current in PRESETS and PRESETS[current].get("note"):
        st.info(PRESETS[current]["note"])


def render_app() -> None:
    st.set_page_config(page_title="神经网络乐高工厂", layout="wide", initial_sidebar_state="expanded")
    render_style()
    default_state()
    ensure_query_preset_loaded()

    st.markdown(
        """
        <div class="playground-hero">
          <h1>神经网络乐高工厂 / 中央控制台</h1>
          <p>用基础组件搭网络，实时检查形状连接，并生成可运行的 PyTorch 模型代码。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([0.8, 0.2])
    with top_left:
        render_preset_controls()
    with top_right:
        st.write("")
        st.write("")
        if st.button("返回主界面", width="stretch"):
            go_home()

    builder_col, result_col = st.columns([0.95, 1.05], gap="large")

    with builder_col:
        st.subheader("模型构建器")
        input_shape_text = st.text_input(
            "输入形状（不含 batch 维）",
            value=st.session_state.playground_input_shape,
            help="例如 MNIST 图像为 (1, 28, 28)，展平后的向量为 (784,)，Transformer token 表示可写 (16, 64)。",
        )
        st.session_state.playground_input_shape = input_shape_text
        render_layer_form()
        st.divider()
        st.subheader("已添加层")
        render_layer_list()
        if st.button("清空层", width="stretch", disabled=not st.session_state.playground_layers):
            st.session_state.playground_layers = []
            st.session_state.playground_loaded_example = ""
            _rerun()
        st.divider()
        render_project_io()

    with result_col:
        st.subheader("训练配置")
        optimizer_name, optimizer_params = render_optimizer_controls()

        st.divider()
        st.subheader("形状推导")
        try:
            input_shape = parse_shape(st.session_state.playground_input_shape)
            steps = infer_shapes(input_shape, st.session_state.playground_layers)
            render_shape_table(steps)
        except Exception as exc:
            input_shape = (1, 28, 28)
            steps = []
            st.error(f"输入形状无法解析：{exc}")

        st.divider()
        st.subheader("PyTorch 代码")
        if steps and all(step.ok for step in steps):
            code = generate_code(
                input_shape,
                st.session_state.playground_layers,
                steps,
                st.session_state.playground_loss,
                optimizer_name,
                optimizer_params,
            )
            st.code(code, language="python")
        else:
            st.warning("修复形状错误后会生成完整 PyTorch 代码。")

        st.divider()
        render_integration_panel(input_shape if steps else None, steps)
        render_linked_training_panel(
            input_shape if steps else None,
            st.session_state.playground_layers,
            steps,
            st.session_state.playground_loss,
            optimizer_name,
            optimizer_params,
        )

    with st.expander("组件注册表", expanded=False):
        render_component_docs()


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def _should_render_app() -> bool:
    return __name__ == "__main__" or _running_under_streamlit()


if _should_render_app():
    try:
        render_app()
    except Exception as error:
        try:
            st.error("中央控制台暂时无法运行。")
            st.warning("请返回主界面重新进入，或检查最近一次组件参数修改。")
            if st.button("返回主界面"):
                go_home()
            with st.expander("错误详情", expanded=False):
                st.code("".join(traceback.format_exception(type(error), error, error.__traceback__)), language="text")
        except Exception:
            raise
