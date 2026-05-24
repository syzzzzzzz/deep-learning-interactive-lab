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

import pandas as pd
import streamlit as st


PLAYGROUND_TARGET = "part6_universal_framework/neural_network_playground"


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
