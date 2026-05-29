"""Pure model graph, shape inference, and codegen core for the neural network playground."""

from __future__ import annotations

import copy
import math
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


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
    "RNN": {
        "name": "RNN",
        "type": "sequence",
        "params": {
            "input_size": 64,
            "hidden_size": 128,
            "num_layers": 1,
            "bidirectional": False,
            "dropout": 0.0,
            "return_sequences": True,
        },
        "input_rule": "输入必须是 (seq_len, input_size)，不包含 batch 维。",
        "output_rule": "输出为 (seq_len, hidden_size * directions)，或只取末步时为 (hidden_size * directions,)。",
        "description": "基础循环神经网络层，用于逐步读取序列特征。",
        "pytorch_code": "nn.RNN(input_size, hidden_size, batch_first=True)",
        "related_topic": "RNN / 序列建模",
    },
    "LSTM": {
        "name": "LSTM",
        "type": "sequence",
        "params": {
            "input_size": 64,
            "hidden_size": 128,
            "num_layers": 1,
            "bidirectional": False,
            "dropout": 0.0,
            "return_sequences": True,
        },
        "input_rule": "输入必须是 (seq_len, input_size)，不包含 batch 维。",
        "output_rule": "输出为 (seq_len, hidden_size * directions)，或只取末步时为 (hidden_size * directions,)。",
        "description": "带门控记忆单元的循环层，适合更长依赖的序列表示。",
        "pytorch_code": "nn.LSTM(input_size, hidden_size, batch_first=True)",
        "related_topic": "LSTM / 序列建模",
    },
    "GRU": {
        "name": "GRU",
        "type": "sequence",
        "params": {
            "input_size": 64,
            "hidden_size": 128,
            "num_layers": 1,
            "bidirectional": False,
            "dropout": 0.0,
            "return_sequences": True,
        },
        "input_rule": "输入必须是 (seq_len, input_size)，不包含 batch 维。",
        "output_rule": "输出为 (seq_len, hidden_size * directions)，或只取末步时为 (hidden_size * directions,)。",
        "description": "门控循环单元，比 LSTM 更轻量，常用于序列分类和预测。",
        "pytorch_code": "nn.GRU(input_size, hidden_size, batch_first=True)",
        "related_topic": "GRU / 序列建模",
    },
    "ConvBlock": {
        "name": "ConvBlock",
        "type": "block",
        "params": {
            "in_channels": 1,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "use_batchnorm": True,
        },
        "input_rule": "输入必须是 (channels, height, width)，且 channels 等于 in_channels。",
        "output_rule": "输出为 (out_channels, H_out, W_out)。",
        "description": "Conv2d + 可选 BatchNorm2d + ReLU 的常用卷积块。",
        "pytorch_code": "ConvBlock(in_channels, out_channels, kernel_size, stride, padding)",
        "related_topic": "CNN / 卷积块",
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
    "Attention": {
        "name": "Attention",
        "type": "attention",
        "params": {"embed_dim": 64, "num_heads": 4, "dropout": 0.1},
        "input_rule": "输入必须是 (seq_len, embed_dim)，即 batch 后的 token 序列表示。",
        "output_rule": "输出形状与输入相同。",
        "description": "自注意力块，让每个 token 根据上下文重新汇聚信息。",
        "pytorch_code": "SelfAttentionBlock(embed_dim, num_heads, dropout)",
        "related_topic": "注意力 / Transformer",
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



def _positive_int_param(params: dict[str, Any], name: str) -> int:
    value = int(params[name])
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数，当前是 {value}。")
    return value


def _dropout_param(params: dict[str, Any], name: str = "dropout") -> float:
    value = float(params.get(name, 0.0))
    if not 0.0 <= value <= 0.9:
        raise ValueError(f"{name} 建议在 0.0 到 0.9 之间，当前是 {value}。")
    return value


def _conv2d_output_shape(component: str, params: dict[str, Any], current: tuple[int, ...]) -> tuple[int, int, int]:
    if len(current) != 3:
        raise ValueError(
            f"{component} 需要 (channels, height, width) 输入；当前上一层输出是 {format_shape(current)}。"
        )
    channels, height, width = current
    in_channels = _positive_int_param(params, "in_channels")
    if channels != in_channels:
        raise ValueError(
            f"{component} 的 in_channels={in_channels}，但上一层输出通道数是 {channels}。"
            "请修改 in_channels，或在前一层把通道数投影到匹配值。"
        )
    out_channels = _positive_int_param(params, "out_channels")
    kernel = _positive_int_param(params, "kernel_size")
    stride = _positive_int_param(params, "stride")
    padding = int(params["padding"])
    if padding < 0:
        raise ValueError(f"{component} 的 padding 不能为负数，当前是 {padding}。")
    h_out = math.floor((height + 2 * padding - kernel) / stride + 1)
    w_out = math.floor((width + 2 * padding - kernel) / stride + 1)
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"{component} 的 kernel_size={kernel}, stride={stride}, padding={padding} 会把 "
            f"{height}x{width} 变成 {h_out}x{w_out}。请调小 kernel_size/stride，或增大 padding。"
        )
    return out_channels, h_out, w_out


def _attention_shape(component: str, params: dict[str, Any], current: tuple[int, ...]) -> tuple[tuple[int, ...], str, str]:
    if len(current) != 2:
        raise ValueError(
            f"{component} 需要 (seq_len, embed_dim) 输入；当前上一层输出是 {format_shape(current)}。"
            "图像特征请先 reshape 成 token 序列，普通向量请先增加序列维。"
        )
    embed_dim = _positive_int_param(params, "embed_dim")
    num_heads = _positive_int_param(params, "num_heads")
    if current[-1] != embed_dim:
        raise ValueError(
            f"{component} 的 embed_dim={embed_dim} 必须等于上一层输出最后一维 {current[-1]}。"
            "可在前面接 Linear 对齐 token 宽度。"
        )
    if embed_dim % num_heads != 0:
        raise ValueError(f"{component} 要求 embed_dim={embed_dim} 能被 num_heads={num_heads} 整除。")
    _dropout_param(params)
    return current, "注意力重新混合 token 信息，序列长度和表示维度保持不变。", layer_code(component, params, current)


def _recurrent_shape(component: str, params: dict[str, Any], current: tuple[int, ...]) -> tuple[tuple[int, ...], str, str]:
    if len(current) != 2:
        raise ValueError(
            f"{component} 需要 (seq_len, input_size) 输入；当前上一层输出是 {format_shape(current)}。"
            "如果来自图像，请先把空间位置整理为序列；如果是单个向量，请增加 seq_len 维。"
        )
    seq_len, feature_dim = current
    input_size = _positive_int_param(params, "input_size")
    hidden_size = _positive_int_param(params, "hidden_size")
    num_layers = _positive_int_param(params, "num_layers")
    dropout = _dropout_param(params)
    if feature_dim != input_size:
        raise ValueError(
            f"{component} 的 input_size={input_size} 必须等于上一层输出最后一维 {feature_dim}。"
            "可先用 Linear 调整每个时间步的特征维。"
        )
    directions = 2 if bool(params.get("bidirectional", False)) else 1
    output_width = hidden_size * directions
    return_sequences = bool(params.get("return_sequences", True))
    if dropout > 0 and num_layers == 1:
        message = "num_layers=1 时 PyTorch 会忽略循环层内部 dropout；输出宽度由 hidden_size 和方向数决定。"
    else:
        message = "循环层按时间步更新隐藏状态；输出宽度由 hidden_size 和方向数决定。"
    output = (seq_len, output_width) if return_sequences else (output_width,)
    return output, message, layer_code(component, params, current)


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
        return _conv2d_output_shape(component, params, current), "按卷积公式计算空间尺寸。", layer_code(component, params, current)

    if component == "ConvBlock":
        return (
            _conv2d_output_shape(component, params, current),
            "Conv2d 改变通道和空间尺寸；BatchNorm/ReLU 不改变形状。",
            layer_code(component, params, current),
        )

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

    if component in {"RNN", "LSTM", "GRU"}:
        return _recurrent_shape(component, params, current)

    if component in {"Attention", "MultiheadAttention"}:
        return _attention_shape(component, params, current)

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
    if component == "ConvBlock":
        return (
            f"ConvBlock({params['in_channels']}, {params['out_channels']}, "
            f"kernel_size={params['kernel_size']}, stride={params['stride']}, padding={params['padding']}, "
            f"use_batchnorm={param_repr(params.get('use_batchnorm', True))})"
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
    if component in {"RNN", "LSTM", "GRU"}:
        return (
            f"RecurrentBlock({param_repr(component)}, {params['input_size']}, {params['hidden_size']}, "
            f"num_layers={params['num_layers']}, "
            f"bidirectional={param_repr(params.get('bidirectional', False))}, "
            f"dropout={params.get('dropout', 0.0)}, "
            f"return_sequences={param_repr(params.get('return_sequences', True))})"
        )
    if component in {"Attention", "MultiheadAttention"}:
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
    if "ConvBlock" in used_components:
        helper_blocks.append(
            """
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
"""
        )
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
    if used_components & {"RNN", "LSTM", "GRU"}:
        helper_blocks.append(
            """
class RecurrentBlock(nn.Module):
    def __init__(
        self,
        kind,
        input_size,
        hidden_size,
        num_layers=1,
        bidirectional=False,
        dropout=0.0,
        return_sequences=True,
    ):
        super().__init__()
        recurrent_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = recurrent_cls(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=effective_dropout,
        )
        self.return_sequences = return_sequences

    def forward(self, x):
        output, _state = self.recurrent(x)
        if self.return_sequences:
            return output
        return output[:, -1, :]
"""
        )
    if used_components & {"Attention", "MultiheadAttention"}:
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
