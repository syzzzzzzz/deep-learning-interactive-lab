"""Training and telemetry logic for the neural network playground."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # quality_check may run without heavy dependencies.
    np = torch = nn = None  # type: ignore[assignment]

from components.playground_core import ShapeStep
from components.playground_core import infer_layer_shape

PLAYGROUND_BATCH_SIZE = 16

try:
    torch.set_num_threads(1)
except (RuntimeError, AttributeError):
    pass

# When torch is unavailable, provide a tiny base class so imports stay safe.
if nn is None:
    class _ModuleStub:
        pass

    nn = type("nn", (), {"Module": _ModuleStub})()  # type: ignore[assignment]
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


class PlaygroundConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.last_feature_maps: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        self.last_feature_maps = output.detach()
        return output


class PlaygroundRecurrentBlock(nn.Module):
    def __init__(
        self,
        kind: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        return_sequences: bool = True,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _state = self.recurrent(x)
        if self.return_sequences:
            return output
        return output[:, -1, :]


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
            if (
                self.last_cnn_feature_maps is None
                and isinstance(module, PlaygroundConvBlock)
                and module.last_feature_maps is not None
            ):
                self.last_cnn_feature_maps = module.last_feature_maps
            if isinstance(module, PlaygroundSelfAttentionBlock) and module.last_attention_weights is not None:
                self.last_attention_weights = module.last_attention_weights
        return x
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
    if component == "ConvBlock":
        return PlaygroundConvBlock(
            int(params["in_channels"]),
            int(params["out_channels"]),
            kernel_size=int(params["kernel_size"]),
            stride=int(params["stride"]),
            padding=int(params["padding"]),
            use_batchnorm=bool(params.get("use_batchnorm", True)),
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
    if component in {"RNN", "LSTM", "GRU"}:
        return PlaygroundRecurrentBlock(
            component,
            int(params["input_size"]),
            int(params["hidden_size"]),
            num_layers=int(params["num_layers"]),
            bidirectional=bool(params.get("bidirectional", False)),
            dropout=float(params.get("dropout", 0.0)),
            return_sequences=bool(params.get("return_sequences", True)),
        )
    if component in {"Attention", "MultiheadAttention"}:
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
    if attention_heatmap is None and components & {"Attention", "MultiheadAttention", "TransformerEncoder"}:
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
