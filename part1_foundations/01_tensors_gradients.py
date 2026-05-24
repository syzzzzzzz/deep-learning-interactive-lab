MODULE_TITLE = "张量与梯度"
MODULE_SUMMARY = "用可视化理解张量形状、自动求导和反向传播。"
MODULE_TAGS = ["基础", "张量", "梯度", "PyTorch"]

import io
import sys
import traceback
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


def tensor_basics_demo() -> dict[str, object]:
    print("=" * 60)
    print("张量基础演示")
    print("=" * 60)
    torch.manual_seed(42)
    scalar = torch.tensor(42.0)
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    image = torch.randn(3, 4, 4)
    batch = torch.randn(32, 3, 224, 224)
    print(f"标量: {scalar}, shape={scalar.shape}, ndim={scalar.ndim}")
    print(f"向量: shape={vector.shape}, ndim={vector.ndim}")
    print(f"矩阵: shape={matrix.shape}, ndim={matrix.ndim}")
    print(f"图像: shape={image.shape}, ndim={image.ndim}")
    print(f"批次: shape={batch.shape}, ndim={batch.ndim}")

    print("\n--- 张量运算 ---")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a @ b = {a @ b}")
    print(f"a.sum() = {a.sum()}")
    print(f"a.mean() = {a.mean()}")
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    C = A @ B
    print(f"\n矩阵乘法: ({A.shape}) @ ({B.shape}) = {C.shape}")
    x = torch.ones(3, 1)
    y = torch.ones(1, 4)
    z = x + y
    print("\n--- 广播机制 ---")
    print(f"广播: {x.shape} + {y.shape} = {z.shape}")
    return {
        "shapes": [
            ("标量", tuple(scalar.shape), scalar.ndim),
            ("向量", tuple(vector.shape), vector.ndim),
            ("矩阵", tuple(matrix.shape), matrix.ndim),
            ("图像", tuple(image.shape), image.ndim),
            ("批次", tuple(batch.shape), batch.ndim),
        ],
        "broadcast_shape": tuple(z.shape),
    }


def gradient_intuition() -> dict[str, list[float]]:
    print("=" * 60)
    print("梯度直观演示：找 f(x) = (x-3)^2 的最小值")
    print("=" * 60)
    x = torch.tensor(0.0, requires_grad=True)
    history: dict[str, list[float]] = {"x": [], "loss": [], "grad": []}
    lr = 0.1
    for step in range(50):
        loss = (x - 3) ** 2
        loss.backward()
        history["x"].append(float(x.item()))
        history["loss"].append(float(loss.item()))
        history["grad"].append(float(x.grad.item()))
        with torch.no_grad():
            x -= lr * x.grad
        x.grad.zero_()
        if step % 10 == 0:
            print(f"Step {step:3d}: x={x.item():.4f}, loss={loss.item():.4f}, grad={history['grad'][-1]:.4f}")
    print(f"\n最终结果: x = {x.item():.6f} (目标: 3.0)")
    return history


def plot_gradient_descent(history: dict[str, list[float]]) -> plt.Figure:
    with safe_mpl_figure(figsize=(15, 4)) as fig:
        axes = fig.subplots(1, 3)
        steps = range(len(history["x"]))
        axes[0].plot(steps, history["x"], "b-o", markersize=3)
        axes[0].axhline(y=3, color="r", linestyle="--", label="目标 x=3")
        axes[0].set_title("参数 x 的变化", fontsize=12)
        axes[0].set_xlabel("训练步数")
        axes[0].set_ylabel("x 的值")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(steps, history["loss"], "r-o", markersize=3)
        axes[1].set_title("损失函数的变化", fontsize=12)
        axes[1].set_xlabel("训练步数")
        axes[1].set_ylabel("Loss")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(steps, history["grad"], "g-o", markersize=3)
        axes[2].axhline(y=0, color="r", linestyle="--", label="梯度=0")
        axes[2].set_title("梯度的变化", fontsize=12)
        axes[2].set_xlabel("训练步数")
        axes[2].set_ylabel("梯度值")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


def plot_loss_surface(history: dict[str, list[float]]) -> plt.Figure:
    with safe_mpl_figure(figsize=(8, 5)) as fig:
        ax = fig.subplots()
        x_range = np.linspace(-1, 7, 200)
        y_range = (x_range - 3) ** 2
        ax.plot(x_range, y_range, "b-", linewidth=2, label="f(x) = (x-3)²")
        ax.scatter(history["x"], history["loss"], c=range(len(history["x"])), cmap="Reds", s=50, zorder=5, label="梯度下降轨迹")
        ax.set_title("梯度下降在损失曲面上的轨迹", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


class ManualNeuron:
    def __init__(self) -> None:
        self.w = 0.5
        self.b = 0.1
        self.x: float | None = None
        self.a: float | None = None

    def sigmoid(self, x: float) -> float:
        return float(1 / (1 + np.exp(-x)))

    def sigmoid_grad(self, x: float) -> float:
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, x: float) -> float:
        self.x = x
        self.a = self.w * x + self.b
        return self.sigmoid(self.a)

    def backward(self, dy: float) -> tuple[float, float, float]:
        if self.x is None or self.a is None:
            raise RuntimeError("forward must be called before backward")
        da = dy * self.sigmoid_grad(self.a)
        return da * self.w, da * self.x, da


def verify_backprop() -> dict[str, float]:
    print("=" * 60)
    print("验证手动反向传播 vs PyTorch 自动微分")
    print("=" * 60)
    x_val, w_val, b_val, target = 2.0, 0.5, 0.1, 0.8
    neuron = ManualNeuron()
    neuron.w = w_val
    neuron.b = b_val
    y_manual = neuron.forward(x_val)
    loss_manual = (y_manual - target) ** 2
    _, dw, db = neuron.backward(2 * (y_manual - target))
    print("手动实现:")
    print(f"  y = {y_manual:.6f}")
    print(f"  loss = {loss_manual:.6f}")
    print(f"  ∂L/∂w = {dw:.6f}")
    print(f"  ∂L/∂b = {db:.6f}")

    x = torch.tensor(x_val)
    w = torch.tensor(w_val, requires_grad=True)
    b = torch.tensor(b_val, requires_grad=True)
    t = torch.tensor(target)
    y = torch.sigmoid(w * x + b)
    loss = (y - t) ** 2
    loss.backward()
    print("\nPyTorch 自动微分:")
    print(f"  y = {y.item():.6f}")
    print(f"  loss = {loss.item():.6f}")
    print(f"  ∂L/∂w = {w.grad.item():.6f}")
    print(f"  ∂L/∂b = {b.grad.item():.6f}")
    print("\n差异:")
    print(f"  ∂L/∂w 差异: {abs(dw - w.grad.item()):.2e}")
    print(f"  ∂L/∂b 差异: {abs(db - b.grad.item()):.2e}")
    return {"manual_dw": float(dw), "manual_db": float(db), "torch_dw": float(w.grad.item()), "torch_db": float(b.grad.item())}


def visualize_computation_graph() -> dict[str, float]:
    print("\n计算图示意：")
    print(
        """
        x ──────────────────────────────────────────┐
                                                     ↓
        w ──→ [w*x] ──→ z ──→ [z+b] ──→ a ──→ [sigmoid] ──→ y ──→ [MSE] ──→ L
                                    ↑
        b ──────────────────────────┘

        反向传播方向与前向计算相反，链式法则把梯度逐节点传回参数。
        """
    )
    x = torch.tensor(2.0, requires_grad=True)
    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.1, requires_grad=True)
    L = (torch.sigmoid(w * x + b) - 0.8) ** 2
    L.backward()
    print("各节点的梯度：")
    print(f"  ∂L/∂w = {w.grad:.6f}")
    print(f"  ∂L/∂b = {b.grad:.6f}")
    print(f"  ∂L/∂x = {x.grad:.6f}")
    return {"dL_dw": float(w.grad), "dL_db": float(b.grad), "dL_dx": float(x.grad)}


class TinyNet(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layer2(self.activation(self.layer1(x))))


def train_xor() -> tuple[TinyNet, list[float], list[tuple[list[float], float, float]]]:
    torch.manual_seed(42)
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    model = TinyNet(hidden_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    criterion = nn.BCELoss()
    losses: list[float] = []
    for _ in range(5000):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    predictions: list[tuple[list[float], float, float]] = []
    with torch.no_grad():
        pred = model(X)
        print("XOR 预测结果：")
        for xi, yi, pi in zip(X, y, pred):
            predictions.append((xi.tolist(), float(yi.item()), float(pi.item())))
            mark = "✓" if abs(pi.item() - yi.item()) < 0.5 else "✗"
            print(f"  {xi.tolist()} → 真实:{yi.item():.0f}, 预测:{pi.item():.4f} ({mark})")
    return model, losses, predictions


def plot_xor_decision_boundary(model: TinyNet, losses: list[float]) -> plt.Figure:
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    with safe_mpl_figure(figsize=(12, 4)) as fig:
        axes = fig.subplots(1, 2)
        axes[0].plot(losses)
        axes[0].set_title("训练损失曲线", fontsize=12)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        with torch.no_grad():
            Z = model(grid).numpy().reshape(xx.shape)
        axes[1].contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.8)
        axes[1].contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
        colors = ["blue", "red", "red", "blue"]
        labels = ["(0,0)→0", "(0,1)→1", "(1,0)→1", "(1,1)→0"]
        for xi, color, label in zip(X, colors, labels):
            axes[1].scatter(xi[0], xi[1], c=color, s=200, zorder=5, label=label)
        axes[1].set_title("XOR 决策边界", fontsize=12)
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


class GradientFlowVisualizer:
    def __init__(self, model: nn.Module) -> None:
        self.grad_history: defaultdict[str, list[float]] = defaultdict(list)
        self._hooks = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._hooks.append(param.register_hook(lambda grad, n=name: self.grad_history[n].append(float(grad.abs().mean().item()))))

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()

    def diagnose(self) -> list[tuple[str, float, str]]:
        print("\n梯度诊断报告：")
        print("-" * 50)
        rows = []
        for name, grads in self.grad_history.items():
            if not grads:
                continue
            avg_grad = float(np.mean(grads[-10:]))
            status = "梯度消失" if avg_grad < 1e-6 else "梯度爆炸" if avg_grad > 100 else "正常"
            rows.append((name, avg_grad, status))
            print(f"  {name:30s}: {avg_grad:.2e}  {status}")
        return rows


class DeepNet(nn.Module):
    def __init__(self, depth: int = 5) -> None:
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(32, 32), nn.Sigmoid()])
        layers.append(nn.Linear(32, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def demo_gradient_flow() -> tuple[dict[str, list[float]], list[tuple[str, float, str]]]:
    torch.manual_seed(42)
    model = DeepNet(depth=5)
    visualizer = GradientFlowVisualizer(model)
    X = torch.randn(100, 32)
    y = torch.randn(100, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(50):
        pred = model(X)
        loss = nn.MSELoss()(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    diagnosis = visualizer.diagnose()
    grad_history = {name: list(values) for name, values in visualizer.grad_history.items()}
    visualizer.remove_hooks()
    return grad_history, diagnosis


def plot_gradient_flow(grad_history: dict[str, list[float]], title: str = "深层网络（Sigmoid激活）") -> plt.Figure:
    with safe_mpl_figure(figsize=(14, 5)) as fig:
        axes = fig.subplots(1, 2)
        names = list(grad_history.keys())
        final_grads = [grad_history[n][-1] for n in names]
        axes[0].barh(range(len(names)), final_grads, color="steelblue")
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels([n.replace(".", "\n") for n in names], fontsize=8)
        axes[0].set_xlabel("梯度绝对均值")
        axes[0].set_title(f"{title} - 各层梯度大小", fontsize=12)
        axes[0].axvline(x=1e-4, color="r", linestyle="--", label="消失阈值")
        axes[0].axvline(x=10, color="orange", linestyle="--", label="爆炸阈值")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        for name in names[:4]:
            axes[1].plot(grad_history[name], label=name, alpha=0.8)
        axes[1].set_xlabel("训练步数")
        axes[1].set_ylabel("梯度绝对均值")
        axes[1].set_title("梯度随训练的变化", fontsize=12)
        axes[1].legend(fontsize=8)
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


def compute_张量基础(save_artifacts: bool = False) -> dict[str, object]:
    """Compute tensor and gradient demos without Streamlit calls."""

    artifacts: list[Path] = []
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        tensor_info = tensor_basics_demo()
        history = gradient_intuition()
        grad_fig = plot_gradient_descent(history)
        surface_fig = plot_loss_surface(history)
        backprop = verify_backprop()
        graph_grads = visualize_computation_graph()
        xor_model, losses, predictions = train_xor()
        xor_fig = plot_xor_decision_boundary(xor_model, losses)
        grad_history, diagnosis = demo_gradient_flow()
        flow_fig = plot_gradient_flow(grad_history)
    figures = [
        ("gradient_descent.png", grad_fig),
        ("gradient_descent_surface.png", surface_fig),
        ("xor_decision_boundary.png", xor_fig),
        ("gradient_flow.png", flow_fig),
    ]
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    return {
        "log": log_buffer.getvalue(),
        "figures": figures,
        "artifacts": artifacts,
        "tensor_info": tensor_info,
        "history": history,
        "backprop": backprop,
        "graph_grads": graph_grads,
        "xor_predictions": predictions,
        "gradient_diagnosis": diagnosis,
    }


def render_张量基础() -> None:
    """Render the tensor and gradient lesson in Streamlit."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import (
        render_backprop_current_flow,
        render_gradient_descent_landscape,
        render_loading_bar,
        render_visual_system,
    )

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        render_visual_system("dark")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("梯度动画加载：损失地形和反向传播电流会把抽象公式变成可观察路径")
        c1, c2 = st.columns(2)
        with c1:
            render_gradient_descent_landscape()
        with c2:
            render_backprop_current_flow()
        data = compute_张量基础(save_artifacts=True)
        st.subheader("张量形状")
        st.dataframe([{"对象": name, "shape": str(shape), "ndim": ndim} for name, shape, ndim in data["tensor_info"]["shapes"]], width="stretch")
        st.subheader("手动反传校验")
        st.json(data["backprop"])
        st.subheader("XOR 预测")
        st.dataframe([{"输入": str(x), "真实": y, "预测": round(pred, 4)} for x, y, pred in data["xor_predictions"]], width="stretch")
        for title, (filename, fig) in zip(("梯度下降过程", "损失曲面轨迹", "XOR 决策边界", "梯度流动诊断"), data["figures"]):
            st.subheader(title)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"已保存产物：{get_artifact_path(filename)}")
        with st.expander("控制台讲解", expanded=False):
            st.code(str(data["log"])[-16000:], language="text")
    except Exception as exc:
        render_module_error("part1_foundations/01_tensors_gradients.py", exc)


render = render_张量基础


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_张量基础(save_artifacts=False)
    return bool(data["figures"]) and bool(data["tensor_info"])


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    _configure_stdio()
    if _running_under_streamlit():
        render_张量基础()
    else:
        try:
            result = compute_张量基础(save_artifacts=True)
            print(result["log"])
            for path in result["artifacts"]:
                print(f"图像已保存: {path}")
        except Exception as e:
            traceback.print_exception(e)
            raise SystemExit(1)
