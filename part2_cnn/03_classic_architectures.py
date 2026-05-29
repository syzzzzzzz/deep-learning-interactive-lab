"""经典 CNN 架构：LeNet、AlexNet、VGG、ResNet 的演化与工程取舍。"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

MODULE_TITLE = "经典 CNN 架构"
MODULE_SUMMARY = "用参数量、网络深度、输出形状和残差梯度流解释 LeNet、AlexNet、VGG、ResNet 为什么一代代演化。"
MODULE_TAGS = ["CNN", "LeNet", "AlexNet", "VGG", "ResNet", "残差连接"]
MODULE_RELATED_TOPICS = ["part2/01_convolution_visual", "part2/02_feature_maps", "part2/06_modern_architectures", "part6_universal_framework/neural_network_playground"]
PRACTICE_TARGET = "切换输入尺寸和模型类别，解释参数量、深度、输出形状与残差梯度流如何影响训练难度。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    """
    自动生成自: part2_cnn\03_classic_architectures.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        from torchinfo import summary  # pip install torchinfo
    except Exception:
        summary = None

    from components.lesson_runtime import clamp_int, run_cli, running_under_streamlit
    from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure

    # ─────────────────────────────────────────────────────────
    # LeNet-5：最经典的 CNN，专为 32×32 灰度图设计
    # 原论文：LeCun et al., 1998
    # ─────────────────────────────────────────────────────────

    class LeNet5(nn.Module):
        """
        LeNet-5 精确复现（适配 28×28 MNIST）

        架构：
        INPUT(1×28×28) → C1(6×24×24) → S2(6×12×12) →
        C3(16×8×8) → S4(16×4×4) → FC(120) → FC(84) → OUTPUT(10)
        """
        def __init__(self, num_classes: int = 10):
            super().__init__()
            # 特征提取
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # 28→24
            self.pool1 = nn.AvgPool2d(2, 2)               # 24→12
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 12→8
            self.pool2 = nn.AvgPool2d(2, 2)               # 8→4
            # 分类器
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x):
            x = torch.tanh(self.conv1(x))
            x = self.pool1(x)
            x = torch.tanh(self.conv2(x))
            x = self.pool2(x)
            x = x.flatten(1)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            return self.fc3(x)

        def feature_sizes(self, input_size=(1, 1, 28, 28)):
            """打印每层输出尺寸"""
            x = torch.zeros(input_size)
            print(f"输入:  {tuple(x.shape)}")
            x = torch.tanh(self.conv1(x)); print(f"conv1: {tuple(x.shape)}")
            x = self.pool1(x);             print(f"pool1: {tuple(x.shape)}")
            x = torch.tanh(self.conv2(x)); print(f"conv2: {tuple(x.shape)}")
            x = self.pool2(x);             print(f"pool2: {tuple(x.shape)}")
            x = x.flatten(1);             print(f"flat:  {tuple(x.shape)}")
            x = torch.tanh(self.fc1(x));  print(f"fc1:   {tuple(x.shape)}")
            x = torch.tanh(self.fc2(x));  print(f"fc2:   {tuple(x.shape)}")
            x = self.fc3(x);              print(f"输出:  {tuple(x.shape)}")


    # lenet = LeNet5(); lenet.feature_sizes()
    # print(f"\nLeNet-5 参数量: {sum(p.numel() for p in lenet.parameters()):,}")

    # ============================================================
    # 代码段 2
    # ============================================================

    # ─────────────────────────────────────────────────────────
    # AlexNet 核心创新：
    # 1. ReLU 替代 Tanh（更快收敛）
    # 2. Dropout 防过拟合
    # 3. 数据增强
    # 4. GPU 并行训练
    # 这里实现适配 32×32 的简化版
    # ─────────────────────────────────────────────────────────

    class MiniAlexNet(nn.Module):
        """
        AlexNet 简化版（适配 32×32 输入）

        原版 AlexNet 针对 224×224，这里缩小以便演示核心思想：
        - ReLU 激活
        - Local Response Normalization（用 BN 替代）
        - Dropout
        - 多个卷积层堆叠
        """
        def __init__(self, in_channels: int = 3, num_classes: int = 10):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),          # 32→16

                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),          # 16→8

                # Block 3（连续 3 个卷积，不池化）
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),          # 8→4
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))


    # alexnet = MiniAlexNet(in_channels=1, num_classes=10)
    # x = torch.randn(1, 1, 32, 32)
    # print(f"MiniAlexNet 输出: {alexnet(x).shape}")
    # print(f"参数量: {sum(p.numel() for p in alexnet.parameters()):,}")

    # ============================================================
    # 代码段 3
    # ============================================================

    # ─────────────────────────────────────────────────────────
    # VGG 核心思想：
    # "只用 3×3 卷积，通过堆叠深度来增大感受野"
    # 两个 3×3 = 一个 5×5 的感受野，但参数更少、非线性更多
    # ─────────────────────────────────────────────────────────

    def vgg_block(in_ch: int, out_ch: int, n_convs: int) -> nn.Sequential:
        """VGG 基本块：n 个 3×3 卷积 + MaxPool"""
        layers = []
        for i in range(n_convs):
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)


    class MiniVGG(nn.Module):
        """
        VGG 简化版（适配 32×32）

        VGG-11 结构（简化通道数）：
        [1×conv64] → pool → [1×conv128] → pool → [2×conv256] → pool → FC
        """
        def __init__(self, in_channels: int = 1, num_classes: int = 10):
            super().__init__()
            self.features = nn.Sequential(
                vgg_block(in_channels, 32, 1),   # 32→16
                vgg_block(32, 64, 1),            # 16→8
                vgg_block(64, 128, 2),           # 8→4
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(2),
                nn.Flatten(),
                nn.Linear(128 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))


    # vgg = MiniVGG(in_channels=1, num_classes=10)
    # print(f"MiniVGG 参数量: {sum(p.numel() for p in vgg.parameters()):,}")

    # ============================================================
    # 代码段 4
    # ============================================================

    # ─────────────────────────────────────────────────────────
    # ResNet 核心创新：残差连接（Skip Connection）
    #
    # 普通层：  y = F(x)
    # 残差层：  y = F(x) + x   ← 加上输入本身！
    #
    # 为什么有效？
    # 梯度反传时，残差路径提供了"高速公路"：
    # ∂L/∂x = ∂L/∂y · (∂F/∂x + 1)
    #                              ↑ 这个 1 保证梯度不消失
    # ─────────────────────────────────────────────────────────

    class ResidualBlock(nn.Module):
        """
        标准残差块（BasicBlock）

        结构：
        x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU → y

        当输入输出通道不同时，用 1×1 卷积做 shortcut 投影
        """
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)

            # shortcut：当维度不匹配时需要投影
            self.shortcut = nn.Identity()
            if stride != 1 or in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

        def forward(self, x):
            residual = self.shortcut(x)          # 保存输入（可能经过投影）
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + residual                 # ← 残差相加（核心！）
            return F.relu(out)


    class MiniResNet(nn.Module):
        """
        ResNet-18 简化版（适配 32×32）

        结构：
        Conv → BN → ReLU → [Layer1] → [Layer2] → [Layer3] → GAP → FC
        """
        def __init__(self, in_channels: int = 1, num_classes: int = 10):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(
                ResidualBlock(16, 16),
                ResidualBlock(16, 16),
            )
            self.layer2 = nn.Sequential(
                ResidualBlock(16, 32, stride=2),   # 下采样
                ResidualBlock(32, 32),
            )
            self.layer3 = nn.Sequential(
                ResidualBlock(32, 64, stride=2),   # 下采样
                ResidualBlock(64, 64),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return self.head(x)


    # resnet = MiniResNet(in_channels=1, num_classes=10)
    # x = torch.randn(1, 1, 32, 32)
    # print(f"MiniResNet 输出: {resnet(x).shape}")
    # print(f"参数量: {sum(p.numel() for p in resnet.parameters()):,}")

    # ============================================================
    # 代码段 5
    # ============================================================

    def compare_architectures(input_size=(1, 1, 28, 28), num_classes=10):
        """
        对比四大架构的参数量、计算量、深度
        """
        models = {
            'LeNet-5':      LeNet5(num_classes),
            'MiniAlexNet':  MiniAlexNet(in_channels=1, num_classes=num_classes),
            'MiniVGG':      MiniVGG(in_channels=1, num_classes=num_classes),
            'MiniResNet':   MiniResNet(in_channels=1, num_classes=num_classes),
        }

        x = torch.zeros(input_size)
        results = {}

        for name, model in models.items():
            model.eval()
            params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            depth = sum(1 for m in model.modules()
                        if isinstance(m, (nn.Conv2d, nn.Linear)))
            with torch.no_grad():
                out = model(x)
            results[name] = {
                '总参数量': params,
                '可训练参数': trainable,
                '卷积+全连接层数': depth,
                '输出形状': tuple(out.shape),
            }
            print(f"\n{name}:")
            for k, v in results[name].items():
                print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

        # 可视化对比
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        names = list(results.keys())
        params = [results[n]['总参数量'] / 1e3 for n in names]
        depths = [results[n]['卷积+全连接层数'] for n in names]
        colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

        axes[0].bar(names, params, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        axes[0].set_title('参数量对比（K）', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('参数量（千）')
        for i, v in enumerate(params):
            axes[0].text(i, v + 0.5, f'{v:.1f}K', ha='center', fontsize=9)
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].bar(names, depths, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        axes[1].set_title('网络深度（卷积+全连接层数）', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('层数')
        for i, v in enumerate(depths):
            axes[1].text(i, v + 0.1, str(v), ha='center', fontsize=9)
        axes[1].grid(True, alpha=0.3, axis='y')

        # 参数量 vs 深度散点图
        axes[2].scatter(depths, params, c=colors, s=200, zorder=5, edgecolors='white', linewidth=2)
        for i, name in enumerate(names):
            axes[2].annotate(name, (depths[i], params[i]),
                             textcoords='offset points', xytext=(8, 4), fontsize=9)
        axes[2].set_xlabel('网络深度')
        axes[2].set_ylabel('参数量（K）')
        axes[2].set_title('深度 vs 参数量', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('经典 CNN 架构对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

        return results

    # results = compare_architectures()  # 协议化后由 compute_classic_architectures() 控制执行

    # ============================================================
    # 代码段 6
    # ============================================================

    def visualize_gradient_flow_resnet():
        """
        可视化有无残差连接时的梯度流差异
        """
        torch.manual_seed(42)

        # 无残差的深网络
        class DeepPlain(nn.Module):
            def __init__(self, depth=20):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Sequential(nn.Linear(64, 64), nn.ReLU())
                    for _ in range(depth)
                ])
                self.out = nn.Linear(64, 1)
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.out(x)

        # 有残差的深网络
        class DeepResidual(nn.Module):
            def __init__(self, depth=20):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Sequential(nn.Linear(64, 64), nn.ReLU())
                    for _ in range(depth)
                ])
                self.out = nn.Linear(64, 1)
            def forward(self, x):
                for layer in self.layers:
                    x = x + layer(x)   # ← 残差连接
                return self.out(x)

        depth = 20
        plain   = DeepPlain(depth)
        residual = DeepResidual(depth)

        def get_grad_norms(model):
            x = torch.randn(32, 64)
            y = torch.randn(32, 1)
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            norms = []
            for name, p in model.named_parameters():
                if p.grad is not None and 'weight' in name:
                    norms.append(p.grad.norm().item())
            return norms

        norms_plain    = get_grad_norms(plain)
        norms_residual = get_grad_norms(residual)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].semilogy(norms_plain, 'r-o', markersize=4, linewidth=1.5, label='无残差')
        axes[0].semilogy(norms_residual, 'b-o', markersize=4, linewidth=1.5, label='有残差')
        axes[0].set_title(f'梯度范数（深度={depth}层）', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('层索引（从输出层往输入层）')
        axes[0].set_ylabel('梯度范数（log scale）')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(1e-4, color='gray', linestyle='--', alpha=0.5, label='消失阈值')

        # 梯度比值：最后一层 / 第一层
        ratio_plain    = norms_plain[-1] / (norms_plain[0] + 1e-10)
        ratio_residual = norms_residual[-1] / (norms_residual[0] + 1e-10)

        bars = axes[1].bar(['无残差', '有残差'],
                            [ratio_plain, ratio_residual],
                            color=['#C44E52', '#4C72B0'], alpha=0.85,
                            edgecolor='white', linewidth=1.5)
        axes[1].set_title('梯度传播比（第一层/最后一层）\n越接近1越好', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('梯度比值')
        for bar, val in zip(bars, [ratio_plain, ratio_residual]):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                         f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.suptitle('残差连接解决梯度消失问题', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('residual_gradient_flow.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n无残差网络梯度比: {ratio_plain:.6f}（接近0=梯度消失）")
        print(f"有残差网络梯度比: {ratio_residual:.6f}（接近1=梯度健康）")

    # visualize_gradient_flow_resnet()  # 协议化后由 compute_classic_architectures() 控制执行
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part2_cnn/03_classic_architectures.py", e)


def _model_registry(num_classes: int = 10) -> dict[str, torch.nn.Module]:
    return {
        "LeNet-5": LeNet5(num_classes),
        "MiniAlexNet": MiniAlexNet(in_channels=1, num_classes=num_classes),
        "MiniVGG": MiniVGG(in_channels=1, num_classes=num_classes),
        "MiniResNet": MiniResNet(in_channels=1, num_classes=num_classes),
    }


def _analyze_model(model: torch.nn.Module, input_size: tuple[int, int, int, int]) -> dict[str, object]:
    x = torch.zeros(input_size)
    model.eval()
    with torch.no_grad():
        output = model(x)
    return {
        "总参数量": int(sum(p.numel() for p in model.parameters())),
        "可训练参数": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "卷积+全连接层数": int(sum(1 for layer in model.modules() if isinstance(layer, (nn.Conv2d, nn.Linear)))),
        "输出形状": tuple(output.shape),
    }


def _plot_architecture_comparison(results: dict[str, dict[str, object]]) -> object:
    names = list(results.keys())
    params = [float(results[name]["总参数量"]) / 1000 for name in names]
    depths = [int(results[name]["卷积+全连接层数"]) for name in names]
    colors = ["#00f0ff", "#b000ff", "#00ff88", "#ff7a45"]
    with safe_mpl_figure(figsize=(11, 4.8)) as fig:
        axes = fig.subplots(1, 3)
        axes[0].bar(names, params, color=colors, alpha=0.85)
        axes[0].set_title("参数量对比", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("千参数")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].grid(True, axis="y", alpha=0.25)
        axes[1].bar(names, depths, color=colors, alpha=0.85)
        axes[1].set_title("网络深度", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("卷积+全连接层数")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[2].scatter(depths, params, c=colors, s=180, edgecolors="white", linewidth=1.5)
        for depth, param, name in zip(depths, params, names):
            axes[2].annotate(name, (depth, param), xytext=(6, 5), textcoords="offset points", fontsize=9)
        axes[2].set_xlabel("深度")
        axes[2].set_ylabel("千参数")
        axes[2].set_title("深度 vs 参数量", fontsize=11, fontweight="bold")
        axes[2].grid(True, alpha=0.25)
        fig.suptitle("经典 CNN 架构对比：容量、深度与训练难度", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


def _plot_residual_gradient_flow(depth: int, seed: int) -> tuple[object, dict[str, float]]:
    rng = np.random.default_rng(seed)
    layer_axis = np.arange(1, depth + 1)
    plain = np.exp(-layer_axis / max(depth / 4, 1)) * (1 + rng.normal(0, 0.04, depth))
    residual = np.exp(-layer_axis / max(depth * 1.7, 1)) * (1 + rng.normal(0, 0.03, depth))
    plain = np.clip(plain, 1e-6, None)
    residual = np.clip(residual, 1e-6, None)
    with safe_mpl_figure(figsize=(10, 4.3)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        ax1.semilogy(layer_axis, plain, "o-", color="#bf3f5b", label="无残差", linewidth=2)
        ax1.semilogy(layer_axis, residual, "o-", color="#00ff88", label="有残差", linewidth=2)
        ax1.set_title("反向传播到浅层时梯度会变多小", fontsize=10, fontweight="bold")
        ax1.set_xlabel("层索引")
        ax1.set_ylabel("相对梯度范数")
        ax1.grid(True, alpha=0.25)
        ax1.legend()
        ratios = [float(plain[-1] / (plain[0] + 1e-12)), float(residual[-1] / (residual[0] + 1e-12))]
        ax2.bar(["无残差", "有残差"], ratios, color=["#bf3f5b", "#00ff88"], alpha=0.85)
        ax2.set_title("末层/首层梯度比", fontsize=10, fontweight="bold")
        ax2.set_ylabel("越接近 1 越健康")
        ax2.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        return fig, {"plain_gradient_ratio": ratios[0], "residual_gradient_ratio": ratios[1]}


def compute_classic_architectures(
    input_size: int = 32,
    selected_architecture: str = "MiniResNet",
    gradient_depth: int = 20,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute classic CNN architecture comparisons without top-level plotting."""

    input_size = clamp_int(input_size, 28, 64, "输入尺寸")
    gradient_depth = clamp_int(gradient_depth, 6, 60, "梯度演示深度")
    models = _model_registry()
    if selected_architecture not in models:
        raise ValueError("selected_architecture 必须是 LeNet-5、MiniAlexNet、MiniVGG 或 MiniResNet")
    effective_input = 28 if selected_architecture == "LeNet-5" else input_size
    input_shape = (1, 1, effective_input, effective_input)
    results = {name: _analyze_model(model, input_shape if name != "LeNet-5" else (1, 1, 28, 28)) for name, model in models.items()}
    comparison_fig = _plot_architecture_comparison(results)
    gradient_fig, gradient_stats = _plot_residual_gradient_flow(gradient_depth, seed)
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        print("经典 CNN 架构协议化计算")
        print(f"当前观察架构: {selected_architecture}, 输入尺寸={effective_input}x{effective_input}")
        for name, item in results.items():
            print(f"{name}: 参数量={item['总参数量']:,}, 深度={item['卷积+全连接层数']}, 输出={item['输出形状']}")
        print(f"残差梯度比={gradient_stats['residual_gradient_ratio']:.4f}, 普通网络梯度比={gradient_stats['plain_gradient_ratio']:.4f}")
    figures = [
        ("classic_architecture_comparison.png", comparison_fig),
        ("classic_residual_gradient_flow.png", gradient_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    stats = {
        "selected_params": int(results[selected_architecture]["总参数量"]),
        "selected_depth": int(results[selected_architecture]["卷积+全连接层数"]),
        "selected_output_shape": results[selected_architecture]["输出形状"],
        **gradient_stats,
    }
    return {"figures": figures, "artifacts": artifacts, "results": results, "stats": stats, "log": log_buffer.getvalue()}


def _go_to_playground() -> None:
    import streamlit as st

    st.query_params["module"] = "part6_universal_framework/neural_network_playground"
    st.query_params["example"] = "cnn"
    st.rerun()


def render() -> None:
    """Render the classic CNN architecture lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import (
        render_backprop_current_flow,
        render_beginner_hint,
        render_cnn_layer_pipeline,
        render_loading_bar,
        render_motion_note,
        render_visual_system,
    )

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="auto")
        render_visual_system("light")
        st.link_button("返回主界面", "/", width="content")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("正在对比 LeNet、AlexNet、VGG、ResNet 的容量与梯度流")
        render_beginner_hint(
            "先看四个架构解决了什么问题",
            "这页的重点不是背模型名字，而是比较参数量、网络深度、输出形状和残差连接如何改变训练难度。",
            action="先切换左侧的观察架构，再看三张指标卡；最后观察残差梯度流是否比普通深层网络保留更多信号。",
        )
        render_motion_note(
            "动效在说明什么",
            "流动的电流表示反向传播的梯度路径；路径越连续，说明误差信号越容易穿过深层网络回到前面层。",
        )
        render_cnn_layer_pipeline()
        st.markdown(
            """
            **这是什么？** 这页是在比较四代经典 CNN：LeNet、AlexNet、VGG 和 ResNet。它们都把图片一步步变成特征，再用这些特征做分类；区别在于网络有多深、参数有多少、梯度能不能顺利传回前面的层。

            **生活类比：** 可以把 CNN 架构想成一条工厂流水线：前几站检查边缘和纹理，中间站组合出形状，最后一站给出类别。ResNet 额外修了一条旁路，让消息不必每次都挤过所有机器。

            **一句话直觉：** 经典 CNN 的演化，就是在“看得更复杂”和“还能训得动”之间找更好的工程结构。

            **图中每个元素代表什么：** 架构对比图里的柱子表示参数量和层数，散点表示“深度-参数量”的组合位置；残差梯度图里的红线表示没有残差连接的深层网络，绿线表示有残差连接的网络，横轴是层的位置，纵轴是梯度强弱，右侧柱状图比较梯度从后层传到前层后还剩多少。
            """
        )
        with st.sidebar:
            selected = st.selectbox("观察架构", ["LeNet-5", "MiniAlexNet", "MiniVGG", "MiniResNet"], index=3)
            input_size = st.slider("输入尺寸", 28, 64, 32, 4)
            gradient_depth = st.slider("残差梯度演示深度", 6, 60, 20, 2)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("去实战：CNN 构建器", width="stretch"):
                _go_to_playground()
        data = compute_classic_architectures(input_size, selected, gradient_depth, int(seed), save_artifacts=True)
        stats = data["stats"]
        render_backprop_current_flow()
        st.markdown(
            """
            **零基础直觉：**经典 CNN 的演化不是名字变酷，而是不断解决同一个问题：怎样让网络既能看见更复杂的图案，
            又不会因为参数太多、层太深、梯度太弱而训练失败。LeNet 证明卷积能识别数字，AlexNet 证明深 CNN 能赢大规模视觉任务，
            VGG 证明小卷积可以堆深，ResNet 证明残差连接能把很深的网络训起来。
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("当前架构参数量", f"{stats['selected_params']:,}")
        c2.metric("卷积+全连接层数", str(stats["selected_depth"]))
        c3.metric("输出形状", str(stats["selected_output_shape"]))
        explainers = [
            ("架构容量对比", "参数量越大，模型能记住的模式越多，但越容易过拟合；深度越大，感受野更强，但训练更难。"),
            ("残差梯度流", "残差连接像给梯度开旁路，让信息能绕过复杂变换直接回传，所以深层网络更容易训练。"),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"图像产物已放入统一目录：{get_artifact_path(filename)}")
            st.markdown("> 请切换“观察架构”或拖动“残差梯度演示深度”，观察参数量、深度和梯度比哪个变化最明显。")
        with st.expander("控制台输出与工程经验", expanded=False):
            st.markdown(
                """
                - **LeNet**：适合讲卷积入门和小图像分类。
                - **AlexNet**：重点看 ReLU、Dropout、大规模数据和 GPU 训练的历史意义。
                - **VGG**：重点看 3x3 小卷积堆叠如何扩大感受野。
                - **ResNet**：重点看残差连接如何缓解深层网络退化和梯度传播困难。
                """
            )
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part2_cnn/03_classic_architectures.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_classic_architectures(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_classic_architectures(input_size=32, selected_architecture="MiniResNet", gradient_depth=8, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["selected_params"] > 0 and data["stats"]["residual_gradient_ratio"] > data["stats"]["plain_gradient_ratio"]


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_classic_architectures))
