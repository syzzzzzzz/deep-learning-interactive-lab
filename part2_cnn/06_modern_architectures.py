"""现代 CNN 架构：MobileNet、DenseNet、EfficientNet 背后的效率设计。"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

MODULE_TITLE = "现代 CNN 架构"
MODULE_SUMMARY = "用深度可分离卷积、倒残差、密集连接和复合缩放解释现代 CNN 如何在精度、速度和参数量之间取舍。"
MODULE_TAGS = ["CNN", "MobileNet", "DenseNet", "EfficientNet", "深度可分离卷积", "复合缩放"]
MODULE_RELATED_TOPICS = ["part2/03_classic_architectures", "part2/07_advanced_convolution", "part6_universal_framework/neural_network_playground"]
PRACTICE_TARGET = "调整输入通道、输出通道、卷积核大小、缩放系数和宽度倍率，解释参数量、FLOPs 和精度趋势如何变化。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    """
    自动生成自: part2_cnn\06_modern_architectures.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
    from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


    class DepthwiseSeparableConv(nn.Module):
        """
        深度可分离卷积

        标准卷积: [C_out, C_in, K, K] → 参数 C_out × C_in × K²
        深度可分离: [C_in, 1, K, K] + [C_out, C_in, 1, 1]
        参数比: (C_in × K² + C_out × C_in) / (C_out × C_in × K²) = 1/K² + 1/C_out
        """
        def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
            super().__init__()
            # Depthwise: 每个输入通道独立卷积
            self.depthwise = nn.Conv2d(
                in_ch, in_ch, kernel_size,
                stride=stride, padding=kernel_size // 2,
                groups=in_ch, bias=False  # groups=in_ch 是关键
            )
            self.bn1 = nn.BatchNorm2d(in_ch)
            # Pointwise: 1×1 卷积混合通道
            self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)

        def forward(self, x):
            x = F.relu6(self.bn1(self.depthwise(x)))  # ReLU6 适合移动端
            x = F.relu6(self.bn2(self.pointwise(x)))
            return x


    class MobileNetV1(nn.Module):
        """
        MobileNet V1（适配 CIFAR-10）

        原版针对 224×224 ImageNet，这里缩小
        """
        def __init__(self, num_classes=10, width_mult=1.0):
            super().__init__()
            def round_channels(ch):
                """宽度乘子：缩小通道数"""
                return max(8, int(ch * width_mult + 0.5) // 8 * 8)

            self.features = nn.Sequential(
                # 标准卷积起步
                nn.Conv2d(3, round_channels(32), 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(round_channels(32)),
                nn.ReLU6(inplace=True),

                # 深度可分离卷积堆叠
                DepthwiseSeparableConv(round_channels(32), round_channels(64), stride=1),
                DepthwiseSeparableConv(round_channels(64), round_channels(128), stride=2),
                DepthwiseSeparableConv(round_channels(128), round_channels(128), stride=1),
                DepthwiseSeparableConv(round_channels(128), round_channels(256), stride=2),
                DepthwiseSeparableConv(round_channels(256), round_channels(256), stride=1),
                DepthwiseSeparableConv(round_channels(256), round_channels(512), stride=2),

                # 5 个 512 通道块
                DepthwiseSeparableConv(round_channels(512), round_channels(512), stride=1),
                DepthwiseSeparableConv(round_channels(512), round_channels(512), stride=1),
                DepthwiseSeparableConv(round_channels(512), round_channels(512), stride=1),
                DepthwiseSeparableConv(round_channels(512), round_channels(512), stride=1),
                DepthwiseSeparableConv(round_channels(512), round_channels(512), stride=1),

                DepthwiseSeparableConv(round_channels(512), round_channels(1024), stride=2),
                DepthwiseSeparableConv(round_channels(1024), round_channels(1024), stride=1),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(round_channels(1024), num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))


    # 对比标准卷积 vs 深度可分离卷积的参数量
    def compare_conv_params():
        in_ch, out_ch, K = 64, 128, 3

        # 标准卷积
        std_conv = nn.Conv2d(in_ch, out_ch, K, padding=1)
        std_params = sum(p.numel() for p in std_conv.parameters())

        # 深度可分离卷积
        ds_conv = DepthwiseSeparableConv(in_ch, out_ch)
        ds_params = sum(p.numel() for p in ds_conv.parameters())

        print(f"标准卷积参数: {std_params:,}")
        print(f"深度可分离卷积参数: {ds_params:,}")
        print(f"压缩比: {std_params / ds_params:.1f}x")


    # compare_conv_params()  # 协议化后由 compute_modern_architectures() 控制执行

    # ============================================================
    # 代码段 2
    # ============================================================

    class InvertedResidual(nn.Module):
        """
        MobileNet V2 的倒残差块

        传统残差: 窄 → 宽 → 窄（bottleneck）
        倒残差:   宽 → 窄 → 宽（expand → compress）

        结构: 1×1 扩展 → 3×3 Depthwise → 1×1 压缩
        残差连接只在 stride=1 且输入输出通道相同时使用
        """
        def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
            super().__init__()
            hidden_ch = in_ch * expand_ratio
            self.use_residual = (stride == 1 and in_ch == out_ch)

            layers = []
            # 扩展（1×1 卷积升维）
            if expand_ratio != 1:
                layers.extend([
                    nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                    nn.BatchNorm2d(hidden_ch),
                    nn.ReLU6(inplace=True),
                ])
            # Depthwise（3×3 深度卷积）
            layers.extend([
                nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride,
                          padding=1, groups=hidden_ch, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU6(inplace=True),
            ])
            # 压缩（1×1 卷积降维，不用 ReLU）
            layers.extend([
                nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                # 注意：压缩后不激活（线性瓶颈）
            ])
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            if self.use_residual:
                return x + self.conv(x)
            return self.conv(x)


    class MobileNetV2(nn.Module):
        """MobileNet V2（适配 32×32 CIFAR）"""
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),

                InvertedResidual(32, 16, stride=1, expand_ratio=1),
                InvertedResidual(16, 24, stride=2, expand_ratio=6),
                InvertedResidual(24, 24, stride=1, expand_ratio=6),
                InvertedResidual(24, 32, stride=2, expand_ratio=6),
                InvertedResidual(32, 32, stride=1, expand_ratio=6),
                InvertedResidual(32, 64, stride=2, expand_ratio=6),
                InvertedResidual(64, 64, stride=1, expand_ratio=6),
                InvertedResidual(64, 96, stride=1, expand_ratio=6),
                InvertedResidual(96, 160, stride=2, expand_ratio=6),
                InvertedResidual(160, 320, stride=1, expand_ratio=6),

                nn.Conv2d(320, 1280, 1, bias=False),
                nn.BatchNorm2d(1280),
                nn.ReLU6(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(1280, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # ============================================================
    # 代码段 3
    # ============================================================

    class DenseLayer(nn.Module):
        """
        DenseNet 的单层

        BN → ReLU → 1×1 Conv（瓶颈） → BN → ReLU → 3×3 Conv
        输出 growth_rate 个新通道，拼接到输入
        """
        def __init__(self, in_channels, growth_rate, bn_size=4):
            super().__init__()
            bottleneck = bn_size * growth_rate
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(in_channels, bottleneck, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(bottleneck)
            self.conv2 = nn.Conv2d(bottleneck, growth_rate, 3, padding=1, bias=False)

        def forward(self, x):
            # Concat 输入来自之前所有层
            out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(F.relu(self.bn2(out)))
            return torch.cat([x, out], dim=1)  # 通道拼接


    class DenseBlock(nn.Module):
        """多个 DenseLayer 堆叠"""
        def __init__(self, in_channels, num_layers, growth_rate):
            super().__init__()
            layers = []
            channels = in_channels
            for i in range(num_layers):
                layers.append(DenseLayer(channels, growth_rate))
                channels += growth_rate  # 每层增加 growth_rate 个通道
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)


    class TransitionLayer(nn.Module):
        """
        过渡层：DenseBlock 之间
        BN → 1×1 Conv（压缩通道） → AvgPool2d（下采样）
        """
        def __init__(self, in_channels, compression=0.5):
            super().__init__()
            out_channels = int(in_channels * compression)
            self.norm = nn.BatchNorm2d(in_channels)
            self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.pool = nn.AvgPool2d(2, 2)

        def forward(self, x):
            return self.pool(self.conv(F.relu(self.norm(x))))


    class DenseNet121(nn.Module):
        """
        DenseNet-121 简化版（适配 32×32）

        结构:
        Conv → DenseBlock(6) → Transition → DenseBlock(12) → Transition
             → DenseBlock(24) → Transition → DenseBlock(16) → GAP → FC
        """
        def __init__(self, in_channels=3, num_classes=10, growth_rate=12):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 2 * growth_rate, 3, padding=1, bias=False),
                nn.BatchNorm2d(2 * growth_rate),
                nn.ReLU(inplace=True),

                DenseBlock(2 * growth_rate, num_layers=3, growth_rate=growth_rate),
                TransitionLayer(2 * growth_rate + 3 * growth_rate),
                DenseBlock(int((2 * growth_rate + 3 * growth_rate) * 0.5),
                           num_layers=3, growth_rate=growth_rate),
                TransitionLayer(int((2 * growth_rate + 3 * growth_rate) * 0.5) + 3 * growth_rate),
                DenseBlock(int(((2 * growth_rate + 3 * growth_rate) * 0.5 + 3 * growth_rate) * 0.5),
                           num_layers=3, growth_rate=growth_rate),
            )

            final_ch = int(((2 * growth_rate + 3 * growth_rate) * 0.5 + 3 * growth_rate) * 0.5) + 3 * growth_rate
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(final_ch, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # ============================================================
    # 代码段 4
    # ============================================================

    class EfficientNetBlock(nn.Module):
        """
        EfficientNet 的 MBConv 块（基于 MobileNetV2 倒残差）

        增加了 SE（Squeeze-and-Excitation）注意力
        """
        def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6, se_ratio=0.25):
            super().__init__()
            hidden_ch = in_ch * expand_ratio
            self.use_residual = (stride == 1 and in_ch == out_ch)
            self.use_expand = (expand_ratio != 1)

            # 扩展
            if self.use_expand:
                self.expand_conv = nn.Sequential(
                    nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                    nn.BatchNorm2d(hidden_ch),
                    nn.SiLU(inplace=True),
                )

            # Depthwise
            self.dw_conv = nn.Sequential(
                nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride,
                          padding=1, groups=hidden_ch, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.SiLU(inplace=True),
            )

            # SE 注意力
            se_ch = max(1, int(in_ch * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_ch, se_ch, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_ch, hidden_ch, 1),
                nn.Sigmoid(),
            )

            # 压缩
            self.project_conv = nn.Sequential(
                nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        def forward(self, x):
            residual = x
            if self.use_expand:
                x = self.expand_conv(x)
            x = self.dw_conv(x)
            x = x * self.se(x)  # SE 注意力：通道加权
            x = self.project_conv(x)
            if self.use_residual:
                return x + residual
            return x


    def compound_scaling_experiment():
        """
        复合缩放实验：对比三种缩放策略的效果
        """
        # 基线配置
        base_depth = 3
        base_width = 32
        base_resolution = 32

        # 缩放系数 φ = 2 (FLOPs ~4x)
        phi = 2
        alpha, beta, gamma = 1.2, 1.1, 1.15  # 满足 α·β²·γ² ≈ 2

        configs = {
            '基线 (B0)': (base_depth, base_width, base_resolution),
            '只缩深度': (int(base_depth * alpha ** phi), base_width, base_resolution),
            '只缩宽度': (base_depth, int(base_width * beta ** phi), base_resolution),
            '复合缩放': (int(base_depth * alpha ** phi),
                         int(base_width * beta ** phi),
                         int(base_resolution * gamma ** phi)),
        }

        print("复合缩放策略对比")
        print("=" * 60)
        print(f"{'策略':12s} {'深度':>6s} {'宽度':>6s} {'分辨率':>6s} {'FLOPs':>10s}")
        print("-" * 60)

        for name, (d, w, r) in configs.items():
            # 估算 FLOPs (简化)
            flops = d * w * w * r * r
            print(f"{name:12s} {d:6d} {w:6d} {r:6d} {flops:10d}")

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        names = list(configs.keys())
        depths = [configs[n][0] for n in names]
        widths = [configs[n][1] for n in names]
        resolutions = [configs[n][2] for n in names]

        x = np.arange(len(names))
        width = 0.25

        axes[0].bar(x - width, depths, width, label='深度', color='#4C72B0')
        axes[0].bar(x, widths, width, label='宽度', color='#55A868')
        axes[0].bar(x + width, resolutions, width, label='分辨率', color='#C44E52')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, fontsize=9)
        axes[0].set_title('三种缩放策略的资源分配', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # 精度 vs FLOPs 示意曲线
        flops_range = np.linspace(0.5, 10, 100)
        # 只缩深度：收益递减快
        acc_depth = 70 + 10 * np.log(flops_range) / np.log(10)
        # 只缩宽度：稍好
        acc_width = 70 + 12 * np.log(flops_range) / np.log(10)
        # 复合缩放：最优
        acc_compound = 70 + 15 * np.log(flops_range) / np.log(10)

        axes[1].plot(flops_range, acc_depth, label='只缩深度', linewidth=2)
        axes[1].plot(flops_range, acc_width, label='只缩宽度', linewidth=2)
        axes[1].plot(flops_range, acc_compound, label='复合缩放', linewidth=2)
        axes[1].set_xlabel('FLOPs (相对值)')
        axes[1].set_ylabel('精度 (%)')
        axes[1].set_title('缩放策略 vs 精度\n（示意曲线）', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('compound_scaling.png', dpi=150)
        plt.show()


    # compound_scaling_experiment()  # 协议化后由 compute_modern_architectures() 控制执行

    # ============================================================
    # 代码段 5
    # ============================================================

    def compare_all_architectures():
        """对比所有 CNN 架构"""
        models = {
            'MobileNetV1': MobileNetV1(num_classes=10, width_mult=0.5),
            'MobileNetV2': MobileNetV2(num_classes=10),
            'DenseNet121': DenseNet121(in_channels=3, num_classes=10),
        }

        x = torch.randn(1, 3, 32, 32)

        print("现代 CNN 架构对比")
        print("=" * 60)
        print(f"{'模型':15s} {'参数量':>10s} {'输出形状':>15s}")
        print("-" * 60)

        for name, model in models.items():
            params = sum(p.numel() for p in model.parameters())
            with torch.no_grad():
                out = model(x)
            print(f"{name:15s} {params:10,d} {str(tuple(out.shape)):>15s}")

        # 参数量可视化
        fig, ax = plt.subplots(figsize=(10, 5))

        names = list(models.keys())
        params = [sum(p.numel() for p in m.parameters()) / 1e3 for m in models.values()]
        colors = ['#4C72B0', '#DD8452', '#55A868']

        bars = ax.bar(names, params, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, params):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f'{val:.1f}K', ha='center', fontsize=11, fontweight='bold')

        ax.set_ylabel('参数量（千）')
        ax.set_title('现代 CNN 架构参数量对比', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('modern_arch_comparison.png', dpi=150)
        plt.show()


    # compare_all_architectures()  # 协议化后由 compute_modern_architectures() 控制执行
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part2_cnn/06_modern_architectures.py", e)


def _conv_param_stats(in_ch: int, out_ch: int, kernel_size: int) -> dict[str, float]:
    standard = out_ch * in_ch * kernel_size * kernel_size
    depthwise = in_ch * kernel_size * kernel_size + out_ch * in_ch
    return {
        "standard_params": float(standard),
        "depthwise_params": float(depthwise),
        "compression_ratio": float(standard / max(depthwise, 1)),
    }


def _plot_depthwise_comparison(in_ch: int, out_ch: int, kernel_size: int) -> tuple[object, dict[str, float]]:
    stats = _conv_param_stats(in_ch, out_ch, kernel_size)
    with safe_mpl_figure(figsize=(8.8, 4.0)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        ax1.bar(["标准卷积", "深度可分离"], [stats["standard_params"], stats["depthwise_params"]], color=["#bf3f5b", "#00ff88"], alpha=0.85)
        ax1.set_title("参数量对比", fontsize=10, fontweight="bold")
        ax1.set_ylabel("参数个数")
        ax1.grid(True, axis="y", alpha=0.25)
        pieces = [in_ch * kernel_size * kernel_size, out_ch * in_ch]
        ax2.pie(pieces, labels=["Depthwise\n逐通道空间卷积", "Pointwise\n1x1 通道混合"], autopct="%1.0f%%", colors=["#00f0ff", "#b000ff"])
        ax2.set_title("深度可分离卷积分工", fontsize=10, fontweight="bold")
        fig.suptitle("MobileNet 的核心：先各通道找空间模式，再用 1x1 混合通道", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig, stats


def _compound_scaling_configs(phi: int, base_depth: int, base_width: int, base_resolution: int) -> dict[str, tuple[int, int, int, float]]:
    alpha, beta, gamma = 1.2, 1.1, 1.15
    configs = {
        "基线": (base_depth, base_width, base_resolution),
        "只加深": (int(base_depth * alpha**phi), base_width, base_resolution),
        "只加宽": (base_depth, int(base_width * beta**phi), base_resolution),
        "复合缩放": (int(base_depth * alpha**phi), int(base_width * beta**phi), int(base_resolution * gamma**phi)),
    }
    enriched: dict[str, tuple[int, int, int, float]] = {}
    for name, (depth, width, resolution) in configs.items():
        flops = float(depth * width * width * resolution * resolution)
        enriched[name] = (depth, width, resolution, flops)
    return enriched


def _plot_compound_scaling(phi: int, base_depth: int, base_width: int, base_resolution: int) -> tuple[object, dict[str, tuple[int, int, int, float]]]:
    configs = _compound_scaling_configs(phi, base_depth, base_width, base_resolution)
    names = list(configs.keys())
    depths = [configs[name][0] for name in names]
    widths = [configs[name][1] for name in names]
    resolutions = [configs[name][2] for name in names]
    flops = np.array([configs[name][3] for name in names], dtype=float)
    rel_flops = flops / max(flops[0], 1)
    with safe_mpl_figure(figsize=(11.0, 4.4)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        x = np.arange(len(names))
        width = 0.25
        ax1.bar(x - width, depths, width, label="深度", color="#00f0ff")
        ax1.bar(x, widths, width, label="宽度", color="#b000ff")
        ax1.bar(x + width, resolutions, width, label="分辨率", color="#00ff88")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=15, ha="right")
        ax1.set_title("资源分配方式", fontsize=10, fontweight="bold")
        ax1.grid(True, axis="y", alpha=0.25)
        ax1.legend(fontsize=8)
        acc_depth = 72 + 7 * np.log1p(rel_flops)
        acc_width = 72 + 8 * np.log1p(rel_flops)
        acc_compound = 72 + 10 * np.log1p(rel_flops)
        ax2.plot(rel_flops, acc_depth, "o-", label="只加深", color="#00f0ff")
        ax2.plot(rel_flops, acc_width, "o-", label="只加宽", color="#b000ff")
        ax2.plot(rel_flops, acc_compound, "o-", label="复合缩放", color="#00ff88")
        ax2.set_xlabel("相对 FLOPs")
        ax2.set_ylabel("教学化精度趋势")
        ax2.set_title("同样增加计算量，复合缩放更均衡", fontsize=10, fontweight="bold")
        ax2.grid(True, alpha=0.25)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        return fig, configs


def _modern_model_registry(width_mult: float) -> dict[str, torch.nn.Module]:
    return {
        "MobileNetV1": MobileNetV1(num_classes=10, width_mult=width_mult),
        "MobileNetV2": MobileNetV2(num_classes=10),
        "DenseNet": DenseNet121(in_channels=3, num_classes=10),
    }


def _plot_modern_architectures(width_mult: float) -> tuple[object, dict[str, dict[str, object]]]:
    models = _modern_model_registry(width_mult)
    x = torch.randn(1, 3, 32, 32)
    results: dict[str, dict[str, object]] = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(x)
        results[name] = {
            "params": int(sum(p.numel() for p in model.parameters())),
            "output_shape": tuple(out.shape),
            "depth": int(sum(1 for layer in model.modules() if isinstance(layer, (nn.Conv2d, nn.Linear)))),
        }
    with safe_mpl_figure(figsize=(9.5, 4.2)) as fig:
        ax1, ax2 = fig.subplots(1, 2)
        names = list(results.keys())
        params = [results[name]["params"] / 1000 for name in names]
        depths = [results[name]["depth"] for name in names]
        colors = ["#00f0ff", "#b000ff", "#00ff88"]
        ax1.bar(names, params, color=colors, alpha=0.85)
        ax1.set_title("参数量", fontsize=10, fontweight="bold")
        ax1.set_ylabel("千参数")
        ax1.tick_params(axis="x", rotation=15)
        ax1.grid(True, axis="y", alpha=0.25)
        ax2.scatter(depths, params, c=colors, s=180, edgecolors="white", linewidth=1.5)
        for name, depth, param in zip(names, depths, params):
            ax2.annotate(name, (depth, param), xytext=(6, 5), textcoords="offset points", fontsize=9)
        ax2.set_xlabel("卷积+全连接层数")
        ax2.set_ylabel("千参数")
        ax2.set_title("深度 vs 参数量", fontsize=10, fontweight="bold")
        ax2.grid(True, alpha=0.25)
        fig.suptitle("现代 CNN 架构对比：轻量、复用、密集连接", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig, results


def compute_modern_architectures(
    in_channels: int = 64,
    out_channels: int = 128,
    kernel_size: int = 3,
    width_mult: float = 0.5,
    scaling_phi: int = 2,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute modern CNN architecture comparisons without top-level plotting."""

    in_channels = clamp_int(in_channels, 8, 256, "输入通道")
    out_channels = clamp_int(out_channels, 8, 512, "输出通道")
    kernel_size = clamp_int(kernel_size, 1, 7, "卷积核大小")
    width_mult = clamp_float(width_mult, 0.25, 1.25, "宽度倍率")
    scaling_phi = clamp_int(scaling_phi, 0, 5, "缩放系数 phi")
    torch.manual_seed(seed)
    np.random.seed(seed)
    depthwise_fig, conv_stats = _plot_depthwise_comparison(in_channels, out_channels, kernel_size)
    scaling_fig, scaling_configs = _plot_compound_scaling(scaling_phi, 3, 32, 32)
    arch_fig, arch_results = _plot_modern_architectures(width_mult)
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        print("现代 CNN 架构协议化计算")
        print(f"Depthwise 参数={conv_stats['depthwise_params']:.0f}, 标准卷积参数={conv_stats['standard_params']:.0f}, 压缩比={conv_stats['compression_ratio']:.2f}x")
        for name, values in scaling_configs.items():
            print(f"{name}: depth={values[0]}, width={values[1]}, resolution={values[2]}, FLOPs≈{values[3]:.0f}")
        for name, item in arch_results.items():
            print(f"{name}: params={item['params']:,}, depth={item['depth']}, output={item['output_shape']}")
        print("工程经验：移动端优先考虑深度可分离卷积；服务端大模型再考虑更强的复合缩放和注意力模块。")
    figures = [
        ("modern_depthwise_comparison.png", depthwise_fig),
        ("modern_compound_scaling.png", scaling_fig),
        ("modern_architecture_comparison.png", arch_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    best_light_model = min(arch_results, key=lambda name: arch_results[name]["params"])
    stats = {
        **conv_stats,
        "best_light_model": best_light_model,
        "best_light_params": int(arch_results[best_light_model]["params"]),
        "compound_flops": float(scaling_configs["复合缩放"][3]),
    }
    return {"figures": figures, "artifacts": artifacts, "stats": stats, "architectures": arch_results, "log": log_buffer.getvalue()}


def _go_to_advanced_conv() -> None:
    import streamlit as st

    st.query_params["module"] = "part2_cnn/07_advanced_convolution"
    st.rerun()


def render() -> None:
    """Render the modern CNN architecture lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import render_convolution_particle_flow, render_loading_bar, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="auto")
        render_visual_system("light")
        st.link_button("返回主界面", "/", width="content")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("正在生成现代 CNN 效率对比：深度可分离、复合缩放、架构参数量")
        with st.sidebar:
            in_channels = st.slider("输入通道", 8, 256, 64, 8)
            out_channels = st.slider("输出通道", 8, 512, 128, 8)
            kernel_size = st.slider("卷积核大小", 1, 7, 3, 2)
            width_mult = st.slider("MobileNet 宽度倍率", 0.25, 1.25, 0.5, 0.05)
            scaling_phi = st.slider("复合缩放 phi", 0, 5, 2, 1)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("继续看：高级卷积", width="stretch"):
                _go_to_advanced_conv()
        data = compute_modern_architectures(in_channels, out_channels, kernel_size, width_mult, scaling_phi, int(seed), save_artifacts=True)
        stats = data["stats"]
        render_convolution_particle_flow()
        st.markdown(
            """
            **零基础直觉：**现代 CNN 的核心不是单纯“更深”，而是更会省力。
            MobileNet 把空间卷积和通道混合拆开，DenseNet 让后面的层直接复用前面的特征，
            EfficientNet 则同时调深度、宽度和分辨率，避免只把一个方向拉得过头。
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("深度可分离压缩比", f"{stats['compression_ratio']:.2f}x")
        c2.metric("最轻模型", str(stats["best_light_model"]))
        c3.metric("最轻模型参数", f"{stats['best_light_params']:,}")
        explainers = [
            ("深度可分离卷积", "标准卷积同时做空间搜索和通道混合；深度可分离卷积把两件事拆开，因此参数量大幅下降。"),
            ("复合缩放", "只加深、只加宽或只加分辨率都会很快遇到收益递减；复合缩放让三种资源更均衡。"),
            ("现代架构对比", "MobileNet 强调移动端轻量，DenseNet 强调特征复用，EfficientNet 思想强调资源配比。"),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"图像产物已放入统一目录：{get_artifact_path(filename)}")
            st.markdown("> 请只调整一个参数，观察压缩比、FLOPs 或参数量如何变化。思考：这是省了空间卷积，还是省了通道混合？")
        with st.expander("控制台输出与工程经验", expanded=False):
            st.markdown(
                """
                - **移动端/边缘设备**：优先考虑 MobileNet 系列和深度可分离卷积。
                - **特征复用明显的任务**：DenseNet 思想有帮助，但连接多会增加显存压力。
                - **扩大模型规模**：不要只加深或只加宽，先考虑深度、宽度、分辨率的平衡。
                """
            )
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part2_cnn/06_modern_architectures.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_modern_architectures(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_modern_architectures(in_channels=16, out_channels=32, kernel_size=3, width_mult=0.25, scaling_phi=1, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["compression_ratio"] > 1 and data["stats"]["best_light_params"] > 0


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_modern_architectures))
