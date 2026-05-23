"""
自动生成自: part2_cnn\02_feature_maps.md
可独立运行的 Python 源码
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons
from typing import List, Dict, Optional

# ─────────────────────────────────────────────────────────
# 核心：多层特征图提取 + 交互式查看
# ─────────────────────────────────────────────────────────

class RealTimeFeatureViewer:
    """
    实时特征图查看器

    功能：
    - 注册任意层的 forward hook
    - 输入一张图像，立即看到所有层的特征图
    - 支持交互式切换层、通道

    使用方法：
        viewer = RealTimeFeatureViewer(model)
        viewer.register_all_conv_layers()
        viewer.show(input_tensor)          # 静态查看
        viewer.interactive_show(input_tensor)  # 交互式
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List = []
        self.layer_outputs: Dict[str, torch.Tensor] = {}
        self.layer_names: List[str] = []

    def register_layer(self, name: str):
        """注册单个层"""
        for n, module in self.model.named_modules():
            if n == name:
                def hook(m, inp, out, _name=name):
                    self.layer_outputs[_name] = out.detach().cpu()
                self.hooks.append(module.register_forward_hook(hook))
                if name not in self.layer_names:
                    self.layer_names.append(name)
                return
        print(f"未找到层: {name}")

    def register_all_conv_layers(self):
        """自动注册所有卷积层"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                self.register_layer(name)
        print(f"已注册 {len(self.layer_names)} 个卷积层: {self.layer_names}")

    def register_all_layers(self, types=(nn.Conv2d, nn.ReLU, nn.BatchNorm2d, nn.Linear)):
        """注册指定类型的所有层"""
        for name, module in self.model.named_modules():
            if isinstance(module, types) and name:
                self.register_layer(name)

    def forward(self, x: torch.Tensor):
        """运行前向传播，捕获所有注册层的输出"""
        self.layer_outputs.clear()
        with torch.no_grad():
            out = self.model(x)
        return out

    def show(self, x: torch.Tensor, max_channels: int = 16,
             cmap: str = 'viridis', figsize_per: float = 1.4):
        """静态显示所有层的特征图"""
        self.forward(x)

        for layer_name in self.layer_names:
            if layer_name not in self.layer_outputs:
                continue
            feat = self.layer_outputs[layer_name]

            if feat.dim() == 4:
                feat = feat[0]  # [C, H, W]
                n_ch = min(feat.shape[0], max_channels)
                n_cols = min(8, n_ch)
                n_rows = (n_ch + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols,
                    figsize=(n_cols * figsize_per, n_rows * figsize_per + 0.5))
                axes = np.array(axes).reshape(n_rows, n_cols)

                vmin, vmax = feat[:n_ch].min().item(), feat[:n_ch].max().item()

                for i in range(n_ch):
                    ax = axes[i // n_cols, i % n_cols]
                    im = ax.imshow(feat[i].numpy(), cmap=cmap,
                                   vmin=vmin, vmax=vmax, aspect='auto')
                    ax.set_title(f'ch{i}', fontsize=7)
                    ax.axis('off')

                for i in range(n_ch, n_rows * n_cols):
                    axes[i // n_cols, i % n_cols].axis('off')

                shape_str = f'{feat.shape[0]}×{feat.shape[1]}×{feat.shape[2]}'
                plt.suptitle(f'层: {layer_name}  |  特征图形状: {shape_str}',
                             fontsize=10, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'feat_{layer_name.replace(".", "_")}.png',
                            dpi=120, bbox_inches='tight')
                plt.show()

    def show_single_layer(self, layer_name: str, x: torch.Tensor,
                           channel: int = 0, cmap: str = 'viridis'):
        """显示单层单通道的特征图，附带统计信息"""
        self.forward(x)
        if layer_name not in self.layer_outputs:
            print(f"层 {layer_name} 未捕获")
            return

        feat = self.layer_outputs[layer_name][0]  # [C, H, W]
        ch_map = feat[channel].numpy()

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # 特征图
        im = axes[0].imshow(ch_map, cmap=cmap)
        plt.colorbar(im, ax=axes[0])
        axes[0].set_title(f'{layer_name} | 通道 {channel}', fontsize=11)
        axes[0].axis('off')

        # 激活值分布直方图
        axes[1].hist(ch_map.flatten(), bins=40, color='steelblue',
                     edgecolor='white', alpha=0.8)
        axes[1].set_title('激活值分布', fontsize=11)
        axes[1].set_xlabel('激活值')
        axes[1].set_ylabel('频次')
        axes[1].axvline(0, color='red', linestyle='--', alpha=0.7, label='零值')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 所有通道的平均激活强度
        ch_means = feat.abs().mean(dim=(1, 2)).numpy()
        colors = ['red' if v == ch_means.max() else 'steelblue' for v in ch_means]
        axes[2].bar(range(len(ch_means)), ch_means, color=colors, alpha=0.8)
        axes[2].set_title('各通道平均激活强度\n（红色=当前通道）', fontsize=10)
        axes[2].set_xlabel('通道索引')
        axes[2].set_ylabel('|激活值| 均值')
        axes[2].grid(True, alpha=0.3)

        stats = (f'均值={ch_map.mean():.3f}  标准差={ch_map.std():.3f}  '
                 f'最大={ch_map.max():.3f}  最小={ch_map.min():.3f}  '
                 f'零值比={( ch_map == 0).mean():.1%}')
        plt.suptitle(stats, fontsize=9)
        plt.tight_layout()
        plt.savefig(f'feat_single_{layer_name.replace(".", "_")}_ch{channel}.png',
                    dpi=120, bbox_inches='tight')
        plt.show()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─────────────────────────────────────────────────────────
# 滤波器响应热力图：哪些输入区域激活了哪个滤波器？
# ─────────────────────────────────────────────────────────

def plot_filter_response_heatmap(model: nn.Module, x: torch.Tensor,
                                  layer_name: str, top_k: int = 4):
    """
    可视化 top-k 最活跃通道的响应热力图，叠加在原图上

    原理：将特征图上采样到原图尺寸，用颜色表示激活强度
    """
    captured = {}

    def hook(m, inp, out):
        captured['feat'] = out.detach().cpu()

    # 注册钩子
    hook_handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            hook_handle = module.register_forward_hook(hook)
            break

    if hook_handle is None:
        print(f"未找到层: {layer_name}")
        return

    with torch.no_grad():
        model(x)
    hook_handle.remove()

    feat = captured['feat'][0]  # [C, H, W]
    # 找 top-k 最活跃通道（按平均激活值排序）
    ch_scores = feat.abs().mean(dim=(1, 2))
    top_channels = ch_scores.topk(min(top_k, feat.shape[0])).indices.tolist()

    # 原图（假设单通道或 RGB）
    orig = x[0].cpu()
    if orig.shape[0] == 1:
        orig_img = orig[0].numpy()
        orig_cmap = 'gray'
    else:
        orig_img = orig.permute(1, 2, 0).numpy()
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
        orig_cmap = None

    H_orig, W_orig = orig_img.shape[:2]

    fig, axes = plt.subplots(1, top_k + 1, figsize=((top_k + 1) * 3.5, 3.5))

    # 原图
    axes[0].imshow(orig_img, cmap=orig_cmap)
    axes[0].set_title('原始输入', fontsize=10)
    axes[0].axis('off')

    for i, ch_idx in enumerate(top_channels):
        heatmap = feat[ch_idx].numpy()
        # 上采样到原图尺寸
        heatmap_resized = torch.nn.functional.interpolate(
            torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float(),
            size=(H_orig, W_orig), mode='bilinear', align_corners=False
        )[0, 0].numpy()

        # 归一化
        hm_min, hm_max = heatmap_resized.min(), heatmap_resized.max()
        heatmap_norm = (heatmap_resized - hm_min) / (hm_max - hm_min + 1e-8)

        axes[i + 1].imshow(orig_img, cmap=orig_cmap, alpha=0.5)
        axes[i + 1].imshow(heatmap_norm, cmap='jet', alpha=0.5)
        axes[i + 1].set_title(f'通道 {ch_idx}\n激活强度={ch_scores[ch_idx]:.3f}', fontsize=9)
        axes[i + 1].axis('off')

    plt.suptitle(f'层 {layer_name} — Top-{top_k} 最活跃通道响应热力图',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'heatmap_{layer_name.replace(".", "_")}.png',
                dpi=120, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────
# 完整演示
# ─────────────────────────────────────────────────────────

def demo_feature_maps():
    torch.manual_seed(42)

    # 构建带命名层的 CNN
    class DebugCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.bn1   = nn.BatchNorm2d(8)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.bn2   = nn.BatchNorm2d(16)
            self.relu2 = nn.ReLU()
            self.pool  = nn.MaxPool2d(2)
            self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
            self.relu3 = nn.ReLU()
            self.gap   = nn.AdaptiveAvgPool2d(1)
            self.fc    = nn.Linear(32, 10)

        def forward(self, x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = self.relu3(self.conv3(x))
            x = self.gap(x).flatten(1)
            return self.fc(x)

    model = DebugCNN()
    x = torch.randn(1, 1, 28, 28)

    # 1. 注册所有卷积层并查看特征图
    viewer = RealTimeFeatureViewer(model)
    viewer.register_all_conv_layers()
    viewer.show(x, max_channels=8)

    # 2. 单层单通道详细分析
    viewer.show_single_layer('conv1', x, channel=0)
    viewer.show_single_layer('conv2', x, channel=3)

    # 3. 响应热力图
    plot_filter_response_heatmap(model, x, layer_name='conv1', top_k=4)
    plot_filter_response_heatmap(model, x, layer_name='conv3', top_k=4)

    viewer.remove_hooks()
    return model, viewer

model, viewer = demo_feature_maps()

# ============================================================
# 代码段 2
# ============================================================

def compare_filters_before_after(model_before: nn.Module,
                                   model_after: nn.Module,
                                   layer_name: str,
                                   max_filters: int = 16):
    """
    对比训练前后卷积核的变化

    model_before: 随机初始化的模型
    model_after:  训练后的模型
    """
    def get_filters(model):
        for name, module in model.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                w = module.weight.detach().cpu()
                w_min, w_max = w.min(), w.max()
                return (w - w_min) / (w_max - w_min + 1e-8)
        return None

    w_before = get_filters(model_before)
    w_after  = get_filters(model_after)

    if w_before is None or w_after is None:
        print(f"未找到卷积层: {layer_name}")
        return

    n = min(max_filters, w_before.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))

    for i in range(n):
        in_ch = w_before.shape[1]
        if in_ch == 1:
            axes[0, i].imshow(w_before[i, 0].numpy(), cmap='RdBu', vmin=0, vmax=1)
            axes[1, i].imshow(w_after[i, 0].numpy(),  cmap='RdBu', vmin=0, vmax=1)
        else:
            axes[0, i].imshow(np.clip(w_before[i].permute(1,2,0).numpy(), 0, 1))
            axes[1, i].imshow(np.clip(w_after[i].permute(1,2,0).numpy(),  0, 1))

        axes[0, i].axis('off')
        axes[1, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('训练前', fontsize=9)
            axes[1, i].set_ylabel('训练后', fontsize=9)

    plt.suptitle(f'卷积核对比: {layer_name}（训练前 vs 训练后）',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'filter_compare_{layer_name.replace(".", "_")}.png',
                dpi=120, bbox_inches='tight')
    plt.show()


# 演示：随机初始化 vs 简单训练后
torch.manual_seed(0)

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.AdaptiveAvgPool2d(4)
        self.fc    = nn.Linear(8*16, 10)
    def forward(self, x):
        return self.fc(self.pool(self.relu(self.conv1(x))).flatten(1))

import copy
model_before = TinyCNN()
model_after  = copy.deepcopy(model_before)

# 快速训练几步
opt = torch.optim.Adam(model_after.parameters(), lr=0.01)
for _ in range(50):
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    loss = nn.CrossEntropyLoss()(model_after(x), y)
    opt.zero_grad(); loss.backward(); opt.step()

compare_filters_before_after(model_before, model_after, 'conv1')
