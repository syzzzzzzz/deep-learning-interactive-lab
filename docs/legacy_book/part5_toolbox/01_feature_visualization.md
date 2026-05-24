# 第九章补充：特征可视化工具

---

## 9.0 为什么要可视化特征？

神经网络是"黑盒"——但通过可视化中间层的特征图，我们可以：
- 理解每一层"看到"了什么
- 诊断模型是否学到了有意义的表示
- 发现过拟合、欠拟合的早期信号

---

## 9.1 CNN 特征图可视化

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────
# 特征图提取器：用 forward hook 捕获中间层输出
# ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    通用特征图提取器

    使用方法：
        extractor = FeatureExtractor(model, ['conv1', 'layer1', 'layer2'])
        features = extractor(input_tensor)
        extractor.visualize('conv1')
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self.features.update({n: out.detach()})
                )
                self._hooks.append(hook)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.features.clear()
        with torch.no_grad():
            out = self.model(x)
        return out

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def visualize(self, layer_name: str, max_channels: int = 16,
                  figsize_per_channel: float = 1.5):
        """
        可视化指定层的特征图

        layer_name: 要可视化的层名
        max_channels: 最多显示多少个通道
        """
        if layer_name not in self.features:
            print(f"层 '{layer_name}' 未找到。可用层: {list(self.features.keys())}")
            return

        feat = self.features[layer_name]

        # 支持 [B, C, H, W] 和 [B, C] 两种形状
        if feat.dim() == 4:
            feat = feat[0]  # 取第一个样本 → [C, H, W]
            n_channels = min(feat.shape[0], max_channels)
            n_cols = min(8, n_channels)
            n_rows = (n_channels + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * figsize_per_channel,
                                              n_rows * figsize_per_channel))
            axes = np.array(axes).reshape(n_rows, n_cols)

            for i in range(n_channels):
                ax = axes[i // n_cols, i % n_cols]
                channel_map = feat[i].numpy()
                ax.imshow(channel_map, cmap='viridis', aspect='auto')
                ax.set_title(f'Ch {i}', fontsize=7)
                ax.axis('off')

            # 隐藏多余的子图
            for i in range(n_channels, n_rows * n_cols):
                axes[i // n_cols, i % n_cols].axis('off')

            plt.suptitle(f'特征图: {layer_name}  shape={tuple(self.features[layer_name][0].shape)}',
                         fontsize=11, fontweight='bold')

        elif feat.dim() == 2:
            # 全连接层：显示激活值分布
            feat = feat[0].numpy()
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            axes[0].bar(range(len(feat)), feat, color='steelblue', alpha=0.7)
            axes[0].set_title(f'{layer_name} 激活值', fontsize=11)
            axes[0].set_xlabel('神经元索引')
            axes[0].set_ylabel('激活值')
            axes[0].grid(True, alpha=0.3)

            axes[1].hist(feat, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
            axes[1].set_title(f'{layer_name} 激活值分布', fontsize=11)
            axes[1].set_xlabel('激活值')
            axes[1].set_ylabel('频次')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = layer_name.replace('.', '_')
        plt.savefig(f'feature_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_all(self, max_channels: int = 8):
        """可视化所有已捕获层的特征图"""
        for name in self.features:
            print(f"\n可视化层: {name}")
            self.visualize(name, max_channels=max_channels)


# ─────────────────────────────────────────────────────────
# 卷积核可视化
# ─────────────────────────────────────────────────────────

def visualize_conv_filters(model: nn.Module, layer_name: str,
                            max_filters: int = 32, figsize_per: float = 1.2):
    """
    可视化卷积层的学习到的滤波器（权重）

    layer_name: 卷积层名称（如 'features.0'）
    """
    # 找到目标层
    target = None
    for name, module in model.named_modules():
        if name == layer_name:
            target = module
            break

    if target is None or not isinstance(target, nn.Conv2d):
        print(f"未找到卷积层 '{layer_name}'")
        return

    weights = target.weight.detach().cpu()  # [out_ch, in_ch, kH, kW]
    n_filters = min(weights.shape[0], max_filters)
    in_ch = weights.shape[1]

    # 归一化到 [0, 1] 用于显示
    w_min, w_max = weights.min(), weights.max()
    weights_norm = (weights - w_min) / (w_max - w_min + 1e-8)

    n_cols = min(8, n_filters)
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * figsize_per, n_rows * figsize_per))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_filters):
        ax = axes[i // n_cols, i % n_cols]
        if in_ch == 1:
            ax.imshow(weights_norm[i, 0].numpy(), cmap='RdBu', vmin=0, vmax=1)
        elif in_ch == 3:
            # RGB 滤波器
            rgb = weights_norm[i].permute(1, 2, 0).numpy()
            ax.imshow(np.clip(rgb, 0, 1))
        else:
            # 多通道：显示第一个通道
            ax.imshow(weights_norm[i, 0].numpy(), cmap='viridis')
        ax.axis('off')
        ax.set_title(f'F{i}', fontsize=7)

    for i in range(n_filters, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis('off')

    plt.suptitle(f'卷积滤波器: {layer_name}  [{weights.shape[0]}个滤波器, {in_ch}通道, {weights.shape[2]}×{weights.shape[3]}]',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'filters_{layer_name.replace(".", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────
# 激活最大化（Activation Maximization）
# 找到让某个神经元激活最大的输入图像
# ─────────────────────────────────────────────────────────

def activation_maximization(model: nn.Module, layer_name: str,
                              channel_idx: int = 0,
                              input_size: tuple = (1, 1, 28, 28),
                              n_steps: int = 200,
                              lr: float = 0.1,
                              l2_reg: float = 0.01):
    """
    激活最大化：找到让指定通道激活最大的输入

    原理：固定模型权重，对输入图像做梯度上升
    """
    model.eval()

    # 捕获目标层输出
    target_activation = [None]

    def hook_fn(module, inp, out):
        target_activation[0] = out

    # 注册钩子
    hook = None
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break

    if hook is None:
        print(f"未找到层 '{layer_name}'")
        return None

    # 初始化输入（随机噪声）
    x = torch.randn(input_size, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        model(x)

        # 最大化目标通道的平均激活
        if target_activation[0] is not None:
            if target_activation[0].dim() == 4:
                loss = -target_activation[0][0, channel_idx].mean()
            else:
                loss = -target_activation[0][0, channel_idx]

            # L2 正则化（防止像素值过大）
            loss = loss + l2_reg * (x ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    hook.remove()

    # 可视化结果
    img = x.detach()[0]
    if img.shape[0] == 1:
        img = img[0].numpy()
        cmap = 'gray'
    else:
        img = img.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        cmap = None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img, cmap=cmap)
    axes[0].set_title(f'激活最大化结果\n层: {layer_name}, 通道: {channel_idx}', fontsize=11)
    axes[0].axis('off')

    axes[1].plot(losses, 'b-', alpha=0.8)
    axes[1].set_title('优化过程（损失 = -激活值）', fontsize=11)
    axes[1].set_xlabel('步数')
    axes[1].set_ylabel('损失')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('激活最大化：让神经元"最兴奋"的输入', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'activation_max_{layer_name.replace(".", "_")}_ch{channel_idx}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    return x.detach()


# ─────────────────────────────────────────────────────────
# 完整演示
# ─────────────────────────────────────────────────────────

def demo_feature_visualization():
    """演示所有特征可视化工具"""
    torch.manual_seed(42)

    # 构建一个简单 CNN
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),   # features.0
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),  # features.2
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 64),       # classifier.0
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    # 给层命名（Sequential 默认用数字索引）
    named_model = nn.Sequential()
    named_model.add_module('conv1', nn.Conv2d(1, 8, 3, padding=1))
    named_model.add_module('relu1', nn.ReLU())
    named_model.add_module('conv2', nn.Conv2d(8, 16, 3, padding=1))
    named_model.add_module('relu2', nn.ReLU())
    named_model.add_module('pool', nn.AdaptiveAvgPool2d(4))
    named_model.add_module('flatten', nn.Flatten())
    named_model.add_module('fc1', nn.Linear(16 * 4 * 4, 64))
    named_model.add_module('relu3', nn.ReLU())
    named_model.add_module('fc2', nn.Linear(64, 10))

    # 生成一张假图像
    x = torch.randn(1, 1, 28, 28)

    # 1. 特征图可视化
    print("1. 特征图可视化")
    extractor = FeatureExtractor(named_model, ['conv1', 'conv2', 'relu2', 'fc1'])
    extractor(x)
    extractor.visualize('conv1', max_channels=8)
    extractor.visualize('conv2', max_channels=16)
    extractor.visualize('fc1')
    extractor.remove_hooks()

    # 2. 卷积核可视化
    print("\n2. 卷积核可视化")
    visualize_conv_filters(named_model, 'conv1', max_filters=8)
    visualize_conv_filters(named_model, 'conv2', max_filters=16)

    # 3. 激活最大化
    print("\n3. 激活最大化")
    activation_maximization(named_model, 'conv1', channel_idx=0,
                             input_size=(1, 1, 28, 28), n_steps=100)

    return named_model, extractor

named_model, extractor = demo_feature_visualization()
```

---

## 小结

| 工具 | 用途 | 关键方法 |
|------|------|----------|
| FeatureExtractor | 捕获并可视化中间层特征图 | `register_forward_hook`, `visualize()` |
| visualize_conv_filters | 查看卷积核学到的模式 | 直接读取 `.weight` |
| activation_maximization | 找到让神经元最兴奋的输入 | 梯度上升优化输入图像 |
