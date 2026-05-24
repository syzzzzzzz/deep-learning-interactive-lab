# CNN 可视化技术：GradCAM / 显著图 / 特征反转 / DeepDream

## 1. 为什么需要可视化？

```
CNN 被称为"黑箱"：
输入图像 → [???] → "这是猫"

可视化回答：
1. 模型在看哪里做决策？ → GradCAM
2. 哪些像素对输出影响最大？ → 显著图
3. 每层学到了什么特征？ → 特征反转
4. 如何让模型"做梦"？ → DeepDream
```

---

## 2. GradCAM：梯度加权类激活图

### 2.1 原理

```
1. 前向传播，记录最后一个卷积层的特征图 A_k
2. 对目标类别 c 的输出 y_c 求梯度：∂y_c / ∂A_k
3. 全局平均池化得到每个通道的重要性权重：
   α_k = (1/Z) × Σ_i Σ_j (∂y_c / ∂A_k(i,j))
4. 加权求和 + ReLU：
   L_GradCAM = ReLU(Σ_k α_k × A_k)

ReLU 只保留对类别 c 有正贡献的区域
```

### 2.2 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    """
    GradCAM 实现

    使用方法：
        model = ...  # 你的 CNN
        grad_cam = GradCAM(model, target_layer='layer4')
        heatmap = grad_cam.generate(input_tensor, target_class=None)
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                # 前向 hook：记录激活
                module.register_forward_hook(self._save_activation)
                # 反向 hook：记录梯度
                module.register_forward_hook(self._save_gradient)
                break

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, target_class=None):
        """
        生成 GradCAM 热力图

        x: [1, C, H, W] 输入张量
        target_class: 目标类别（None=取预测类别）
        返回: [H, W] 热力图（0-1）
        """
        # 前向传播
        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播：只对目标类别求梯度
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算通道权重
        # gradients: [1, C, h, w] → 全局平均池化 → [1, C, 1, 1]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # 加权求和
        # activations: [1, C, h, w], weights: [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # 只保留正贡献

        # 归一化到 0-1
        cam = cam.squeeze().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, target_class

    def generate_and_overlay(self, x, original_image, target_class=None):
        """
        生成热力图并叠加到原图

        original_image: [H, W, C] 或 [H, W] numpy 数组 (0-255 或 0-1)
        """
        cam, target_class = self.generate(x, target_class)

        # 上采样到原图尺寸
        H, W = original_image.shape[:2]
        cam_resized = F.interpolate(
            torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().numpy()

        # 创建彩色热力图
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]

        # 叠加
        if original_image.max() > 1.0:
            original_image = original_image / 255.0
        overlay = 0.5 * original_image + 0.5 * heatmap

        return overlay, target_class, cam_resized


# ─────────────────────────────────────────────────────────
# 用简单 CNN 演示 GradCAM
# ─────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """带命名层的简单 CNN"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # 最后一层卷积
        x = self.pool(x).flatten(1)
        return self.fc(x)


def demo_gradcam():
    """GradCAM 演示"""
    model = SimpleCNN()
    model.eval()

    # 生成测试图像
    img = np.zeros((28, 28), dtype=np.float32)
    img[8:20, 8:20] = 1.0  # 方块

    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

    grad_cam = GradCAM(model, target_layer='conv3')
    cam, target_class = grad_cam.generate(x)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('输入图像', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f'GradCAM 热力图\n类别: {target_class}', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray', alpha=0.5)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title('叠加效果', fontsize=12)
    axes[2].axis('off')

    plt.suptitle('GradCAM：模型在看哪里？', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gradcam_demo.png', dpi=150)
    plt.show()


demo_gradcam()
```

---

## 3. 显著图（Saliency Map）

### 3.1 原理

```
显著图：对输入像素求梯度，看哪些像素对输出影响最大

∂y_c / ∂x_ij = 像素 (i,j) 对类别 c 输出的敏感度

计算步骤：
1. 前向传播得到输出 y_c
2. 对输入 x 求梯度：∂y_c / ∂x
3. 取绝对值并在通道维度取最大值
4. 归一化到 0-1
```

### 3.2 实现

```python
def compute_saliency_map(model, x, target_class=None):
    """
    计算显著图

    model: 评估模式的 CNN
    x: [1, C, H, W] 输入（需要 grad）
    target_class: 目标类别
    返回: [H, W] 显著图
    """
    x.requires_grad_(True)

    output = model(x)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # 对目标类别的输出求梯度
    score = output[0, target_class]
    score.backward()

    # 取梯度的绝对值，通道维度取最大值
    saliency = x.grad.abs().max(dim=1)[0].squeeze().numpy()

    # 归一化
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return saliency, target_class


def demo_saliency():
    """显著图演示"""
    model = SimpleCNN()
    model.eval()

    img = np.zeros((28, 28), dtype=np.float32)
    # 十字形状
    img[12:16, :] = 1.0
    img[:, 12:16] = 1.0

    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

    saliency, target_class = compute_saliency_map(model, x)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('输入图像', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title(f'显著图\n类别: {target_class}', fontsize=12)
    axes[1].axis('off')

    # SmoothGrad：添加噪声取平均（更稳定）
    n_samples = 50
    stdev = 0.1
    smooth_saliency = np.zeros_like(saliency)
    for _ in range(n_samples):
        noise = torch.randn_like(x) * stdev
        noisy_x = (x + noise).detach().requires_grad_(True)
        s, _ = compute_saliency_map(model, noisy_x, target_class)
        smooth_saliency += s
    smooth_saliency /= n_samples

    axes[2].imshow(smooth_saliency, cmap='hot')
    axes[2].set_title(f'SmoothGrad\n({n_samples}次平均)', fontsize=12)
    axes[2].axis('off')

    plt.suptitle('显著图：哪些像素最重要？', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('saliency_demo.png', dpi=150)
    plt.show()


demo_saliency()
```

---

## 4. 特征反转（Feature Inversion）

### 4.1 原理

```
问题：卷积层的特征向量到底编码了什么信息？

方法：从特征向量反推原图
1. 给定目标特征向量 φ*（某层对某图的输出）
2. 从随机噪声 x 开始
3. 优化 x 使其特征 φ(x) 接近 φ*
4. L(x) = ||φ(x) - φ*||² + TV 正则化

恢复的图像展示了该层保留了什么信息
浅层：颜色、边缘（接近原图）
深层：语义内容（丢失纹理细节）
```

### 4.2 实现

```python
def feature_inversion(model, target_image, layer_name, num_iter=100, lr=0.1):
    """
    特征反转

    从随机噪声开始，优化出能产生相同特征的图像
    """
    # 获取目标特征
    target_feat = {}
    def hook(m, inp, out):
        target_feat['value'] = out.detach()

    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break

    with torch.no_grad():
        model(target_image)
    target_feature = target_feat['value']
    handle.remove()

    # 从随机噪声开始优化
    generated = torch.randn_like(target_image, requires_grad=True)
    optimizer = torch.optim.Adam([generated], lr=lr)

    # TV 正则化损失
    def total_variation(x):
        return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
               torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

    generated_feats = {}
    def hook2(m, inp, out):
        generated_feats['value'] = out

    for name, module in model.named_modules():
        if name == layer_name:
            handle2 = module.register_forward_hook(hook2)
            break

    history = []
    for i in range(num_iter):
        optimizer.zero_grad()
        model(generated)
        feat_loss = F.mse_loss(generated_feats['value'], target_feature)
        tv_loss = total_variation(generated) * 0.001
        loss = feat_loss + tv_loss
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    handle2.remove()

    return generated.detach(), history


def demo_feature_inversion():
    """演示不同层的特征反转"""
    model = SimpleCNN()
    model.eval()

    # 目标图像
    img = np.zeros((28, 28), dtype=np.float32)
    img[8:20, 8:20] = 1.0
    target = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('原图', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    layers = ['conv1', 'conv2', 'conv3']
    for idx, layer_name in enumerate(layers):
        gen, hist = feature_inversion(model, target, layer_name,
                                       num_iter=100, lr=0.05)
        axes[idx + 1].imshow(gen[0, 0].numpy(), cmap='gray')
        axes[idx + 1].set_title(f'{layer_name} 反转\n保留了高层特征',
                                 fontsize=10, fontweight='bold')
        axes[idx + 1].axis('off')

    plt.suptitle('特征反转：每层编码了什么？', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_inversion.png', dpi=150)
    plt.show()


demo_feature_inversion()
```

---

## 5. DeepDream：让神经网络做梦

### 5.1 原理

```
DeepDream = 反向优化的特征可视化

普通训练: 优化权重使特征匹配目标
DeepDream: 优化输入使某层特征最大化

核心：对输入图像做梯度上升
x_new = x + lr × ∂(激活值) / ∂x

效果：放大模型在该层"看到"的模式
浅层：边缘、纹理
深层：物体零件、整个物体
```

### 5.2 实现

```python
def deepdream(model, image, layer_name, channel=None,
              num_iter=50, lr=0.05, clip=True):
    """
    DeepDream 实现

    通过梯度上升最大化指定层的激活值
    """
    x = image.clone().detach().requires_grad_(True)

    acts = {}
    def hook(m, inp, out):
        acts['value'] = out

    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break

    for i in range(num_iter):
        model(x)
        activation = acts['value']

        if channel is not None:
            # 最大化特定通道
            loss = -activation[0, channel].mean()
        else:
            # 最大化所有通道
            loss = -activation.mean()

        loss.backward()

        # 梯度上升
        grad = x.grad.data
        # 归一化梯度步长
        grad = grad / (grad.std() + 1e-8)
        x.data += lr * grad

        if clip:
            x.data = torch.clamp(x.data, 0, 1)

        x.grad.zero_()

    handle.remove()
    return x.detach()


def demo_deepdream():
    """DeepDream 演示"""
    model = SimpleCNN()
    model.eval()

    # 起始图像：带微弱噪声的灰色背景
    base = np.full((28, 28), 0.5, dtype=np.float32)
    base += np.random.randn(28, 28).astype(np.float32) * 0.1
    base = np.clip(base, 0, 1)

    image = torch.from_numpy(base).float().unsqueeze(0).unsqueeze(0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(base, cmap='gray')
    axes[0, 0].set_title('起始图像', fontsize=12)
    axes[0, 0].axis('off')

    # 不同层的 Dream
    for idx, layer_name in enumerate(['conv1', 'conv2', 'conv3']):
        dream = deepdream(model, image, layer_name, num_iter=50, lr=0.05)
        axes[0, idx + 1].imshow(dream[0, 0].numpy(), cmap='gray')
        axes[0, idx + 1].set_title(f'{layer_name} Dream', fontsize=12)
        axes[0, idx + 1].axis('off')

    # 不同通道的 Dream
    for idx, ch in enumerate([0, 4, 8]):
        dream = deepdream(model, image, 'conv2', channel=ch,
                          num_iter=50, lr=0.05)
        axes[1, idx + 1].imshow(dream[0, 0].numpy(), cmap='gray')
        axes[1, idx + 1].set_title(f'conv2 ch{ch} Dream', fontsize=12)
        axes[1, idx + 1].axis('off')

    axes[1, 0].imshow(base, cmap='gray')
    axes[1, 0].set_title('起始图像', fontsize=12)
    axes[1, 0].axis('off')

    plt.suptitle('DeepDream：让神经网络做梦', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('deepdream_demo.png', dpi=150)
    plt.show()


demo_deepdream()
```

---

## 6. 综合可视化工具箱

```python
class CNNVisualizer:
    """
    CNN 综合可视化工具

    整合 GradCAM、显著图、特征反转、DeepDream
    """

    def __init__(self, model, last_conv_layer):
        self.model = model
        self.last_conv_layer = last_conv_layer

    def full_analysis(self, x, class_names=None):
        """一次性生成所有可视化"""
        self.model.eval()

        # 1. 预测
        with torch.no_grad():
            output = self.model(x)
            probs = F.softmax(output, dim=1)[0]
            pred_class = output.argmax(dim=1).item()

        # 2. GradCAM
        grad_cam = GradCAM(self.model, self.last_conv_layer)
        cam, _ = grad_cam.generate(x, pred_class)

        # 3. 显著图
        x_saliency = x.clone().detach().requires_grad_(True)
        saliency, _ = compute_saliency_map(self.model, x_saliency, pred_class)

        # 4. 可视化
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # 原图
        img = x[0, 0].numpy() if x.shape[1] == 1 else x[0].permute(1, 2, 0).numpy()
        axes[0, 0].imshow(img, cmap='gray' if x.shape[1] == 1 else None)
        label = class_names[pred_class] if class_names else str(pred_class)
        axes[0, 0].set_title(f'输入 → 预测: {label}\n置信度: {probs[pred_class]:.2%}',
                              fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # GradCAM
        axes[0, 1].imshow(img, cmap='gray' if x.shape[1] == 1 else None, alpha=0.5)
        axes[0, 1].imshow(cam, cmap='jet', alpha=0.5)
        axes[0, 1].set_title('GradCAM\n模型关注区域', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # 显著图
        axes[1, 0].imshow(saliency, cmap='hot')
        axes[1, 0].set_title('显著图\n像素重要性', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        # 概率分布
        top_k = min(10, len(probs))
        top_probs, top_idx = probs.topk(top_k)
        labels = [class_names[i] if class_names else str(i) for i in top_idx.numpy()]
        axes[1, 1].barh(range(top_k), top_probs.numpy(), color='steelblue')
        axes[1, 1].set_yticks(range(top_k))
        axes[1, 1].set_yticklabels(labels)
        axes[1, 1].set_xlabel('概率')
        axes[1, 1].set_title('预测概率分布', fontsize=12, fontweight='bold')
        axes[1, 1].invert_yaxis()

        plt.suptitle('CNN 综合可视化分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('cnn_full_visualization.png', dpi=150)
        plt.show()

        return pred_class, probs


# 使用示例
def run_visualization_demo():
    model = SimpleCNN()
    model.eval()

    img = np.zeros((28, 28), dtype=np.float32)
    img[8:20, 8:20] = 1.0
    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

    visualizer = CNNVisualizer(model, last_conv_layer='conv3')
    visualizer.full_analysis(x)


run_visualization_demo()
```

---

## 小结

| 技术 | 回答的问题 | 输入 | 输出 |
|------|-----------|------|------|
| GradCAM | 模型在看哪里？ | 图像 + 类别 | 空间热力图 |
| 显著图 | 哪些像素最重要？ | 图像 + 类别 | 像素级敏感度 |
| SmoothGrad | 哪些像素最重要（稳定版）？ | 图像 + 噪声 | 平滑敏感度 |
| 特征反转 | 每层编码了什么？ | 特征向量 | 恢复的图像 |
| DeepDream | 模型学到了什么模式？ | 噪声/图像 | 放大的模式 |
