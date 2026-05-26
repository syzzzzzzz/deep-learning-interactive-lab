"""CNN 可解释性可视化：Grad-CAM、显著图、特征反转和 DeepDream。"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

MODULE_TITLE = "CNN 可视化与 Grad-CAM"
MODULE_SUMMARY = "用 Grad-CAM、像素显著图、特征反转和 DeepDream 解释 CNN 到底看到了什么。"
MODULE_TAGS = ["CNN", "Grad-CAM", "显著图", "可解释性", "DeepDream"]
MODULE_RELATED_TOPICS = ["part2/02_feature_maps", "part2/07_advanced_convolution", "part5/01_feature_visualization", "part5/02_gradient_monitor"]
PRACTICE_TARGET = "切换输入图案、目标类别、平滑次数和 Dream 强度，解释热力图、显著图和概率分布为什么变化。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    """
    自动生成自: part2_cnn\08_visualization_gradcam.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        from PIL import Image
    except Exception:
        Image = None

    from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
    from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure


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
                    module.register_full_backward_hook(self._save_gradient)
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


    # demo_gradcam()  # 协议化后由 compute_visualization_gradcam() 控制执行

    # ============================================================
    # 代码段 2
    # ============================================================

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


    # demo_saliency()  # 协议化后由 compute_visualization_gradcam() 控制执行

    # ============================================================
    # 代码段 3
    # ============================================================

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


    # demo_feature_inversion()  # 协议化后由 compute_visualization_gradcam() 控制执行

    # ============================================================
    # 代码段 4
    # ============================================================

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


    # demo_deepdream()  # 协议化后由 compute_visualization_gradcam() 控制执行

    # ============================================================
    # 代码段 5
    # ============================================================

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


    # run_visualization_demo()  # 协议化后由 compute_visualization_gradcam() 控制执行
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part2_cnn/08_visualization_gradcam.py", e)


def _make_visual_pattern(pattern: str, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((28, 28), dtype=np.float32)
    if pattern == "方块":
        image[8:20, 8:20] = 1.0
    elif pattern == "十字":
        image[12:16, :] = 1.0
        image[:, 12:16] = 1.0
    elif pattern == "圆环":
        yy, xx = np.mgrid[0:28, 0:28]
        dist = np.sqrt((yy - 14) ** 2 + (xx - 14) ** 2)
        image[(dist > 6) & (dist < 10)] = 1.0
    else:
        image[6:22, 12:16] = 1.0
        image[18:22, 8:20] = 0.85
    image += rng.normal(0, noise, image.shape).astype(np.float32)
    return np.clip(image, 0, 1)


def _smooth_grad(model: torch.nn.Module, x: torch.Tensor, target_class: int, samples: int, stdev: float, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    samples = clamp_int(samples, 1, 24, "SmoothGrad 次数")
    smooth = None
    for _ in range(samples):
        noisy_x = (x + torch.randn_like(x) * stdev).detach().requires_grad_(True)
        saliency, _ = compute_saliency_map(model, noisy_x, target_class)
        smooth = saliency if smooth is None else smooth + saliency
    return smooth / samples


def _safe_gradcam(model: torch.nn.Module, x: torch.Tensor, target_class: int | None) -> tuple[np.ndarray, int]:
    grad_cam = GradCAM(model, target_layer="conv3")
    cam, predicted = grad_cam.generate(x, target_class)
    if cam.ndim == 0:
        cam = np.zeros((28, 28), dtype=np.float32)
    cam = F.interpolate(torch.from_numpy(cam).float().view(1, 1, *cam.shape), size=(28, 28), mode="bilinear", align_corners=False)
    cam_np = cam.squeeze().numpy()
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    return cam_np, int(predicted)


def _light_feature_inversion(model: torch.nn.Module, x: torch.Tensor, layer_name: str, steps: int, lr: float) -> tuple[np.ndarray, list[float]]:
    steps = clamp_int(steps, 4, 40, "反转步数")
    generated, history = feature_inversion(model, x, layer_name, num_iter=steps, lr=lr)
    image = generated[0, 0].detach().numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image, [float(value) for value in history]


def _light_deepdream(model: torch.nn.Module, x: torch.Tensor, layer_name: str, strength: float, steps: int) -> np.ndarray:
    steps = clamp_int(steps, 4, 40, "Dream 步数")
    dreamed = deepdream(model, x, layer_name, num_iter=steps, lr=strength, clip=True)
    image = dreamed[0, 0].detach().numpy()
    return np.clip(image, 0, 1)


def _plot_explainability_panel(
    image: np.ndarray,
    cam: np.ndarray,
    saliency: np.ndarray,
    smooth_saliency: np.ndarray,
    probs: np.ndarray,
    predicted_class: int,
) -> object:
    with safe_mpl_figure(figsize=(11, 6.4)) as fig:
        axes = fig.subplots(2, 3)
        axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("输入图案", fontsize=10, fontweight="bold")
        axes[0, 1].imshow(image, cmap="gray", alpha=0.45)
        axes[0, 1].imshow(cam, cmap="jet", alpha=0.55)
        axes[0, 1].set_title("Grad-CAM\n关注区域", fontsize=10, fontweight="bold")
        axes[0, 2].imshow(saliency, cmap="hot")
        axes[0, 2].set_title("显著图\n像素敏感度", fontsize=10, fontweight="bold")
        axes[1, 0].imshow(smooth_saliency, cmap="hot")
        axes[1, 0].set_title("SmoothGrad\n降噪后显著图", fontsize=10, fontweight="bold")
        axes[1, 1].bar(range(len(probs)), probs, color="#00f0ff")
        axes[1, 1].set_title(f"预测概率\n类别 {predicted_class}", fontsize=10, fontweight="bold")
        axes[1, 1].set_xticks(range(len(probs)))
        axes[1, 2].imshow(image, cmap="gray", alpha=0.35)
        axes[1, 2].imshow(cam * smooth_saliency, cmap="viridis", alpha=0.75)
        axes[1, 2].set_title("CAM x SmoothGrad\n交叉验证", fontsize=10, fontweight="bold")
        for ax in axes.flat:
            if ax is not axes[1, 1]:
                ax.axis("off")
        fig.suptitle("CNN 可解释性面板：区域、像素和概率一起看", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig


def _plot_feature_dream_panel(original: np.ndarray, inversion: np.ndarray, dream: np.ndarray, history: list[float]) -> object:
    with safe_mpl_figure(figsize=(10.5, 3.8)) as fig:
        axes = fig.subplots(1, 4)
        axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("原图", fontsize=9, fontweight="bold")
        axes[1].imshow(inversion, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("特征反转", fontsize=9, fontweight="bold")
        axes[2].imshow(dream, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("DeepDream", fontsize=9, fontweight="bold")
        axes[3].plot(history, color="#00ff88", linewidth=2)
        axes[3].set_title("反转损失", fontsize=9, fontweight="bold")
        axes[3].set_xlabel("步数")
        axes[3].grid(True, alpha=0.25)
        for ax in axes[:3]:
            ax.axis("off")
        fig.suptitle("特征可视化：从“模型看哪里”到“模型想要什么”", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig


def compute_visualization_gradcam(
    pattern: str = "方块",
    target_class: int = -1,
    smooth_samples: int = 8,
    noise: float = 0.03,
    dream_strength: float = 0.04,
    optimization_steps: int = 12,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute lightweight CNN explainability visuals without running import-time demos."""

    if pattern not in {"方块", "十字", "圆环", "折线"}:
        raise ValueError("pattern 必须是 方块、十字、圆环 或 折线")
    target_class = clamp_int(int(target_class), -1, 9, "目标类别")
    smooth_samples = clamp_int(smooth_samples, 1, 24, "SmoothGrad 次数")
    noise = clamp_float(noise, 0.0, 0.35, "噪声强度")
    dream_strength = clamp_float(dream_strength, 0.005, 0.12, "Dream 强度")
    optimization_steps = clamp_int(optimization_steps, 4, 40, "优化步数")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleCNN()
    model.eval()
    image = _make_visual_pattern(pattern, noise, seed)
    x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().numpy()
    selected_class = None if target_class < 0 else target_class
    cam, predicted_class = _safe_gradcam(model, x, selected_class)
    class_for_grad = predicted_class if selected_class is None else selected_class
    saliency, _ = compute_saliency_map(model, x.clone().detach().requires_grad_(True), class_for_grad)
    smooth_saliency = _smooth_grad(model, x, class_for_grad, smooth_samples, stdev=max(noise, 0.03), seed=seed)
    inversion, inversion_history = _light_feature_inversion(model, x, "conv2", optimization_steps, lr=0.04)
    dream = _light_deepdream(model, x, "conv2", dream_strength, optimization_steps)

    panel_fig = _plot_explainability_panel(image, cam, saliency, smooth_saliency, probs, predicted_class)
    feature_fig = _plot_feature_dream_panel(image, inversion, dream, inversion_history)
    figures = [
        ("visualization_gradcam_panel.png", panel_fig),
        ("visualization_feature_dream.png", feature_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        print("CNN 可解释性协议化计算")
        print(f"输入图案={pattern}, 目标类别={'预测类别' if target_class < 0 else target_class}, SmoothGrad={smooth_samples}, noise={noise:.2f}")
        print(f"预测类别={predicted_class}, 置信度={probs[predicted_class]:.3f}")
        print(f"Grad-CAM 均值={cam.mean():.3f}, 显著图均值={saliency.mean():.3f}, Dream 强度={dream_strength:.3f}")
        print("解释建议：Grad-CAM 看区域，显著图看像素敏感度，特征反转/DeepDream 看某层偏好的模式。")
    stats = {
        "predicted_class": int(predicted_class),
        "confidence": float(probs[predicted_class]),
        "cam_mean": float(cam.mean()),
        "saliency_mean": float(saliency.mean()),
        "smooth_saliency_mean": float(smooth_saliency.mean()),
        "inversion_final_loss": float(inversion_history[-1]) if inversion_history else 0.0,
    }
    return {"figures": figures, "artifacts": artifacts, "stats": stats, "log": log_buffer.getvalue()}


def _go_to_feature_visualization() -> None:
    import streamlit as st

    st.query_params["module"] = "part5_toolbox/01_feature_visualization"
    st.rerun()


def render() -> None:
    """Render the Grad-CAM and CNN explainability lesson."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import render_attention_light_beams, render_loading_bar, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="expanded")
        render_visual_system("dark")
        st.link_button("返回主界面", "/", width="small")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("正在生成 Grad-CAM、显著图、SmoothGrad、特征反转和 DeepDream")
        with st.sidebar:
            pattern = st.selectbox("输入图案", ["方块", "十字", "圆环", "折线"])
            target_label = st.selectbox("目标类别", ["使用预测类别"] + [str(i) for i in range(10)])
            target_class = -1 if target_label == "使用预测类别" else int(target_label)
            smooth_samples = st.slider("SmoothGrad 次数", 1, 24, 8, 1)
            noise = st.slider("噪声强度", 0.0, 0.35, 0.03, 0.01)
            dream_strength = st.slider("Dream 强度", 0.005, 0.12, 0.04, 0.005)
            optimization_steps = st.slider("优化步数", 4, 40, 12, 1)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("去实战：特征可视化工具箱", width="stretch"):
                _go_to_feature_visualization()
        data = compute_visualization_gradcam(pattern, target_class, smooth_samples, noise, dream_strength, optimization_steps, int(seed), save_artifacts=True)
        stats = data["stats"]
        render_attention_light_beams()
        st.markdown(
            """
            **零基础直觉：**可解释性不是让模型“开口说话”，而是用几种探针去观察它。
            Grad-CAM 看它大概盯着哪个区域，显著图看哪个像素一改就影响结果，特征反转和 DeepDream 则反过来问：
            某一层最想看到什么样的图案？
            """
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("预测类别", str(stats["predicted_class"]))
        c2.metric("置信度", f"{stats['confidence']:.1%}")
        c3.metric("CAM 均值", f"{stats['cam_mean']:.3f}")
        c4.metric("显著图均值", f"{stats['saliency_mean']:.3f}")
        explainers = [
            ("综合可解释性面板", "Grad-CAM 给区域级证据，显著图给像素级证据，SmoothGrad 用多次噪声平均让显著图更稳定。"),
            ("特征反转与 DeepDream", "特征反转试图还原能产生类似特征的图；DeepDream 则把某层喜欢的模式不断放大。"),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"图像产物已放入统一目录：{get_artifact_path(filename)}")
            st.markdown("> 请切换输入图案或目标类别，观察 Grad-CAM 和显著图是否跟着移动。思考：权重热区是解释线索，还是完整因果证明？")
        with st.expander("常见误区与控制台输出", expanded=False):
            st.markdown(
                """
                - **误区 1：热力图越红就一定是因果解释。** 正确理解：它是线索，不是完整证明。
                - **误区 2：显著图噪声多说明模型没学会。** 正确理解：梯度本来就敏感，SmoothGrad 是常见降噪办法。
                - **误区 3：DeepDream 是真实图片生成。** 正确理解：它是在放大某层偏好的模式，更像模型偏好的可视化。
                """
            )
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part2_cnn/08_visualization_gradcam.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_visualization_gradcam(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_visualization_gradcam(pattern="方块", smooth_samples=2, optimization_steps=4, seed=7, save_artifacts=False)
    return bool(data["figures"]) and 0 <= data["stats"]["predicted_class"] <= 9 and data["stats"]["confidence"] > 0


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_visualization_gradcam))
