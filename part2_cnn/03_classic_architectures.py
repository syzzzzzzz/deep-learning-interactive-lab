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
    from torchinfo import summary  # pip install torchinfo

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


    lenet = LeNet5()
    lenet.feature_sizes()
    print(f"\nLeNet-5 参数量: {sum(p.numel() for p in lenet.parameters()):,}")

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


    alexnet = MiniAlexNet(in_channels=1, num_classes=10)
    x = torch.randn(1, 1, 32, 32)
    print(f"MiniAlexNet 输出: {alexnet(x).shape}")
    print(f"参数量: {sum(p.numel() for p in alexnet.parameters()):,}")

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


    vgg = MiniVGG(in_channels=1, num_classes=10)
    print(f"MiniVGG 参数量: {sum(p.numel() for p in vgg.parameters()):,}")

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


    resnet = MiniResNet(in_channels=1, num_classes=10)
    x = torch.randn(1, 1, 32, 32)
    print(f"MiniResNet 输出: {resnet(x).shape}")
    print(f"参数量: {sum(p.numel() for p in resnet.parameters()):,}")

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

    results = compare_architectures()

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

    visualize_gradient_flow_resnet()
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part2_cnn/03_classic_architectures.py", e)
