"""MNIST 玩具实验：用轻量模拟和真实前向检查理解 CNN 分类训练闭环。"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

MODULE_TITLE = "MNIST 玩具实验"
MODULE_SUMMARY = "用训练曲线、混淆矩阵、预测样例和逐层激活解释手写数字分类从输入到输出的完整流程。"
MODULE_TAGS = ["CNN", "MNIST", "训练曲线", "混淆矩阵", "调试"]
MODULE_RELATED_TOPICS = ["part2/03_classic_architectures", "part2/04_debug_panel", "part5/03_training_dynamics", "part5/data_training"]
PRACTICE_TARGET = "切换模型、训练轮数、学习率和噪声强度，解释准确率、损失、混淆矩阵和激活图为什么变化。"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    """
    自动生成自: part2_cnn\05_mnist_toy.md
    可独立运行的 Python 源码
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt
    from components.lesson_runtime import clamp_float, clamp_int, run_cli, running_under_streamlit
    from components.resource_manager import clean_old_artifacts, get_artifact_path, safe_mpl_figure

    # ─────────────────────────────────────────────────────────
    # LeNet-5（1998）：第一个成功的 CNN
    # ─────────────────────────────────────────────────────────

    class LeNet5(nn.Module):
        """
        LeNet-5 原始架构
    # 输入: 32×32 灰度图
    # 输出: 10类
        """
        def __init__(self, num_classes=10):
            super().__init__()
            # 特征提取
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5),    # 32→28, 6通道
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),        # 28→14
                nn.Conv2d(6, 16, kernel_size=5),   # 14→10, 16通道
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=2),        # 10→5
            )
            # 分类器
            self.classifier = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.Tanh(),
                nn.Linear(120, 84),
                nn.Tanh(),
                nn.Linear(84, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x


    # ─────────────────────────────────────────────────────────
    # 简化版 VGG（2014）：深而窄的 3×3 卷积
    # ─────────────────────────────────────────────────────────

    class MiniVGG(nn.Module):
        """
        VGG 的核心思想：用多个 3×3 卷积替代大卷积核
        两个 3×3 卷积 = 一个 5×5 卷积的感受野，但参数更少
        """
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: 2个 3×3 卷积
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),  # 28→14

                # Block 2: 2个 3×3 卷积
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),  # 14→7
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(64 * 7 * 7, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)


    # ─────────────────────────────────────────────────────────
    # 残差网络（ResNet）：跳跃连接解决梯度消失
    # ─────────────────────────────────────────────────────────

    class ResidualBlock(nn.Module):
        """
        残差块：y = F(x) + x
        关键思想：学习残差 F(x) = H(x) - x，而不是直接学习 H(x)
        """
        def __init__(self, channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

            # 如果步幅不为1，需要调整 shortcut 的维度
            self.shortcut = nn.Identity()
            if stride != 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(channels, channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(channels)
                )

        def forward(self, x):
            identity = self.shortcut(x)

            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))

            out = out + identity  # 残差连接！
            out = F.relu(out)
            return out

    class MiniResNet(nn.Module):
        """简化版 ResNet，适合 MNIST"""
        def __init__(self, num_classes=10):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
            self.layer1 = nn.Sequential(
                ResidualBlock(32),
                ResidualBlock(32),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                ResidualBlock(64),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)


    # ─────────────────────────────────────────────────────────
    # 网络架构对比
    # ─────────────────────────────────────────────────────────

    def compare_architectures():
        """对比不同网络的参数量和计算量"""
        models = {
            'LeNet-5': LeNet5(),
            'MiniVGG': MiniVGG(),
            'MiniResNet': MiniResNet(),
        }

        print("=" * 60)
        print("网络架构对比")
        print("=" * 60)
        print(f"{'模型':15s} {'参数量':12s} {'输入':15s} {'输出':10s}")
        print("-" * 60)

        for name, model in models.items():
            n_params = sum(p.numel() for p in model.parameters())
            # 测试前向传播
            if name == 'LeNet-5':
                x = torch.randn(1, 1, 32, 32)
            else:
                x = torch.randn(1, 1, 28, 28)
            with torch.no_grad():
                out = model(x)
            print(f"{name:15s} {n_params:12,d} {str(tuple(x.shape)):15s} {str(tuple(out.shape)):10s}")

    # compare_architectures()  # 协议化后由 compute_mnist_toy() 控制执行

    # ============================================================
    # 代码段 2
    # ============================================================

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    try:
        import torchvision
        import torchvision.transforms as transforms
    except Exception:
        torchvision = None
        transforms = None
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from collections import defaultdict

    # ─────────────────────────────────────────────────────────
    # 完整训练框架
    # ─────────────────────────────────────────────────────────

    class MNISTTrainer:
        """
        MNIST 手写数字分类完整训练框架
        支持：多种模型、实时可视化、调试面板
        """

        def __init__(
            self,
            model_name: str = 'MiniResNet',
            batch_size: int = 64,
            lr: float = 0.001,
            epochs: int = 10,
            device: str = None,
        ):
            self.model_name = model_name
            self.batch_size = batch_size
            self.lr = lr
            self.epochs = epochs
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

            # 训练历史
            self.history = defaultdict(list)

            # 初始化
            self._setup_data()
            self._setup_model()

        def _setup_data(self):
            """加载 MNIST 数据集"""
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            self.test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )

            self.train_loader = DataLoader(
                self.train_dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=0, pin_memory=True
            )
            self.test_loader = DataLoader(
                self.test_dataset, batch_size=256,
                shuffle=False, num_workers=0
            )

            print(f"训练集: {len(self.train_dataset)} 样本")
            print(f"测试集: {len(self.test_dataset)} 样本")

        def _setup_model(self):
            """初始化模型"""
            model_map = {
                'LeNet5': LeNet5,
                'MiniVGG': MiniVGG,
                'MiniResNet': MiniResNet,
            }

            if self.model_name not in model_map:
                raise ValueError(f"未知模型: {self.model_name}. 可选: {list(model_map.keys())}")

            self.model = model_map[self.model_name]().to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
            self.criterion = nn.CrossEntropyLoss()

            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"\n模型: {self.model_name}")
            print(f"参数量: {n_params:,}")
            print(f"设备: {self.device}")

        def train_epoch(self, epoch: int) -> dict:
            """训练一个 epoch"""
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            start_time = time.time()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # LeNet5 需要 32×32 输入
                if self.model_name == 'LeNet5':
                    data = F.interpolate(data, size=(32, 32))

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                if batch_idx % 100 == 0:
                    print(f"\r  Epoch {epoch} [{batch_idx*len(data)}/{len(self.train_loader.dataset)}] "
                          f"Loss: {loss.item():.4f}", end='')

            elapsed = time.time() - start_time
            return {
                'loss': total_loss / len(self.train_loader),
                'acc': correct / total,
                'time': elapsed,
            }

        def evaluate(self) -> dict:
            """在测试集上评估"""
            self.model.eval()
            total_loss = 0
            correct = 0
            total = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    if self.model_name == 'LeNet5':
                        data = F.interpolate(data, size=(32, 32))

                    output = self.model(data)
                    loss = self.criterion(output, target)

                    total_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

            return {
                'loss': total_loss / len(self.test_loader),
                'acc': correct / total,
                'preds': np.array(all_preds),
                'targets': np.array(all_targets),
            }

        def train(self):
            """完整训练流程"""
            print(f"\n开始训练 {self.model_name}...")
            print("=" * 60)

            best_acc = 0
            for epoch in range(1, self.epochs + 1):
                train_metrics = self.train_epoch(epoch)
                test_metrics = self.evaluate()
                self.scheduler.step()

                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics['acc'])
                self.history['test_loss'].append(test_metrics['loss'])
                self.history['test_acc'].append(test_metrics['acc'])
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

                if test_metrics['acc'] > best_acc:
                    best_acc = test_metrics['acc']
                    torch.save(self.model.state_dict(), f'best_{self.model_name}.pth')

                print(f"\nEpoch {epoch:3d}/{self.epochs}: "
                      f"Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['acc']:.4f} | "
                      f"Test Loss={test_metrics['loss']:.4f} Acc={test_metrics['acc']:.4f} | "
                      f"LR={self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Time={train_metrics['time']:.1f}s")

            print(f"\n最佳测试准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
            return test_metrics

        def plot_training_history(self):
            """绘制训练历史"""
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            epochs = range(1, len(self.history['train_loss']) + 1)

            # 损失曲线
            axes[0].plot(epochs, self.history['train_loss'], 'b-o', markersize=4, label='训练')
            axes[0].plot(epochs, self.history['test_loss'], 'r-o', markersize=4, label='测试')
            axes[0].set_title('损失曲线', fontsize=12)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 准确率曲线
            axes[1].plot(epochs, [a*100 for a in self.history['train_acc']],
                         'b-o', markersize=4, label='训练')
            axes[1].plot(epochs, [a*100 for a in self.history['test_acc']],
                         'r-o', markersize=4, label='测试')
            axes[1].set_title('准确率曲线', fontsize=12)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('准确率 (%)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 100)

            # 学习率曲线
            axes[2].plot(epochs, self.history['lr'], 'g-o', markersize=4)
            axes[2].set_title('学习率变化', fontsize=12)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('学习率')
            axes[2].grid(True, alpha=0.3)

            plt.suptitle(f'{self.model_name} 训练历史', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'training_history_{self.model_name}.png', dpi=150, bbox_inches='tight')
            plt.show()

        def plot_confusion_matrix(self, test_metrics: dict):
            """绘制混淆矩阵"""
            from sklearn.metrics import confusion_matrix
            import seaborn as sns

            cm = confusion_matrix(test_metrics['targets'], test_metrics['preds'])
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 原始混淆矩阵
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=range(10), yticklabels=range(10), ax=axes[0])
            axes[0].set_title('混淆矩阵（原始计数）', fontsize=12)
            axes[0].set_xlabel('预测类别')
            axes[0].set_ylabel('真实类别')

            # 归一化混淆矩阵
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=range(10), yticklabels=range(10), ax=axes[1])
            axes[1].set_title('混淆矩阵（归一化）', fontsize=12)
            axes[1].set_xlabel('预测类别')
            axes[1].set_ylabel('真实类别')

            plt.suptitle(f'{self.model_name} 混淆矩阵', fontsize=13)
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{self.model_name}.png', dpi=150, bbox_inches='tight')
            plt.show()

        def visualize_predictions(self, n_samples: int = 20):
            """可视化预测结果"""
            self.model.eval()
            images, labels = next(iter(self.test_loader))
            images = images[:n_samples]
            labels = labels[:n_samples]

            if self.model_name == 'LeNet5':
                inputs = F.interpolate(images.to(self.device), size=(32, 32))
            else:
                inputs = images.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

            n_cols = 10
            n_rows = (n_samples + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 2))
            axes = axes.flatten()

            for i in range(n_samples):
                img = images[i, 0].numpy()
                pred = preds[i].item()
                true = labels[i].item()
                conf = probs[i, pred].item()

                axes[i].imshow(img, cmap='gray')
                color = 'green' if pred == true else 'red'
                axes[i].set_title(f'预测:{pred}\n真实:{true}\n{conf:.0%}',
                                   fontsize=7, color=color)
                axes[i].axis('off')

            for i in range(n_samples, len(axes)):
                axes[i].axis('off')

            plt.suptitle('预测结果（绿色=正确，红色=错误）', fontsize=12)
            plt.tight_layout()
            plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
            plt.show()

        def interactive_debug(self):
            """
            交互式调试面板
            输入一张图片，查看每层的激活
            """
            self.model.eval()
            images, labels = next(iter(self.test_loader))
            img = images[0:1].to(self.device)
            true_label = labels[0].item()

            if self.model_name == 'LeNet5':
                img_input = F.interpolate(img, size=(32, 32))
            else:
                img_input = img

            # 注册钩子
            activations = {}
            hooks = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
                    hook = module.register_forward_hook(
                        lambda m, inp, out, n=name: activations.update({n: out.detach().cpu()})
                    )
                    hooks.append(hook)

            with torch.no_grad():
                output = self.model(img_input)
                probs = torch.softmax(output, dim=1)[0]
                pred = output.argmax(dim=1).item()

            # 移除钩子
            for hook in hooks:
                hook.remove()

            # 可视化
            fig = plt.figure(figsize=(20, 10))

            # 输入图像
            ax_input = fig.add_subplot(2, 6, 1)
            ax_input.imshow(img[0, 0].cpu().numpy(), cmap='gray')
            ax_input.set_title(f'输入\n真实:{true_label} 预测:{pred}', fontsize=10)
            ax_input.axis('off')

            # 预测概率
            ax_prob = fig.add_subplot(2, 6, 2)
            bars = ax_prob.bar(range(10), probs.cpu().numpy(),
                                color=['green' if i == true_label else
                                       'red' if i == pred else 'steelblue'
                                       for i in range(10)])
            ax_prob.set_title('预测概率', fontsize=10)
            ax_prob.set_xlabel('类别')
            ax_prob.set_xticks(range(10))

            # 各层激活
            conv_acts = {k: v for k, v in activations.items() if len(v.shape) == 4}
            for idx, (name, act) in enumerate(list(conv_acts.items())[:10]):
                ax = fig.add_subplot(2, 6, idx + 3)
                # 显示通道平均激活
                avg_act = act[0].mean(0).numpy()
                ax.imshow(avg_act, cmap='viridis', aspect='equal')
                ax.set_title(f'{name}\n{tuple(act.shape[1:])}', fontsize=7)
                ax.axis('off')

            plt.suptitle(f'{self.model_name} 交互式调试面板', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig('debug_panel.png', dpi=150, bbox_inches='tight')
            plt.show()

            print(f"\n预测结果: {pred} (真实: {true_label})")
            print(f"置信度: {probs[pred].item():.4f}")
            print(f"\n各类别概率:")
            for i, p in enumerate(probs):
                bar = '█' * int(p.item() * 30)
                print(f"  {i}: {bar:30s} {p.item():.4f}")


    # ─────────────────────────────────────────────────────────
    # 运行完整实验
    # ─────────────────────────────────────────────────────────

    def run_mnist_experiment():
        """运行完整的 MNIST 实验"""
        print("MNIST 手写数字分类实验")
        print("=" * 60)

        # 可以修改这些参数来实验！
        trainer = MNISTTrainer(
            model_name='MiniResNet',  # 'LeNet5', 'MiniVGG', 'MiniResNet'
            batch_size=64,
            lr=0.001,
            epochs=5,  # 快速演示用5个epoch
        )

        # 训练
        test_metrics = trainer.train()

        # 可视化结果
        trainer.plot_training_history()
        trainer.visualize_predictions()
        trainer.interactive_debug()

        try:
            trainer.plot_confusion_matrix(test_metrics)
        except ImportError:
            print("提示: pip install scikit-learn seaborn 以查看混淆矩阵")

        return trainer

    # trainer = run_mnist_experiment()
    # print("MNIST 玩具已准备好！取消注释最后一行来运行。")
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part2_cnn/05_mnist_toy.py", e)


def _model_registry() -> dict[str, torch.nn.Module]:
    return {
        "LeNet5": LeNet5(),
        "MiniVGG": MiniVGG(),
        "MiniResNet": MiniResNet(),
    }


def _make_digit_like_images(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = clamp_int(n_samples, 10, 80, "样本数")
    noise = clamp_float(noise, 0.0, 0.6, "噪声强度")
    images = np.zeros((n_samples, 28, 28), dtype=np.float32)
    labels = np.arange(n_samples, dtype=np.int64) % 10
    yy, xx = np.mgrid[0:28, 0:28]
    centers = {
        0: [(14, 14, 8, 0.95), (14, 14, 4, -0.65)],
        1: [(14, 8, 3, 0.9), (14, 15, 4, 0.75)],
        2: [(9, 14, 5, 0.9), (19, 14, 6, 0.75)],
        3: [(9, 15, 5, 0.9), (19, 15, 5, 0.85)],
        4: [(13, 8, 4, 0.8), (13, 19, 4, 0.8), (18, 14, 5, 0.75)],
        5: [(8, 13, 5, 0.8), (18, 14, 6, 0.85)],
        6: [(16, 13, 8, 0.9), (12, 16, 4, -0.45)],
        7: [(8, 14, 7, 0.85), (18, 18, 4, 0.65)],
        8: [(10, 14, 5, 0.9), (19, 14, 5, 0.9), (14, 14, 3, -0.35)],
        9: [(12, 14, 7, 0.9), (20, 12, 4, 0.55)],
    }
    for idx, label in enumerate(labels):
        canvas = np.zeros((28, 28), dtype=np.float32)
        for cy, cx, radius, value in centers[int(label)]:
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius))
            canvas += value * blob
        canvas += rng.normal(0, noise, size=canvas.shape).astype(np.float32)
        images[idx] = np.clip(canvas, 0, 1)
    return images, labels


def _simulate_mnist_training(model_name: str, epochs: int, learning_rate: float, noise: float, seed: int) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed + 101)
    axis = np.arange(1, epochs + 1)
    architecture_bonus = {"LeNet5": 0.00, "MiniVGG": 0.045, "MiniResNet": 0.065}[model_name]
    lr_penalty = abs(np.log10(learning_rate) - np.log10(0.001)) * 0.055
    noise_penalty = noise * 0.22
    target_acc = np.clip(0.86 + architecture_bonus - lr_penalty - noise_penalty, 0.58, 0.985)
    speed = np.clip(learning_rate * 1800, 0.25, 4.0)
    train_acc = target_acc - 0.34 * np.exp(-axis / max(epochs, 1) * speed * 2.4)
    test_acc = target_acc - 0.27 * np.exp(-axis / max(epochs, 1) * speed * 1.85)
    if learning_rate > 0.015:
        wobble = np.sin(axis * 0.9) * (learning_rate - 0.015) * 3.5
        test_acc -= np.abs(wobble)
    train_acc += rng.normal(0, 0.006, epochs)
    test_acc += rng.normal(0, 0.008, epochs)
    train_acc = np.clip(train_acc, 0.25, 0.995)
    test_acc = np.clip(test_acc, 0.20, 0.99)
    train_loss = np.clip(1.45 - train_acc + rng.normal(0, 0.006, epochs), 0.015, 1.8)
    test_loss = np.clip(1.42 - test_acc + rng.normal(0, 0.008, epochs), 0.02, 1.8)
    lr_curve = learning_rate * (0.5 * (1 + np.cos(np.linspace(0, np.pi, epochs))))
    return {
        "epochs": axis,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "lr_curve": lr_curve,
        "final_test_acc": float(test_acc[-1]),
        "final_test_loss": float(test_loss[-1]),
    }


def _plot_training_history(history: dict[str, np.ndarray | float], model_name: str) -> object:
    with safe_mpl_figure(figsize=(11, 3.9)) as fig:
        axes = fig.subplots(1, 3)
        axes[0].plot(history["epochs"], history["train_loss"], "o-", color="#00f0ff", label="训练")
        axes[0].plot(history["epochs"], history["test_loss"], "o-", color="#bf3f5b", label="测试")
        axes[0].set_title("损失曲线", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=8)
        axes[1].plot(history["epochs"], np.asarray(history["train_acc"]) * 100, "o-", color="#b000ff", label="训练")
        axes[1].plot(history["epochs"], np.asarray(history["test_acc"]) * 100, "o-", color="#00ff88", label="测试")
        axes[1].set_title("准确率曲线", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("%")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8)
        axes[2].plot(history["epochs"], history["lr_curve"], "o-", color="#00ff88")
        axes[2].set_title("余弦学习率", fontsize=10, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.25)
        fig.suptitle(f"{model_name} MNIST 教学训练曲线", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig


def _plot_prediction_grid(images: np.ndarray, labels: np.ndarray, final_acc: float, seed: int) -> tuple[object, dict[str, int]]:
    rng = np.random.default_rng(seed + 202)
    n_samples = min(20, len(images))
    correct_count = int(round(n_samples * final_acc))
    preds = labels[:n_samples].copy()
    wrong_indices = rng.choice(n_samples, size=max(n_samples - correct_count, 0), replace=False)
    for idx in wrong_indices:
        preds[idx] = (preds[idx] + int(rng.integers(1, 10))) % 10
    with safe_mpl_figure(figsize=(10, 4.2)) as fig:
        axes = fig.subplots(2, 10)
        axes = axes.flatten()
        for i in range(n_samples):
            axes[i].imshow(images[i], cmap="gray", vmin=0, vmax=1)
            color = "#00ff88" if preds[i] == labels[i] else "#bf3f5b"
            axes[i].set_title(f"预测:{preds[i]}\n真实:{labels[i]}", fontsize=7, color=color, fontweight="bold")
            axes[i].axis("off")
        for ax in axes[n_samples:]:
            ax.axis("off")
        fig.suptitle("预测样例：绿色代表正确，红色代表错误", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig, {"shown_samples": n_samples, "shown_correct": int((preds == labels[:n_samples]).sum())}


def _plot_confusion_matrix(final_acc: float, seed: int) -> tuple[object, np.ndarray]:
    rng = np.random.default_rng(seed + 303)
    matrix = np.zeros((10, 10), dtype=int)
    per_class = 24
    correct = int(round(per_class * final_acc))
    for digit in range(10):
        matrix[digit, digit] = correct
        remaining = per_class - correct
        for _ in range(max(remaining, 0)):
            target = int((digit + rng.integers(1, 10)) % 10)
            matrix[digit, target] += 1
    with safe_mpl_figure(figsize=(6, 5.2)) as fig:
        ax = fig.subplots(1, 1)
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title("混淆矩阵：看模型把哪个数字认错成哪个", fontsize=10, fontweight="bold")
        for i in range(10):
            for j in range(10):
                if matrix[i, j] > 0:
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
        fig.tight_layout()
        return fig, matrix


def _plot_activation_debug(model_name: str, image: np.ndarray) -> tuple[object, tuple[int, ...]]:
    models = _model_registry()
    model = models[model_name]
    model.eval()
    x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    if model_name == "LeNet5":
        x = F.interpolate(x, size=(32, 32))
    activations: dict[str, torch.Tensor] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(lambda _m, _inp, out, n=name: activations.update({n: out.detach().cpu()})))
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    for hook in hooks:
        hook.remove()
    with safe_mpl_figure(figsize=(10, 4.5)) as fig:
        axes = fig.subplots(2, 5)
        axes = axes.flatten()
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("输入", fontsize=8, fontweight="bold")
        axes[0].axis("off")
        axes[1].bar(range(10), probs, color="#00f0ff")
        axes[1].set_title("预测概率", fontsize=8, fontweight="bold")
        axes[1].set_xticks(range(10))
        for ax, (name, activation) in zip(axes[2:], list(activations.items())[:8]):
            avg = activation[0].mean(0).numpy()
            ax.imshow(avg, cmap="viridis")
            ax.set_title(f"{name}\n{tuple(activation.shape[1:])}", fontsize=7)
            ax.axis("off")
        for ax in axes[2 + len(activations):]:
            ax.axis("off")
        fig.suptitle("模型调试面板：输入、概率和卷积层平均激活", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig, tuple(logits.shape)


def compute_mnist_toy(
    model_name: str = "MiniResNet",
    epochs: int = 8,
    learning_rate: float = 0.001,
    noise: float = 0.12,
    sample_count: int = 40,
    seed: int = 42,
    save_artifacts: bool = False,
) -> dict[str, object]:
    """Compute a lightweight MNIST teaching experiment without downloading data."""

    if model_name not in {"LeNet5", "MiniVGG", "MiniResNet"}:
        raise ValueError("model_name 必须是 LeNet5、MiniVGG 或 MiniResNet")
    epochs = clamp_int(epochs, 2, 80, "训练轮数")
    learning_rate = clamp_float(learning_rate, 0.0001, 0.05, "学习率")
    noise = clamp_float(noise, 0.0, 0.6, "噪声强度")
    sample_count = clamp_int(sample_count, 10, 80, "样本数")
    torch.manual_seed(seed)
    np.random.seed(seed)
    images, labels = _make_digit_like_images(sample_count, noise, seed)
    history = _simulate_mnist_training(model_name, epochs, learning_rate, noise, seed)
    training_fig = _plot_training_history(history, model_name)
    prediction_fig, prediction_stats = _plot_prediction_grid(images, labels, float(history["final_test_acc"]), seed)
    confusion_fig, matrix = _plot_confusion_matrix(float(history["final_test_acc"]), seed)
    activation_fig, output_shape = _plot_activation_debug(model_name, images[0])
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        print("MNIST 玩具协议化计算")
        print(f"模型={model_name}, epochs={epochs}, lr={learning_rate:.4f}, noise={noise:.2f}, sample_count={sample_count}")
        print(f"最终测试准确率={history['final_test_acc']:.3f}, 最终测试损失={history['final_test_loss']:.3f}, 前向输出={output_shape}")
        print(f"展示样例正确数={prediction_stats['shown_correct']}/{prediction_stats['shown_samples']}")
        print("说明：本页默认使用轻量教学模拟，不下载 MNIST；真实训练框架仍保留在源码中，可在本地扩展运行。")
    figures = [
        ("mnist_toy_training_history.png", training_fig),
        ("mnist_toy_predictions.png", prediction_fig),
        ("mnist_toy_confusion_matrix.png", confusion_fig),
        ("mnist_toy_activation_debug.png", activation_fig),
    ]
    artifacts: list[Path] = []
    if save_artifacts:
        for filename, fig in figures:
            path = get_artifact_path(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            artifacts.append(path)
    stats = {
        "final_test_acc": float(history["final_test_acc"]),
        "final_test_loss": float(history["final_test_loss"]),
        "output_shape": output_shape,
        "confusion_trace": int(np.trace(matrix)),
        **prediction_stats,
    }
    return {"figures": figures, "artifacts": artifacts, "stats": stats, "history": history, "log": log_buffer.getvalue()}


def _go_to_training_dynamics() -> None:
    import streamlit as st

    st.query_params["module"] = "part5_toolbox/03_training_dynamics"
    st.rerun()


def render() -> None:
    """Render the MNIST toy teaching experiment."""

    import streamlit as st
    from components.error_boundary import render_module_error
    from components.visual_system import render_loading_bar, render_training_dashboard_gauges, render_visual_system

    try:
        clean_old_artifacts()
        st.set_page_config(page_title=MODULE_TITLE, layout="wide", initial_sidebar_state="auto")
        render_visual_system("light")
        st.link_button("返回主界面", "/", width="content")
        st.title(MODULE_TITLE)
        st.caption(MODULE_SUMMARY)
        render_loading_bar("正在生成 MNIST 教学训练闭环：曲线、混淆矩阵、预测样例和激活图")
        with st.sidebar:
            model_name = st.selectbox("模型", ["LeNet5", "MiniVGG", "MiniResNet"], index=2)
            epochs = st.slider("训练轮数", 2, 80, 8, 1)
            learning_rate = st.slider("学习率", 0.0001, 0.05, 0.001, 0.0001, format="%.4f")
            noise = st.slider("噪声强度", 0.0, 0.6, 0.12, 0.02)
            sample_count = st.slider("样本数", 10, 80, 40, 5)
            seed = st.number_input("随机种子", 0, 9999, 42, 1)
            if st.button("去实战：训练动态", width="stretch"):
                _go_to_training_dynamics()
        data = compute_mnist_toy(model_name, epochs, learning_rate, noise, sample_count, int(seed), save_artifacts=True)
        stats = data["stats"]
        render_training_dashboard_gauges()
        st.markdown(
            """
            **零基础直觉：**MNIST 是深度学习里的“九九乘法表”：任务简单，但训练闭环完整。
            你能在这里看到模型如何从图片得到概率、如何用损失改参数、如何用混淆矩阵发现它最容易把哪个数字认错。
            """
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("测试准确率", f"{stats['final_test_acc']:.1%}")
        c2.metric("测试损失", f"{stats['final_test_loss']:.3f}")
        c3.metric("前向输出", str(stats["output_shape"]))
        c4.metric("样例正确", f"{stats['shown_correct']}/{stats['shown_samples']}")
        explainers = [
            ("训练历史", "损失下降表示模型在减少错误；测试准确率同步上升才说明它不是只记住训练样本。"),
            ("预测样例", "绿色代表预测正确，红色代表预测错误。噪声越大，数字边界越不清楚，错误会增加。"),
            ("混淆矩阵", "对角线越亮越好；非对角线亮，表示模型常把某个真实数字误判成另一个数字。"),
            ("调试面板", "输入图、概率条和卷积激活放在一起看，可以判断模型到底看到了哪些局部结构。"),
        ]
        for (filename, fig), (title, body) in zip(data["figures"], explainers):
            st.subheader(title)
            st.write(body)
            st.pyplot(fig, clear_figure=False)
            st.caption(f"图像产物已放入统一目录：{get_artifact_path(filename)}")
            st.markdown("> 请只改一个参数，再观察曲线和混淆矩阵。思考：变化来自模型结构、学习率，还是输入噪声？")
        with st.expander("控制台输出与工程说明", expanded=False):
            st.markdown(
                """
                - 本页为了流畅教学默认不下载真实 MNIST，不跑重训练。
                - 源码中仍保留 `MNISTTrainer`，可以作为真实训练扩展入口。
                - 真训练时优先检查数据归一化、学习率、混淆矩阵和过拟合差距。
                """
            )
            st.code(str(data["log"])[-12000:], language="text")
    except Exception as exc:
        render_module_error("part2_cnn/05_mnist_toy.py", exc)


def compute(seed: int = 42) -> dict[str, object]:
    """Backward-compatible compute entry used by generic runners."""

    return compute_mnist_toy(seed=seed, save_artifacts=False)


def smoke() -> bool:
    """Lightweight self-check used by quality gates."""

    data = compute_mnist_toy(model_name="LeNet5", epochs=2, sample_count=10, seed=7, save_artifacts=False)
    return bool(data["figures"]) and data["stats"]["output_shape"] == (1, 10) and data["stats"]["final_test_acc"] > 0


if __name__ == "__main__":
    if running_under_streamlit():
        render()
    else:
        raise SystemExit(run_cli(compute_mnist_toy))
