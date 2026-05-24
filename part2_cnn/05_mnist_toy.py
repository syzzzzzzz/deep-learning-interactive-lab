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

    compare_architectures()

    # ============================================================
    # 代码段 2
    # ============================================================

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
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
    print("MNIST 玩具已准备好！取消注释最后一行来运行。")
    print("可以修改的参数：")
    print("  model_name: 'LeNet5', 'MiniVGG', 'MiniResNet'")
    print("  batch_size: 16, 32, 64, 128, 256")
    print("  lr: 0.0001, 0.001, 0.01, 0.1")
    print("  epochs: 1-100")
except Exception as e:
    from components.error_boundary import render_module_error

    render_module_error("part2_cnn/05_mnist_toy.py", e)
