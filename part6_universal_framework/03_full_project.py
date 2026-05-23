"""
自动生成自: part6_universal_framework\03_full_project.md
可独立运行的 Python 源码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from typing import Dict, Any, Optional, Callable
import time
import json
import os


def print_learning_guide():
    print("""
学习导读：完整项目骨架要解决的不是“能不能跑一次”，而是“下个月还能不能复现实验”。

1. UniversalTrainer 在管什么
   - _run_epoch 统一训练和验证流程，确保 train/eval 模式、no_grad、loss、metric 的边界清楚。
   - fit 负责学习率调度、早停、保存最优模型和记录 history。
   - evaluate 只做评估，不更新参数；plot 把 loss、metric、lr 放在一起看。

2. UniversalVisualizer 在管什么
   - model_summary 检查层名、类型、参数量，先确认模型结构没有搭错。
   - plot_parameter_distributions 检查权重是否异常偏移、过窄或过宽。
   - visualize_predictions 把预测样本、真实标签和错误案例摆出来，避免只看总准确率。

3. 默认值怎么落地
   - AdamW + lr=0.001 + weight_decay=1e-4 是小型分类项目常用起点。
   - grad_clip=1.0 是训练保险丝，能挡住偶发梯度尖峰，但不能替代学习率诊断。
   - early_stopping_patience=5 适合演示；真实项目通常根据总 epoch 和验证集噪声设为 5% 到 15%。

工程坑案例：
   我见过项目只保存 best_model.pth，却没有保存 config 和数据切分方式。三周后指标复现不了，没人知道当时用的是哪份数据。
   完整项目目录必须把 config、checkpoint、training_history.json、训练曲线和最终评估放在同一个实验目录中。

进阶思考：
   如果验证 loss 改善但业务指标变差，你应该先查 metric_fn、数据切分，还是模型结构？为什么 evaluate 不应该偷偷调用 optimizer.step()？
""".strip())


# ─────────────────────────────────────────────────────────
# 统一训练器：支持任意 PyTorch 模型
# ─────────────────────────────────────────────────────────

class UniversalTrainer:
    """
    万能训练器：一套代码训练任意 PyTorch 模型

    特性：
    - 自动设备管理（CPU/GPU）
    - 内置梯度裁剪
    - 学习率调度
    - 早停机制
    - 自动保存最优模型
    - 完整训练历史记录
    - 一键可视化

    使用方法：
        trainer = UniversalTrainer(
            model=MyModel(),
            optimizer=torch.optim.Adam(model.parameters()),
            loss_fn=nn.CrossEntropyLoss(),
        )
        trainer.fit(train_loader, val_loader, epochs=50)
        trainer.plot()
        trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        scheduler=None,
        device: str = None,
        grad_clip: float = 1.0,
        early_stopping_patience: int = None,
        save_best: bool = True,
        save_path: str = 'best_model.pth',
        metric_fn: Callable = None,
        metric_name: str = 'metric',
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_clip = grad_clip
        self.early_stopping_patience = early_stopping_patience
        self.save_best = save_best
        self.save_path = save_path
        self.metric_fn = metric_fn
        self.metric_name = metric_name

        self.model.to(self.device)
        self.history = defaultdict(list)
        self._best_val_loss = float('inf')
        self._patience_counter = 0
        self._epoch = 0

        n_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {n_params:,}")
        print(f"训练设备: {self.device}")

    def _run_epoch(self, loader, training: bool) -> dict:
        """运行一个 epoch（训练或验证）"""
        self.model.train() if training else self.model.eval()

        total_loss = 0.0
        total_metric = 0.0
        n_batches = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                # 支持 (x, y) 或 (x,) 格式
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.model(x)
                    loss = self.loss_fn(pred, y)
                else:
                    x = batch[0].to(self.device)
                    pred = self.model(x)
                    loss = self.loss_fn(pred, x)  # 自编码器等
                    y = x

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.grad_clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                total_loss += loss.item()
                if self.metric_fn is not None:
                    total_metric += self.metric_fn(pred, y)
                n_batches += 1

        result = {'loss': total_loss / max(n_batches, 1)}
        if self.metric_fn is not None:
            result[self.metric_name] = total_metric / max(n_batches, 1)
        return result

    def fit(self, train_loader, val_loader=None, epochs: int = 10, verbose: bool = True):
        """训练模型"""
        print(f"\n开始训练，共 {epochs} 个 epoch...")
        print("=" * 70)

        for epoch in range(1, epochs + 1):
            self._epoch = epoch
            t0 = time.time()

            # 训练
            train_metrics = self._run_epoch(train_loader, training=True)
            self.history['train_loss'].append(train_metrics['loss'])
            if self.metric_fn:
                self.history[f'train_{self.metric_name}'].append(
                    train_metrics.get(self.metric_name, 0)
                )

            # 验证
            val_str = ""
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, training=False)
                self.history['val_loss'].append(val_metrics['loss'])
                if self.metric_fn:
                    self.history[f'val_{self.metric_name}'].append(
                        val_metrics.get(self.metric_name, 0)
                    )
                val_str = f" | Val Loss={val_metrics['loss']:.4f}"
                if self.metric_fn:
                    val_str += f" {self.metric_name}={val_metrics.get(self.metric_name, 0):.4f}"

                # 早停检查
                if val_metrics['loss'] < self._best_val_loss:
                    self._best_val_loss = val_metrics['loss']
                    self._patience_counter = 0
                    if self.save_best:
                        torch.save(self.model.state_dict(), self.save_path)
                else:
                    self._patience_counter += 1

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loader:
                        self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(lr)

            elapsed = time.time() - t0
            if verbose:
                train_str = f"Train Loss={train_metrics['loss']:.4f}"
                if self.metric_fn:
                    train_str += f" {self.metric_name}={train_metrics.get(self.metric_name, 0):.4f}"
                print(f"Epoch {epoch:4d}/{epochs} | {train_str}{val_str} | "
                      f"LR={lr:.2e} | {elapsed:.1f}s")

            # 早停
            if (self.early_stopping_patience and
                    self._patience_counter >= self.early_stopping_patience):
                print(f"\n早停触发！验证损失 {self.early_stopping_patience} 个 epoch 未改善。")
                break

        print(f"\n训练完成！最优验证损失: {self._best_val_loss:.4f}")
        return self

    def evaluate(self, loader) -> dict:
        """在测试集上评估"""
        metrics = self._run_epoch(loader, training=False)
        print(f"\n测试集评估: {metrics}")
        return metrics

    def plot(self, figsize=(15, 4)):
        """绘制训练历史"""
        metrics_to_plot = ['loss']
        if self.metric_fn:
            metrics_to_plot.append(self.metric_name)
        metrics_to_plot.append('lr')

        n_plots = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics_to_plot):
            if metric == 'lr':
                if 'lr' in self.history:
                    ax.plot(self.history['lr'], 'g-o', markersize=3)
                    ax.set_title('学习率', fontsize=11)
                    ax.set_yscale('log')
            else:
                train_key = f'train_{metric}'
                val_key = f'val_{metric}'
                if train_key in self.history:
                    ax.plot(self.history[train_key], 'b-', label='训练', linewidth=1.5)
                if val_key in self.history:
                    ax.plot(self.history[val_key], 'r-', label='验证', linewidth=1.5)
                ax.set_title(metric, fontsize=11)
                ax.legend()
                if 'loss' in metric:
                    ax.set_yscale('log')
                elif 'acc' in metric:
                    ax.set_ylim(0, 1.05)

            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)

        plt.suptitle('训练历史', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('universal_training.png', dpi=150, bbox_inches='tight')
        plt.show()

    def save_history(self, path: str = 'training_history.json'):
        """保存训练历史到 JSON"""
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        print(f"训练历史已保存: {path}")

    def load_best(self):
        """加载最优模型权重"""
        if os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
            print(f"已加载最优模型: {self.save_path}")
        return self


# ─────────────────────────────────────────────────────────
# 万能可视化面板
# ─────────────────────────────────────────────────────────

class UniversalVisualizer:
    """
    万能可视化面板：一键生成完整的模型分析报告

    支持：
    - 模型结构摘要
    - 参数分布可视化
    - 特征图可视化（CNN）
    - 注意力权重可视化（Transformer）
    - 预测结果可视化
    - 错误案例分析
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

    def model_summary(self):
        """打印模型结构摘要"""
        print("=" * 70)
        print("模型结构摘要")
        print("=" * 70)
        print(f"{'层名':40s} {'类型':20s} {'参数量':12s}")
        print("-" * 70)

        total_params = 0
        trainable_params = 0

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                n_params = sum(p.numel() for p in module.parameters())
                n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total_params += n_params
                trainable_params += n_trainable
                if n_params > 0:
                    print(f"{name:40s} {type(module).__name__:20s} {n_params:12,d}")

        print("-" * 70)
        print(f"{'总参数量':40s} {'':20s} {total_params:12,d}")
        print(f"{'可训练参数':40s} {'':20s} {trainable_params:12,d}")
        print(f"{'冻结参数':40s} {'':20s} {total_params - trainable_params:12,d}")
        print(f"\n模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    def plot_parameter_distributions(self, figsize=(16, 8)):
        """可视化所有参数的分布"""
        params = [(name, p.detach().cpu().numpy().flatten())
                  for name, p in self.model.named_parameters()
                  if p.requires_grad and p.numel() > 1]

        if not params:
            print("没有可训练参数")
            return

        n_params = len(params)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 4, n_rows * 3))
        axes = np.array(axes).flatten()

        for ax, (name, values) in zip(axes, params):
            ax.hist(values, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
            ax.set_title(f'{name}\nmean={values.mean():.3f} std={values.std():.3f}',
                         fontsize=8)
            ax.set_xlabel('参数值')
            ax.grid(True, alpha=0.3)

        for ax in axes[n_params:]:
            ax.axis('off')

        plt.suptitle('模型参数分布', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('parameter_distributions.png', dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_predictions(self, loader, n_samples=20,
                               class_names=None, task='classification'):
        """可视化预测结果"""
        self.model.eval()
        all_inputs, all_preds, all_targets = [], [], []

        with torch.no_grad():
            for batch in loader:
                if len(all_inputs) >= n_samples:
                    break
                x, y = batch
                x = x.to(self.device)
                pred = self.model(x)
                all_inputs.extend(x.cpu())
                all_preds.extend(pred.cpu())
                all_targets.extend(y.cpu())

        all_inputs = all_inputs[:n_samples]
        all_preds = all_preds[:n_samples]
        all_targets = all_targets[:n_samples]

        if task == 'classification':
            self._plot_classification_results(
                all_inputs, all_preds, all_targets, class_names
            )
        elif task == 'regression':
            self._plot_regression_results(all_preds, all_targets)

    def _plot_classification_results(self, inputs, preds, targets, class_names):
        n = len(inputs)
        n_cols = min(10, n)
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 1.5, n_rows * 2))
        axes = np.array(axes).flatten()

        for i, (inp, pred, target) in enumerate(zip(inputs, preds, targets)):
            ax = axes[i]
            # 尝试显示图像
            if inp.dim() == 3:
                img = inp.permute(1, 2, 0).numpy()
                if img.shape[2] == 1:
                    img = img.squeeze(2)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            else:
                ax.bar(range(len(inp)), inp.numpy())

            pred_class = pred.argmax().item() if pred.dim() > 0 else round(pred.item())
            true_class = target.item() if target.dim() == 0 else target.argmax().item()

            pred_name = class_names[pred_class] if class_names else str(pred_class)
            true_name = class_names[true_class] if class_names else str(true_class)

            color = 'green' if pred_class == true_class else 'red'
            ax.set_title(f'P:{pred_name}\nT:{true_name}', fontsize=7, color=color)
            ax.axis('off')

        for ax in axes[n:]:
            ax.axis('off')

        plt.suptitle('预测结果（绿=正确，红=错误）', fontsize=12)
        plt.tight_layout()
        plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
        plt.show()

    def _plot_regression_results(self, preds, targets):
        preds_np = torch.stack(preds).numpy().flatten()
        targets_np = torch.stack(targets).numpy().flatten()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].scatter(targets_np, preds_np, alpha=0.5, s=20)
        lim = [min(targets_np.min(), preds_np.min()),
               max(targets_np.max(), preds_np.max())]
        axes[0].plot(lim, lim, 'r--', label='完美预测')
        axes[0].set_xlabel('真实值')
        axes[0].set_ylabel('预测值')
        axes[0].set_title('预测值 vs 真实值', fontsize=11)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        residuals = preds_np - targets_np
        axes[1].hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
        axes[1].axvline(x=0, color='red', linestyle='--')
        axes[1].set_xlabel('残差（预测 - 真实）')
        axes[1].set_ylabel('频次')
        axes[1].set_title(f'残差分布\nMAE={np.abs(residuals).mean():.4f}', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('regression_results.png', dpi=150, bbox_inches='tight')
        plt.show()

    def full_report(self, loader=None, task='classification', class_names=None):
        """生成完整的模型分析报告"""
        print("\n" + "=" * 70)
        print("完整模型分析报告")
        print("=" * 70)

        self.model_summary()
        self.plot_parameter_distributions()

        if loader is not None:
            self.visualize_predictions(loader, task=task, class_names=class_names)

        print("\n报告生成完毕！")


# ─────────────────────────────────────────────────────────
# 完整端到端示例：MNIST 分类
# ─────────────────────────────────────────────────────────

def full_pipeline_demo():
    """
    完整端到端示例：
    数据加载 → 模型定义 → 训练 → 评估 → 可视化
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, random_split

    print("完整深度学习流水线演示")
    print("=" * 70)

    # 1. 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    # 2. 模型（残差网络）
    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
            )
        def forward(self, x):
            return F.relu(self.net(x) + x)

    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        ResBlock(32), ResBlock(32),
        nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        ResBlock(64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )

    # 3. 准确率指标
    def accuracy(pred, target):
        return (pred.argmax(1) == target).float().mean().item()

    # 4. 训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    trainer = UniversalTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        scheduler=scheduler,
        grad_clip=1.0,
        early_stopping_patience=5,
        save_best=True,
        save_path='best_mnist.pth',
        metric_fn=accuracy,
        metric_name='acc',
    )

    trainer.fit(train_loader, val_loader, epochs=10)
    trainer.plot()

    # 5. 测试
    trainer.load_best()
    test_metrics = trainer.evaluate(test_loader)

    # 6. 可视化报告
    visualizer = UniversalVisualizer(model)
    visualizer.full_report(test_loader, task='classification',
                            class_names=[str(i) for i in range(10)])

    return trainer, visualizer

if __name__ == "__main__":
    print_learning_guide()
    print("万能框架已准备好！取消注释下面一行来运行完整流水线。")
    print("\n使用方法：")
    print("  trainer = UniversalTrainer(model, optimizer, loss_fn, ...)")
    print("  trainer.fit(train_loader, val_loader, epochs=50)")
    print("  trainer.plot()")
    print("  visualizer = UniversalVisualizer(model)")
    print("  visualizer.full_report(test_loader)")
    # trainer, visualizer = full_pipeline_demo()

# ============================================================
# 代码段 2
# ============================================================

# ─────────────────────────────────────────────────────────
# 复制这个模板，5分钟开始你的实验
# ─────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ① 定义你的模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 在这里定义你的层
        self.net = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

# ② 准备数据
X_train = torch.randn(1000, 10)
y_train = (X_train.sum(1) > 0).float().unsqueeze(1)
X_val = torch.randn(200, 10)
y_val = (X_val.sum(1) > 0).float().unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

# ③ 初始化训练器
# 取消注释下面几行即可运行这个快速模板。
# model = MyModel()
# trainer = UniversalTrainer(
#     model=model,
#     optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
#     loss_fn=nn.BCEWithLogitsLoss(),
#     early_stopping_patience=10,
#     metric_fn=lambda p, y: ((p > 0) == y).float().mean().item(),
#     metric_name='acc',
# )

# ④ 训练
# trainer.fit(train_loader, val_loader, epochs=50)

# ⑤ 可视化
# trainer.plot()
# visualizer = UniversalVisualizer(model)
# visualizer.model_summary()
# visualizer.plot_parameter_distributions()
