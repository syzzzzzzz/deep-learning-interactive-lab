# 一键训练与评估：自动保存、日志、曲线绘制

## 1. 设计目标

```
目标：一行代码启动完整训练流程

trainer = ExperimentRunner(config)
trainer.run()

自动完成：
✓ 设备检测与模型迁移
✓ 训练循环 + 验证循环
✓ 学习率调度
✓ 梯度裁剪
✓ 早停机制
✓ 最优模型保存
✓ CSV/JSON 日志
✓ 训练曲线自动绘制
✓ 测试集最终评估
```

---

## 2. ExperimentRunner 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import os
import time
from collections import defaultdict
from typing import Dict, Any, Optional, Callable


class ExperimentRunner:
    """
    一键训练与评估

    使用方法：
        config = {
            'model': 'cnn',
            'model_params': {'in_channels': 1, 'num_classes': 10},
            'dataset': 'mnist',
            'task': 'classification',
            'epochs': 20,
            'lr': 0.001,
            'batch_size': 64,
            'save_dir': './experiments/mnist_cnn',
        }
        runner = ExperimentRunner(config)
        runner.run()
    """

    def __init__(self, config: dict,
                 model: nn.Module = None,
                 train_loader: DataLoader = None,
                 val_loader: DataLoader = None,
                 test_loader: DataLoader = None,
                 loss_fn=None,
                 metrics: dict = None):
        self.config = config
        self.device = self._resolve_device()

        # 模型（支持传入或从注册表构建）
        if model is not None:
            self.model = model.to(self.device)
        else:
            from part6_universal_framework.04_plugin_system import build_model
            self.model = build_model(
                config['model'],
                **config.get('model_params', {})
            ).to(self.device)

        # 数据（支持传入或从注册表构建）
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 任务
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.metrics = metrics or {'accuracy': lambda p, t: (p.argmax(1) == t).float().mean().item()}

        # 训练组件
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # 记录
        self.history = defaultdict(list)
        self.save_dir = config.get('save_dir', './experiments/default')
        os.makedirs(self.save_dir, exist_ok=True)

        # 状态
        self.best_val_metric = -float('inf')
        self.epochs_no_improve = 0
        self.global_step = 0

    def _resolve_device(self):
        device = self.config.get('device', 'auto')
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _build_optimizer(self):
        name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('lr', 0.001)
        wd = self.config.get('weight_decay', 1e-4)

        if name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=wd)
        elif name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"未知优化器: {name}")

    def _build_scheduler(self):
        name = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', 20)

        if name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif name == 'step':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif name == 'none':
            return None
        return None

    # ─────────────────────────────────────────────────────────
    # 训练循环
    # ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0
        all_metrics = defaultdict(float)
        n_batches = 0
        start = time.time()

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            # 前向
            logits = self.model(x)
            loss = self.loss_fn(logits, y)

            # 反向
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            grad_clip = self.config.get('grad_clip', 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            for name, fn in self.metrics.items():
                all_metrics[name] += fn(logits.detach(), y).item()
            n_batches += 1

            # 进度条
            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                pct = (batch_idx + 1) / len(self.train_loader) * 100
                print(f"\r  Epoch {epoch} [{pct:.0f}%] loss={loss.item():.4f}", end='')

        elapsed = time.time() - start

        result = {'loss': total_loss / n_batches, 'time': elapsed}
        for name in self.metrics:
            result[name] = all_metrics[name] / n_batches

        print(f"\r  Epoch {epoch} train: loss={result['loss']:.4f}", end='')
        return result

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0
        all_metrics = defaultdict(float)
        n_batches = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)

            total_loss += loss.item()
            for name, fn in self.metrics.items():
                all_metrics[name] += fn(logits, y).item()
            n_batches += 1

        result = {'loss': total_loss / n_batches}
        for name in self.metrics:
            result[name] = all_metrics[name] / n_batches
        return result

    # ─────────────────────────────────────────────────────────
    # 完整训练
    # ─────────────────────────────────────────────────────────

    def run(self):
        """一键运行完整训练流程"""
        print("=" * 60)
        print(f"实验: {self.config.get('name', 'unnamed')}")
        print(f"设备: {self.device}")
        print(f"模型: {sum(p.numel() for p in self.model.parameters()):,} 参数")
        print(f"训练集: {len(self.train_loader.dataset)} 样本")
        if self.val_loader:
            print(f"验证集: {len(self.val_loader.dataset)} 样本")
        print("=" * 60)

        epochs = self.config.get('epochs', 20)
        patience = self.config.get('patience', 10)
        monitor = self.config.get('monitor', 'accuracy')

        for epoch in range(1, epochs + 1):
            # 训练
            train_result = self._train_epoch(epoch)

            # 验证
            val_result = {}
            if self.val_loader:
                val_result = self._validate(self.val_loader)

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()

            # 记录历史
            lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_result['loss'])
            self.history['lr'].append(lr)
            for k, v in train_result.items():
                if k not in ('loss', 'time'):
                    self.history[f'train_{k}'].append(v)
            for k, v in val_result.items():
                self.history[f'val_{k}'].append(v)

            # 打印
            val_str = ' | '.join(f'val_{k}={v:.4f}' for k, v in val_result.items())
            print(f" | lr={lr:.6f} | {val_str}")

            # 保存最优模型
            val_metric = val_result.get(monitor, -float('inf'))
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.epochs_no_improve = 0
                self._save_checkpoint('best.pt', epoch, val_metric)
                print(f"  ✓ 最优模型已保存 ({monitor}={val_metric:.4f})")
            else:
                self.epochs_no_improve += 1

            # 每N个epoch保存检查点
            if epoch % self.config.get('save_every', 10) == 0:
                self._save_checkpoint(f'epoch_{epoch}.pt', epoch, val_metric)

            # 早停
            if self.epochs_no_improve >= patience:
                print(f"\n早停！{patience} 个 epoch 无改善")
                break

            # 写日志
            self._log_epoch(epoch, train_result, val_result, lr)

        # 训练结束
        print("\n" + "=" * 60)
        print(f"训练完成！最优 {monitor}: {self.best_val_metric:.4f}")
        print("=" * 60)

        # 加载最优模型评估
        self._load_checkpoint('best.pt')

        # 测试集评估
        if self.test_loader:
            test_result = self._validate(self.test_loader)
            print("测试集结果:", test_result)
            self._log_final(test_result)

        # 绘制曲线
        if self.config.get('plot_curves', True):
            self.plot()

        # 保存配置
        self._save_config()

        return self.history

    # ─────────────────────────────────────────────────────────
    # 保存与加载
    # ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename, epoch, metric):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric': metric,
            'config': self.config,
        }, path)

    def _load_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"已加载模型: {filename} (epoch={ckpt['epoch']}, metric={ckpt['metric']:.4f})")

    def _log_epoch(self, epoch, train_result, val_result, lr):
        """追加写入 CSV 日志"""
        log_path = os.path.join(self.save_dir, 'training_log.csv')
        row = {'epoch': epoch, 'lr': lr}
        row.update({f'train_{k}': v for k, v in train_result.items()})
        row.update({f'val_{k}': v for k, v in val_result.items()})

        file_exists = os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _log_final(self, test_result):
        """保存最终结果"""
        result_path = os.path.join(self.save_dir, 'final_result.json')
        result = {
            'best_val_metric': self.best_val_metric,
            'test_metrics': test_result,
            'total_epochs': len(self.history['train_loss']),
        }
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    def _save_config(self):
        """保存实验配置"""
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

    # ─────────────────────────────────────────────────────────
    # 可视化
    # ─────────────────────────────────────────────────────────

    def plot(self, figsize=(14, 10)):
        """自动绘制训练曲线"""
        n_plots = 1 + len([k for k in self.history if k.startswith('val_')])
        n_plots = min(n_plots, 4)  # 最多4个子图

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        epochs = range(1, len(self.history['train_loss']) + 1)

        # 损失曲线
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='训练')
        if 'val_loss' in self.history:
            axes[0].plot(epochs, self.history['val_loss'], 'r-', label='验证')
        axes[0].set_title('损失曲线', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 指标曲线
        plot_idx = 1
        for key in list(self.history.keys()):
            if key.startswith('val_') and key != 'val_loss':
                metric_name = key[4:]
                train_key = f'train_{metric_name}'
                if train_key in self.history:
                    axes[plot_idx].plot(epochs, self.history[train_key], 'b-', label=f'训练 {metric_name}')
                axes[plot_idx].plot(epochs, self.history[key], 'r-', label=f'验证 {metric_name}')
                axes[plot_idx].set_title(f'{metric_name} 曲线', fontsize=12, fontweight='bold')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
                if plot_idx >= 3:
                    break

        # 学习率曲线
        axes[3].plot(epochs, self.history['lr'], 'g-')
        axes[3].set_title('学习率变化', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Epoch')
        axes[3].grid(True, alpha=0.3)

        for i in range(plot_idx, 3):
            axes[i].axis('off')

        plt.suptitle(f'训练曲线 — {self.config.get("name", "experiment")}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"曲线已保存: {save_path}")
```

---

## 3. 使用示例

```python
def demo_one_click_training():
    """一键训练演示"""
    import torchvision
    import torchvision.transforms as transforms

    # 准备数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

    # 分割验证集
    val_size = 5000
    train_size = len(train_set) - val_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    # 配置（只改这里就能切换实验）
    config = {
        'name': 'mnist_cnn_v1',
        'model': 'cnn',
        'model_params': {'in_channels': 1, 'num_classes': 10},
        'epochs': 10,
        'lr': 0.001,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'grad_clip': 1.0,
        'patience': 5,
        'monitor': 'accuracy',
        'save_dir': './experiments/mnist_cnn',
        'plot_curves': True,
        'device': 'auto',
    }

    # 一键训练！
    runner = ExperimentRunner(config, train_loader=train_loader,
                               val_loader=val_loader, test_loader=test_loader)
    history = runner.run()


# demo_one_click_training()  # 取消注释运行
```

---

## 小结

| 功能 | 自动化程度 | 输出 |
|------|-----------|------|
| 训练循环 | 全自动 | 每 epoch 打印 |
| 验证评估 | 全自动 | val_loss + val_metrics |
| 模型保存 | 全自动 | best.pt + epoch_N.pt |
| 日志记录 | 全自动 | training_log.csv |
| 最终评估 | 全自动 | final_result.json |
| 曲线绘制 | 全自动 | training_curves.png |
| 早停 | 全自动 | patience 触发 |
| 学习率调度 | 全自动 | 支持 cosine/step |
