"""
自动生成自: part6_universal_framework\05_one_click_training.md
可独立运行的 Python 源码
"""

def print_learning_guide():
    print("""
学习导读：一键训练不是把训练藏进一个按钮，而是把一次实验从配置到产物完整串起来。

1. 训练流程怎么看
   - config 决定实验名称、模型、学习率、优化器、调度器、梯度裁剪、早停和保存目录。
   - _train_epoch 只做训练，_validate 只做验证，run 负责把训练、验证、调度、保存、日志串成闭环。
   - monitor 指标决定 best.pt 保存哪一轮；分类常用 accuracy 或 F1，回归常用 val_loss、MAE 或 RMSE。

2. 产物怎么看
   - best.pt：验证集指标最好的模型，通常比最后一轮更可靠。
   - training_log.csv：每个 epoch 的 loss、metric、lr 和耗时，是排查曲线异常的第一证据。
   - config.json：复现实验的说明书，没有它就很难知道当时到底用了什么参数。
   - training_curves.png：把训练/验证曲线和学习率放在一起看，判断过拟合、欠拟合和学习率是否合适。

3. 默认值怎么落地
   - lr=0.001、optimizer=adam、scheduler=cosine、grad_clip=1.0、patience=5 是演示级默认值。
   - 真实项目要根据验证集噪声扩大 patience，并把 save_dir 命名成包含任务、模型、日期和关键参数的实验目录。

工程坑案例：
   我见过只保存 last.pt 的项目，训练后期已经过拟合，但上线用的是最后一轮，指标比 best epoch 低很多。
   一键训练必须默认保存 best checkpoint，并把 monitor 指标、epoch 和 config 一起写进 checkpoint。

进阶思考：
   如果训练中断，要恢复实验至少需要哪些文件？为什么日志、配置和 checkpoint 必须放在同一个 save_dir 下？
""".strip())

# SYNTAX_SKIP: import torch
# SYNTAX_SKIP: import torch.nn as nn
# SYNTAX_SKIP: import torch.nn.functional as F
# SYNTAX_SKIP: from torch.utils.data import DataLoader, random_split
# SYNTAX_SKIP: import numpy as np
# SYNTAX_SKIP: import matplotlib.pyplot as plt
# SYNTAX_SKIP: import json
# SYNTAX_SKIP: import csv
# SYNTAX_SKIP: import os
# SYNTAX_SKIP: import time
# SYNTAX_SKIP: from collections import defaultdict
# SYNTAX_SKIP: from typing import Dict, Any, Optional, Callable
# SYNTAX_SKIP: 
# SYNTAX_SKIP: 
# SYNTAX_SKIP: class ExperimentRunner:
# SYNTAX_SKIP:     """
# SYNTAX_SKIP:     一键训练与评估
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     使用方法：
# SYNTAX_SKIP:         config = {
# SYNTAX_SKIP:             'model': 'cnn',
# SYNTAX_SKIP:             'model_params': {'in_channels': 1, 'num_classes': 10},
# SYNTAX_SKIP:             'dataset': 'mnist',
# SYNTAX_SKIP:             'task': 'classification',
# SYNTAX_SKIP:             'epochs': 20,
# SYNTAX_SKIP:             'lr': 0.001,
# SYNTAX_SKIP:             'batch_size': 64,
# SYNTAX_SKIP:             'save_dir': './experiments/mnist_cnn',
# SYNTAX_SKIP:         }
# SYNTAX_SKIP:         runner = ExperimentRunner(config)
# SYNTAX_SKIP:         runner.run()
# SYNTAX_SKIP:     """
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def __init__(self, config: dict,
# SYNTAX_SKIP:                  model: nn.Module = None,
# SYNTAX_SKIP:                  train_loader: DataLoader = None,
# SYNTAX_SKIP:                  val_loader: DataLoader = None,
# SYNTAX_SKIP:                  test_loader: DataLoader = None,
# SYNTAX_SKIP:                  loss_fn=None,
# SYNTAX_SKIP:                  metrics: dict = None):
# SYNTAX_SKIP:         self.config = config
# SYNTAX_SKIP:         self.device = self._resolve_device()
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 模型（支持传入或从注册表构建）
# SYNTAX_SKIP:         if model is not None:
# SYNTAX_SKIP:             self.model = model.to(self.device)
# SYNTAX_SKIP:         else:
# SYNTAX_SKIP:             from part6_universal_framework.04_plugin_system import build_model
# SYNTAX_SKIP:             self.model = build_model(
# SYNTAX_SKIP:                 config['model'],
# SYNTAX_SKIP: # **config.get('model_params', {})
# SYNTAX_SKIP:             ).to(self.device)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 数据（支持传入或从注册表构建）
# SYNTAX_SKIP:         self.train_loader = train_loader
# SYNTAX_SKIP:         self.val_loader = val_loader
# SYNTAX_SKIP:         self.test_loader = test_loader
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 任务
# SYNTAX_SKIP:         self.loss_fn = loss_fn or nn.CrossEntropyLoss()
# SYNTAX_SKIP:         self.metrics = metrics or {'accuracy': lambda p, t: (p.argmax(1) == t).float().mean().item()}
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 训练组件
# SYNTAX_SKIP:         self.optimizer = self._build_optimizer()
# SYNTAX_SKIP:         self.scheduler = self._build_scheduler()
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 记录
# SYNTAX_SKIP:         self.history = defaultdict(list)
# SYNTAX_SKIP:         self.save_dir = config.get('save_dir', './experiments/default')
# SYNTAX_SKIP:         os.makedirs(self.save_dir, exist_ok=True)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 状态
# SYNTAX_SKIP:         self.best_val_metric = -float('inf')
# SYNTAX_SKIP:         self.epochs_no_improve = 0
# SYNTAX_SKIP:         self.global_step = 0
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def _resolve_device(self):
# SYNTAX_SKIP:         device = self.config.get('device', 'auto')
# SYNTAX_SKIP:         if device == 'auto':
# SYNTAX_SKIP:             return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# SYNTAX_SKIP:         return torch.device(device)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def _build_optimizer(self):
# SYNTAX_SKIP:         name = self.config.get('optimizer', 'adam').lower()
# SYNTAX_SKIP:         lr = self.config.get('lr', 0.001)
# SYNTAX_SKIP:         wd = self.config.get('weight_decay', 1e-4)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         if name == 'adam':
# SYNTAX_SKIP:             return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
# SYNTAX_SKIP:         elif name == 'sgd':
# SYNTAX_SKIP:             return torch.optim.SGD(self.model.parameters(), lr=lr,
# SYNTAX_SKIP:                                     momentum=0.9, weight_decay=wd)
# SYNTAX_SKIP:         elif name == 'adamw':
# SYNTAX_SKIP:             return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
# SYNTAX_SKIP:         else:
# SYNTAX_SKIP:             raise ValueError(f"未知优化器: {name}")
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def _build_scheduler(self):
# SYNTAX_SKIP:         name = self.config.get('scheduler', 'cosine')
# SYNTAX_SKIP:         epochs = self.config.get('epochs', 20)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         if name == 'cosine':
# SYNTAX_SKIP:             return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
# SYNTAX_SKIP:         elif name == 'step':
# SYNTAX_SKIP:             return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
# SYNTAX_SKIP:         elif name == 'none':
# SYNTAX_SKIP:             return None
# SYNTAX_SKIP:         return None
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     # ─────────────────────────────────────────────────────────
# SYNTAX_SKIP:     # 训练循环
# SYNTAX_SKIP:     # ─────────────────────────────────────────────────────────
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def _train_epoch(self, epoch: int) -> dict:
# SYNTAX_SKIP:         self.model.train()
# SYNTAX_SKIP:         total_loss = 0
# SYNTAX_SKIP:         all_metrics = defaultdict(float)
# SYNTAX_SKIP:         n_batches = 0
# SYNTAX_SKIP:         start = time.time()
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         for batch_idx, (x, y) in enumerate(self.train_loader):
# SYNTAX_SKIP:             x, y = x.to(self.device), y.to(self.device)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 前向
# SYNTAX_SKIP:             logits = self.model(x)
# SYNTAX_SKIP:             loss = self.loss_fn(logits, y)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 反向
# SYNTAX_SKIP:             self.optimizer.zero_grad()
# SYNTAX_SKIP:             loss.backward()
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 梯度裁剪
# SYNTAX_SKIP:             grad_clip = self.config.get('grad_clip', 1.0)
# SYNTAX_SKIP:             if grad_clip > 0:
# SYNTAX_SKIP:                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             self.optimizer.step()
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 记录
# SYNTAX_SKIP:             total_loss += loss.item()
# SYNTAX_SKIP:             for name, fn in self.metrics.items():
# SYNTAX_SKIP:                 all_metrics[name] += fn(logits.detach(), y).item()
# SYNTAX_SKIP:             n_batches += 1
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 进度条
# SYNTAX_SKIP:             if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
# SYNTAX_SKIP:                 pct = (batch_idx + 1) / len(self.train_loader) * 100
# SYNTAX_SKIP:                 print(f"\r  Epoch {epoch} [{pct:.0f}%] loss={loss.item():.4f}", end='')
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         elapsed = time.time() - start
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         result = {'loss': total_loss / n_batches, 'time': elapsed}
# SYNTAX_SKIP:         for name in self.metrics:
# SYNTAX_SKIP:             result[name] = all_metrics[name] / n_batches
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         print(f"\r  Epoch {epoch} train: loss={result['loss']:.4f}", end='')
# SYNTAX_SKIP:         return result
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     @torch.no_grad()
# SYNTAX_SKIP:     def _validate(self, loader: DataLoader) -> dict:
# SYNTAX_SKIP:         self.model.eval()
# SYNTAX_SKIP:         total_loss = 0
# SYNTAX_SKIP:         all_metrics = defaultdict(float)
# SYNTAX_SKIP:         n_batches = 0
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         for x, y in loader:
# SYNTAX_SKIP:             x, y = x.to(self.device), y.to(self.device)
# SYNTAX_SKIP:             logits = self.model(x)
# SYNTAX_SKIP:             loss = self.loss_fn(logits, y)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             total_loss += loss.item()
# SYNTAX_SKIP:             for name, fn in self.metrics.items():
# SYNTAX_SKIP:                 all_metrics[name] += fn(logits, y).item()
# SYNTAX_SKIP:             n_batches += 1
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         result = {'loss': total_loss / n_batches}
# SYNTAX_SKIP:         for name in self.metrics:
# SYNTAX_SKIP:             result[name] = all_metrics[name] / n_batches
# SYNTAX_SKIP:         return result
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     # ─────────────────────────────────────────────────────────
# SYNTAX_SKIP:     # 完整训练
# SYNTAX_SKIP:     # ─────────────────────────────────────────────────────────
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def run(self):
# SYNTAX_SKIP:         """一键运行完整训练流程"""
# SYNTAX_SKIP:         print("=" * 60)
# SYNTAX_SKIP:         print(f"实验: {self.config.get('name', 'unnamed')}")
# SYNTAX_SKIP:         print(f"设备: {self.device}")
# SYNTAX_SKIP:         print(f"模型: {sum(p.numel() for p in self.model.parameters()):,} 参数")
# SYNTAX_SKIP:         print(f"训练集: {len(self.train_loader.dataset)} 样本")
# SYNTAX_SKIP:         if self.val_loader:
# SYNTAX_SKIP:             print(f"验证集: {len(self.val_loader.dataset)} 样本")
# SYNTAX_SKIP:         print("=" * 60)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         epochs = self.config.get('epochs', 20)
# SYNTAX_SKIP:         patience = self.config.get('patience', 10)
# SYNTAX_SKIP:         monitor = self.config.get('monitor', 'accuracy')
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         for epoch in range(1, epochs + 1):
# SYNTAX_SKIP:             # 训练
# SYNTAX_SKIP:             train_result = self._train_epoch(epoch)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 验证
# SYNTAX_SKIP:             val_result = {}
# SYNTAX_SKIP:             if self.val_loader:
# SYNTAX_SKIP:                 val_result = self._validate(self.val_loader)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 学习率调度
# SYNTAX_SKIP:             if self.scheduler:
# SYNTAX_SKIP:                 self.scheduler.step()
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 记录历史
# SYNTAX_SKIP:             lr = self.optimizer.param_groups[0]['lr']
# SYNTAX_SKIP:             self.history['train_loss'].append(train_result['loss'])
# SYNTAX_SKIP:             self.history['lr'].append(lr)
# SYNTAX_SKIP:             for k, v in train_result.items():
# SYNTAX_SKIP:                 if k not in ('loss', 'time'):
# SYNTAX_SKIP:                     self.history[f'train_{k}'].append(v)
# SYNTAX_SKIP:             for k, v in val_result.items():
# SYNTAX_SKIP:                 self.history[f'val_{k}'].append(v)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 打印
# SYNTAX_SKIP:             val_str = ' | '.join(f'val_{k}={v:.4f}' for k, v in val_result.items())
# SYNTAX_SKIP:             print(f" | lr={lr:.6f} | {val_str}")
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 保存最优模型
# SYNTAX_SKIP:             val_metric = val_result.get(monitor, -float('inf'))
# SYNTAX_SKIP:             if val_metric > self.best_val_metric:
# SYNTAX_SKIP:                 self.best_val_metric = val_metric
# SYNTAX_SKIP:                 self.epochs_no_improve = 0
# SYNTAX_SKIP:                 self._save_checkpoint('best.pt', epoch, val_metric)
# SYNTAX_SKIP:                 print(f"  ✓ 最优模型已保存 ({monitor}={val_metric:.4f})")
# SYNTAX_SKIP:             else:
# SYNTAX_SKIP:                 self.epochs_no_improve += 1
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 每N个epoch保存检查点
# SYNTAX_SKIP:             if epoch % self.config.get('save_every', 10) == 0:
# SYNTAX_SKIP:                 self._save_checkpoint(f'epoch_{epoch}.pt', epoch, val_metric)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 早停
# SYNTAX_SKIP:             if self.epochs_no_improve >= patience:
# SYNTAX_SKIP:                 print(f"\n早停！{patience} 个 epoch 无改善")
# SYNTAX_SKIP:                 break
# SYNTAX_SKIP: 
# SYNTAX_SKIP:             # 写日志
# SYNTAX_SKIP:             self._log_epoch(epoch, train_result, val_result, lr)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 训练结束
# SYNTAX_SKIP:         print("\n" + "=" * 60)
# SYNTAX_SKIP:         print(f"训练完成！最优 {monitor}: {self.best_val_metric:.4f}")
# SYNTAX_SKIP:         print("=" * 60)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 加载最优模型评估
# SYNTAX_SKIP:         self._load_checkpoint('best.pt')
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 测试集评估
# SYNTAX_SKIP:         if self.test_loader:
# SYNTAX_SKIP:             test_result = self._validate(self.test_loader)
# SYNTAX_SKIP:             print("测试集结果:", test_result)
# SYNTAX_SKIP:             self._log_final(test_result)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 绘制曲线
# SYNTAX_SKIP:         if self.config.get('plot_curves', True):
# SYNTAX_SKIP:             self.plot()
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 保存配置
# SYNTAX_SKIP:         self._save_config()
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         return self.history
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     # ─────────────────────────────────────────────────────────
# SYNTAX_SKIP:     # 保存与加载
# SYNTAX_SKIP:     # ─────────────────────────────────────────────────────────
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def _save_checkpoint(self, filename, epoch, metric):
# SYNTAX_SKIP:         path = os.path.join(self.save_dir, filename)
# SYNTAX_SKIP:         torch.save({
# SYNTAX_SKIP:             'epoch': epoch,
# SYNTAX_SKIP:             'model_state_dict': self.model.state_dict(),
# SYNTAX_SKIP:             'optimizer_state_dict': self.optimizer.state_dict(),
# SYNTAX_SKIP:             'metric': metric,
# SYNTAX_SKIP:             'config': self.config,
# SYNTAX_SKIP:         }, path)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def _load_checkpoint(self, filename):
# SYNTAX_SKIP:         path = os.path.join(self.save_dir, filename)
# SYNTAX_SKIP:         if os.path.exists(path):
# SYNTAX_SKIP:             ckpt = torch.load(path, map_location=self.device)
# SYNTAX_SKIP:             self.model.load_state_dict(ckpt['model_state_dict'])
# SYNTAX_SKIP:             print(f"已加载模型: {filename} (epoch={ckpt['epoch']}, metric={ckpt['metric']:.4f})")
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def _log_epoch(self, epoch, train_result, val_result, lr):
# SYNTAX_SKIP:         """追加写入 CSV 日志"""
# SYNTAX_SKIP:         log_path = os.path.join(self.save_dir, 'training_log.csv')
# SYNTAX_SKIP:         row = {'epoch': epoch, 'lr': lr}
# SYNTAX_SKIP:         row.update({f'train_{k}': v for k, v in train_result.items()})
# SYNTAX_SKIP:         row.update({f'val_{k}': v for k, v in val_result.items()})
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         file_exists = os.path.exists(log_path)
# SYNTAX_SKIP:         with open(log_path, 'a', newline='') as f:
# SYNTAX_SKIP:             writer = csv.DictWriter(f, fieldnames=row.keys())
# SYNTAX_SKIP:             if not file_exists:
# SYNTAX_SKIP:                 writer.writeheader()
# SYNTAX_SKIP:             writer.writerow(row)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def _log_final(self, test_result):
# SYNTAX_SKIP:         """保存最终结果"""
# SYNTAX_SKIP:         result_path = os.path.join(self.save_dir, 'final_result.json')
# SYNTAX_SKIP:         result = {
# SYNTAX_SKIP:             'best_val_metric': self.best_val_metric,
# SYNTAX_SKIP:             'test_metrics': test_result,
# SYNTAX_SKIP:             'total_epochs': len(self.history['train_loss']),
# SYNTAX_SKIP:         }
# SYNTAX_SKIP:         with open(result_path, 'w') as f:
# SYNTAX_SKIP:             json.dump(result, f, indent=2, default=str)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def _save_config(self):
# SYNTAX_SKIP:         """保存实验配置"""
# SYNTAX_SKIP:         config_path = os.path.join(self.save_dir, 'config.json')
# SYNTAX_SKIP:         with open(config_path, 'w') as f:
# SYNTAX_SKIP:             json.dump(self.config, f, indent=2, default=str)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     # ─────────────────────────────────────────────────────────
# SYNTAX_SKIP:     # 可视化
# SYNTAX_SKIP:     # ─────────────────────────────────────────────────────────
# SYNTAX_SKIP: 
# SYNTAX_SKIP:     def plot(self, figsize=(14, 10)):
# SYNTAX_SKIP:         """自动绘制训练曲线"""
# SYNTAX_SKIP:         n_plots = 1 + len([k for k in self.history if k.startswith('val_')])
# SYNTAX_SKIP:         n_plots = min(n_plots, 4)  # 最多4个子图
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         fig, axes = plt.subplots(2, 2, figsize=figsize)
# SYNTAX_SKIP:         axes = axes.flatten()
# SYNTAX_SKIP:         epochs = range(1, len(self.history['train_loss']) + 1)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 损失曲线
# SYNTAX_SKIP:         axes[0].plot(epochs, self.history['train_loss'], 'b-', label='训练')
# SYNTAX_SKIP:         if 'val_loss' in self.history:
# SYNTAX_SKIP:             axes[0].plot(epochs, self.history['val_loss'], 'r-', label='验证')
# SYNTAX_SKIP:         axes[0].set_title('损失曲线', fontsize=12, fontweight='bold')
# SYNTAX_SKIP:         axes[0].set_xlabel('Epoch')
# SYNTAX_SKIP:         axes[0].legend()
# SYNTAX_SKIP:         axes[0].grid(True, alpha=0.3)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 指标曲线
# SYNTAX_SKIP:         plot_idx = 1
# SYNTAX_SKIP:         for key in list(self.history.keys()):
# SYNTAX_SKIP:             if key.startswith('val_') and key != 'val_loss':
# SYNTAX_SKIP:                 metric_name = key[4:]
# SYNTAX_SKIP:                 train_key = f'train_{metric_name}'
# SYNTAX_SKIP:                 if train_key in self.history:
# SYNTAX_SKIP:                     axes[plot_idx].plot(epochs, self.history[train_key], 'b-', label=f'训练 {metric_name}')
# SYNTAX_SKIP:                 axes[plot_idx].plot(epochs, self.history[key], 'r-', label=f'验证 {metric_name}')
# SYNTAX_SKIP:                 axes[plot_idx].set_title(f'{metric_name} 曲线', fontsize=12, fontweight='bold')
# SYNTAX_SKIP:                 axes[plot_idx].set_xlabel('Epoch')
# SYNTAX_SKIP:                 axes[plot_idx].legend()
# SYNTAX_SKIP:                 axes[plot_idx].grid(True, alpha=0.3)
# SYNTAX_SKIP:                 plot_idx += 1
# SYNTAX_SKIP:                 if plot_idx >= 3:
# SYNTAX_SKIP:                     break
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         # 学习率曲线
# SYNTAX_SKIP:         axes[3].plot(epochs, self.history['lr'], 'g-')
# SYNTAX_SKIP:         axes[3].set_title('学习率变化', fontsize=12, fontweight='bold')
# SYNTAX_SKIP:         axes[3].set_xlabel('Epoch')
# SYNTAX_SKIP:         axes[3].grid(True, alpha=0.3)
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         for i in range(plot_idx, 3):
# SYNTAX_SKIP:             axes[i].axis('off')
# SYNTAX_SKIP: 
# SYNTAX_SKIP:         plt.suptitle(f'训练曲线 — {self.config.get("name", "experiment")}',
# SYNTAX_SKIP:                      fontsize=14, fontweight='bold')
# SYNTAX_SKIP:         plt.tight_layout()
# SYNTAX_SKIP:         save_path = os.path.join(self.save_dir, 'training_curves.png')
# SYNTAX_SKIP:         plt.savefig(save_path, dpi=150)
# SYNTAX_SKIP:         plt.show()
# SYNTAX_SKIP:         print(f"曲线已保存: {save_path}")

# ============================================================
# 代码段 2
# ============================================================

def demo_one_click_training():
    """一键训练演示"""
    print_learning_guide()
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

if __name__ == "__main__":
    print_learning_guide()
