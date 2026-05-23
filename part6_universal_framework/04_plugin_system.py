"""
自动生成自: part6_universal_framework\04_plugin_system.md
可独立运行的 Python 源码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Type, Any, Callable, Optional
from dataclasses import dataclass


_GUIDE_PRINTED = False


def print_learning_guide():
    global _GUIDE_PRINTED
    if _GUIDE_PRINTED:
        return
    _GUIDE_PRINTED = True
    print("""
学习导读：插件系统的核心不是炫技，而是让模型、数据集和任务可以被稳定替换。

1. 注册表怎么看
   - MODEL_REGISTRY 把模型名称映射到模型类和默认参数。
   - DATASET_REGISTRY 把数据集名称映射到 Dataset 类和默认参数。
   - TASK_REGISTRY 把任务名称映射到 loss、metrics 和任务对象。
   - build_model/build_dataset/build_task 是统一入口，配置文件只需要写 name 和 params。

2. 钩子系统怎么看
   - on_epoch_end 这类钩子点是训练流程的插槽，适合记录学习率、监控梯度、早停和保存额外日志。
   - 钩子不能悄悄改变核心训练语义，否则排查会变困难；它应该只做可观察、可关闭、可记录的事情。

3. 配置模板怎么看
   - model/dataset/task/training/hooks/output 分别对应模型、数据、任务、训练参数、扩展逻辑和产物位置。
   - 推荐默认值：lr=0.001、weight_decay=1e-4、grad_clip=1.0、patience=10，先跑通再根据验证曲线微调。

工程坑案例：
   插件系统最常见的问题是名称冲突和默认参数覆盖不清。真实项目里要在启动时打印最终合并后的 config，
   并检查重复注册；插件加载失败必须报出文件名和异常，不能静默跳过。

进阶思考：
   哪些变化频繁的组件值得插件化？如果一个插件需要修改训练循环内部很多行，说明抽象边界设计得对吗？
""".strip())


# ─────────────────────────────────────────────────────────
# 模型注册表
# ─────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_model(name: str, default_config: dict = None):
    """
    装饰器：注册模型到全局表

    用法：
        @register_model('my_cnn', default_config={'hidden': 64})
        class MyCNN(nn.Module):
            ...
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = {
            'class': cls,
            'default_config': default_config or {},
        }
        return cls
    return decorator

def build_model(name: str, **kwargs) -> nn.Module:
    """根据名称构建模型，自动填充默认配置"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: '{name}'。可用: {list(MODEL_REGISTRY.keys())}")
    entry = MODEL_REGISTRY[name]
    config = {**entry['default_config'], **kwargs}
    return entry['class'](**config)


# ─────────────────────────────────────────────────────────
# 数据集注册表
# ─────────────────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_dataset(name: str, default_config: dict = None):
    """装饰器：注册数据集"""
    def decorator(cls):
        DATASET_REGISTRY[name] = {
            'class': cls,
            'default_config': default_config or {},
        }
        return cls
    return decorator

def build_dataset(name: str, **kwargs):
    """根据名称构建数据集"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: '{name}'。可用: {list(DATASET_REGISTRY.keys())}")
    entry = DATASET_REGISTRY[name]
    config = {**entry['default_config'], **kwargs}
    return entry['class'](**config)


# ─────────────────────────────────────────────────────────
# 任务注册表
# ─────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_task(name: str, loss_fn=None, metrics=None, default_config: dict = None):
    """
    装饰器：注册任务

    任务 = 损失函数 + 评估指标 + 推理函数
    """
    def decorator(cls):
        TASK_REGISTRY[name] = {
            'class': cls,
            'loss_fn': loss_fn,
            'metrics': metrics or {},
            'default_config': default_config or {},
        }
        return cls
    return decorator

def build_task(name: str, **kwargs):
    """根据名称构建任务"""
    if name not in TASK_REGISTRY:
        raise ValueError(f"未知任务: '{name}'。可用: {list(TASK_REGISTRY.keys())}")
    entry = TASK_REGISTRY[name]
    config = {**entry['default_config'], **kwargs}
    return entry['class'](loss_fn=entry['loss_fn'], metrics=entry['metrics'], **config)

# ============================================================
# 代码段 2
# ============================================================

# ─────────────────────────────────────────────────────────
# 注册内置模型
# ─────────────────────────────────────────────────────────

@register_model('mlp', default_config={'input_dim': 784, 'hidden_dim': 256, 'output_dim': 10})
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    def forward(self, x):
        return self.net(x)


@register_model('cnn', default_config={'in_channels': 1, 'num_classes': 10})
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


@register_model('lstm', default_config={'vocab_size': 1000, 'embed_dim': 64, 'hidden_dim': 128, 'num_classes': 10})
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64, hidden_dim=128, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h)


@register_model('transformer', default_config={'vocab_size': 1000, 'd_model': 128, 'nhead': 4, 'num_layers': 2, 'num_classes': 10})
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, nhead=4, num_layers=2, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        seq_len = x.shape[1]
        emb = self.embedding(x) * (self.embedding.embedding_dim ** 0.5) + self.pos_enc[:, :seq_len, :]
        out = self.encoder(emb)
        return self.fc(out.mean(dim=1))


# ─────────────────────────────────────────────────────────
# 注册内置数据集
# ─────────────────────────────────────────────────────────

@register_dataset('mnist', default_config={'root': './data', 'train': True})
class MNISTDataset(Dataset):
    def __init__(self, root='./data', train=True):
        import torchvision
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.data = torchvision.datasets.MNIST(root, train=train, download=True, transform=transform)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


@register_dataset('synthetic_classification', default_config={'n_samples': 1000, 'n_features': 784, 'n_classes': 10})
class SyntheticDataset(Dataset):
    def __init__(self, n_samples=1000, n_features=784, n_classes=10):
        self.X = torch.randn(n_samples, n_features)
        self.y = torch.randint(0, n_classes, (n_samples,))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────
# 注册内置任务
# ─────────────────────────────────────────────────────────

@register_task('classification', loss_fn=nn.CrossEntropyLoss(),
               metrics={'accuracy': lambda pred, target: (pred.argmax(1) == target).float().mean().item()})
class ClassificationTask:
    def __init__(self, loss_fn=None, metrics=None):
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.metrics = metrics or {'accuracy': lambda p, t: (p.argmax(1) == t).float().mean().item()}

    def compute_loss(self, logits, targets):
        return self.loss_fn(logits, targets)

    def compute_metrics(self, logits, targets):
        return {name: fn(logits, targets) for name, fn in self.metrics.items()}


@register_task('regression', loss_fn=nn.MSELoss(),
               metrics={'mse': lambda pred, target: F.mse_loss(pred, target).item(),
                        'mae': lambda pred, target: F.l1_loss(pred, target).item()})
class RegressionTask:
    def __init__(self, loss_fn=None, metrics=None):
        self.loss_fn = loss_fn or nn.MSELoss()
        self.metrics = metrics or {
            'mse': lambda p, t: F.mse_loss(p, t).item(),
            'mae': lambda p, t: F.l1_loss(p, t).item(),
        }

    def compute_loss(self, pred, targets):
        return self.loss_fn(pred, targets)

    def compute_metrics(self, pred, targets):
        return {name: fn(pred, targets) for name, fn in self.metrics.items()}

# ============================================================
# 代码段 3
# ============================================================

class HookSystem:
    """
    钩子系统：在训练流程的关键点插入自定义逻辑

    支持的钩子点：
    - on_train_start:    训练开始前
    - on_epoch_start:    每个 epoch 开始
    - on_batch_start:    每个批次开始
    - on_batch_end:      每个批次结束
    - on_epoch_end:      每个 epoch 结束
    - on_train_end:      训练结束
    """

    def __init__(self):
        self._hooks: Dict[str, list] = {}

    def register(self, hook_name: str, fn: Callable):
        """注册钩子函数"""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(fn)

    def fire(self, hook_name: str, context: dict = None):
        """触发钩子"""
        context = context or {}
        for fn in self._hooks.get(hook_name, []):
            fn(context)

    def clear(self, hook_name: str = None):
        """清除钩子"""
        if hook_name:
            self._hooks.pop(hook_name, None)
        else:
            self._hooks.clear()


# 内置钩子示例
def learning_rate_logger(context):
    """记录学习率"""
    epoch = context.get('epoch', 0)
    lr = context.get('optimizer').param_groups[0]['lr']
    print(f"  [Hook] Epoch {epoch} 学习率: {lr:.6f}")


def gradient_monitor(context):
    """监控梯度范数"""
    model = context.get('model')
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm > 100:
        print(f"  [Hook] ⚠ 梯度范数过大: {total_norm:.2f}")
    elif total_norm < 1e-6:
        print(f"  [Hook] ⚠ 梯度消失: {total_norm:.2e}")


def early_stopping_hook(patience=5, min_delta=0.001):
    """生成早停钩子"""
    best_loss = float('inf')
    counter = 0

    def hook(context):
        nonlocal best_loss, counter
        val_loss = context.get('val_loss', float('inf'))
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"  [Hook] 早停触发！{patience} 个 epoch 无改善")
                context['should_stop'] = True

    return hook

# ============================================================
# 代码段 4
# ============================================================

import yaml

def create_experiment_config():
    """生成实验配置模板"""
    print_learning_guide()
    config = {
        'experiment': {
            'name': 'mnist_classification',
            'seed': 42,
            'device': 'auto',  # auto/cuda/cpu
        },
        'model': {
            'name': 'cnn',
            'params': {
                'in_channels': 1,
                'num_classes': 10,
            },
        },
        'dataset': {
            'name': 'mnist',
            'params': {
                'root': './data',
            },
            'batch_size': 64,
            'val_ratio': 0.1,
            'num_workers': 0,
        },
        'task': {
            'name': 'classification',
        },
        'training': {
            'epochs': 20,
            'optimizer': 'adam',
            'lr': 0.001,
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'scheduler': 'cosine',
        },
        'hooks': {
            'on_epoch_end': ['lr_logger', 'gradient_monitor'],
            'early_stopping': {'patience': 10, 'min_delta': 0.001},
        },
        'output': {
            'save_dir': './experiments',
            'save_best': True,
            'plot_curves': True,
        },
    }

    yaml_str = yaml.dump(config, default_flow_style=False, allow_unicode=True)
    print("实验配置模板:")
    print(yaml_str)
    return config


if __name__ == "__main__":
    config = create_experiment_config()

# ============================================================
# 代码段 5
# ============================================================

# ─────────────────────────────────────────────────────────
# 示例：添加一个新模型（ResNet）
# ─────────────────────────────────────────────────────────

@register_model('resnet', default_config={'in_channels': 3, 'num_classes': 10})
class ResNetCustom(nn.Module):
    """新添加的 ResNet 模型"""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────────────────────
# 示例：添加一个新数据集
# ─────────────────────────────────────────────────────────

@register_dataset('cifar10', default_config={'root': './data', 'train': True})
class CIFAR10Dataset(Dataset):
    def __init__(self, root='./data', train=True):
        import torchvision
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip() if train else lambda x: x,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.2470, 0.2435, 0.2616)),
        ])
        self.data = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=transform)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────────────────
# 示例：添加一个新任务（多标签分类）
# ─────────────────────────────────────────────────────────

@register_task('multilabel',
               loss_fn=nn.BCEWithLogitsLoss(),
               metrics={'f1_micro': lambda p, t: 0.0})  # 占位
class MultiLabelTask:
    def __init__(self, loss_fn=None, metrics=None):
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()
        self.metrics = metrics or {}

    def compute_loss(self, logits, targets):
        return self.loss_fn(logits, targets.float())

    def compute_metrics(self, logits, targets):
        preds = (logits.sigmoid() > 0.5).float()
        exact_match = (preds == targets.float()).all(dim=1).float().mean().item()
        return {'exact_match': exact_match}


# ─────────────────────────────────────────────────────────
# 验证：切换模型只需改一行
# ─────────────────────────────────────────────────────────

def demo_switch_models():
    """演示：同一训练代码，切换不同模型"""
    print_learning_guide()
    print("可用模型:", list(MODEL_REGISTRY.keys()))
    print("可用数据集:", list(DATASET_REGISTRY.keys()))
    print("可用任务:", list(TASK_REGISTRY.keys()))

    # 只改这一行，就能切换模型！
    for model_name in ['mlp', 'cnn', 'lstm', 'transformer']:
        model = build_model(model_name)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {model_name}: {params:,} 参数")


if __name__ == "__main__":
    demo_switch_models()

# ============================================================
# 代码段 6
# ============================================================

import importlib
import os

def auto_discover_plugins(plugin_dir='plugins'):
    """
    自动发现并加载插件目录中的模块

    约定：
    - plugins/ 目录下每个 .py 文件是一个插件
    - 插件文件中使用 @register_model 等装饰器注册组件
    - 框架自动导入所有插件，组件自动注册
    """
    if not os.path.exists(plugin_dir):
        print(f"插件目录 {plugin_dir} 不存在")
        return

    for filename in os.listdir(plugin_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, os.path.join(plugin_dir, filename)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"已加载插件: {module_name}")
            except Exception as e:
                print(f"加载插件 {module_name} 失败: {e}")
