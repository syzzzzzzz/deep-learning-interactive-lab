# 第十四章：模块化结构——切换模型只需改一行

---

## 14.1 模块化设计原则

好的框架应该让你：
- 换模型：改一行 `model = ...`
- 换数据：改一行 `dataset = ...`
- 换任务：改一行 `criterion = ...`
- 其余代码完全不动

---

## 14.1.1 统一模型注册表

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Type, Optional, Callable, Any
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────
# 模型注册表：用装饰器注册所有模型
# ─────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    """装饰器：将模型类注册到全局注册表"""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def build_model(name: str, **kwargs) -> nn.Module:
    """从注册表构建模型"""
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"未知模型: '{name}'。可用模型: {available}")
    return MODEL_REGISTRY[name](**kwargs)


# ─────────────────────────────────────────────────────────
# 注册所有模型
# ─────────────────────────────────────────────────────────

@register_model('mlp')
class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim=2, hidden_dims=[64, 64], output_dim=2,
                 dropout=0.0, activation='relu'):
        super().__init__()
        act = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}[activation]
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), act()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


@register_model('cnn')
class SimpleCNN(nn.Module):
    """简单 CNN（图像分类）"""
    def __init__(self, in_channels=1, num_classes=10, base_ch=32):
        super().__init__()
        c = base_ch
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c*2, 3, padding=1), nn.BatchNorm2d(c*2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c*2*16, num_classes))
    def forward(self, x):
        return self.head(self.features(x))


@register_model('lstm')
class LSTMClassifier(nn.Module):
    """LSTM 序列分类器"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 num_classes=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


@register_model('transformer_encoder')
class TransformerEncoderClassifier(nn.Module):
    """Transformer Encoder 分类器"""
    def __init__(self, vocab_size=100, d_model=64, n_heads=4,
                 n_layers=2, num_classes=2, max_len=50, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, num_classes)
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos(pos)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


@register_model('resnet')
class MiniResNet(nn.Module):
    """Mini ResNet（图像分类）"""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        def res_block(in_ch, out_ch, stride=1):
            layers = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            shortcut = nn.Identity() if (in_ch == out_ch and stride == 1) else \
                nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                               nn.BatchNorm2d(out_ch))
            class Block(nn.Module):
                def forward(self, x):
                    return nn.functional.relu(layers(x) + shortcut(x))
            b = Block()
            b.layers = layers
            b.shortcut = shortcut
            return b

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
            res_block(16, 16), res_block(16, 32, stride=2),
            res_block(32, 64, stride=2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────
# 统一训练配置（dataclass）
# ─────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """训练配置：所有超参数集中管理"""
    # 模型
    model_name: str = 'mlp'
    model_kwargs: dict = field(default_factory=dict)

    # 训练
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 10

    # 调度器
    scheduler: str = 'cosine'   # 'cosine', 'step', 'none'
    warmup_epochs: int = 5

    # 设备
    device: str = 'auto'

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─────────────────────────────────────────────────────────
# 模块化训练器
# ─────────────────────────────────────────────────────────

class ModularTrainer:
    """
    模块化训练器：切换模型/数据/任务只需改配置

    使用方法：
        config = TrainConfig(model_name='mlp', model_kwargs={...}, epochs=50)
        trainer = ModularTrainer(config)
        history = trainer.fit(train_loader, val_loader)
        trainer.compare_models(['mlp', 'lstm', 'cnn'], model_kwargs_list, ...)
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.model = build_model(config.model_name, **config.model_kwargs)
        self.model.to(config.device)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def _make_optimizer_scheduler(self, n_steps_per_epoch: int):
        opt = torch.optim.AdamW(self.model.parameters(),
                                 lr=self.config.lr,
                                 weight_decay=self.config.weight_decay)
        total = self.config.epochs * n_steps_per_epoch
        warmup = self.config.warmup_epochs * n_steps_per_epoch

        if self.config.scheduler == 'cosine':
            def lr_lambda(step):
                if step < warmup:
                    return step / max(1, warmup)
                progress = (step - warmup) / max(1, total - warmup)
                return 0.5 * (1 + np.cos(np.pi * progress))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        elif self.config.scheduler == 'step':
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.config.epochs // 3, gamma=0.1)
        else:
            sched = None

        return opt, sched

    def fit(self, train_loader, val_loader=None,
            criterion=None, verbose: bool = True) -> dict:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        opt, sched = self._make_optimizer_scheduler(len(train_loader))
        best_val = float('inf')
        patience_cnt = 0
        best_state = None

        for epoch in range(self.config.epochs):
            # 训练
            self.model.train()
            t_losses, t_correct, t_total = [], 0, 0
            for batch in train_loader:
                x, y = [b.to(self.config.device) for b in batch]
                logits = self.model(x)
                loss = criterion(logits, y)
                opt.zero_grad()
                loss.backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                opt.step()
                if sched and self.config.scheduler == 'cosine':
                    sched.step()
                t_losses.append(loss.item())
                if logits.dim() > 1 and logits.shape[1] > 1:
                    t_correct += (logits.argmax(1) == y).sum().item()
                    t_total += len(y)

            t_loss = np.mean(t_losses)
            t_acc = t_correct / t_total if t_total > 0 else 0
            self.history['train_loss'].append(t_loss)
            self.history['train_acc'].append(t_acc)

            # 验证
            if val_loader:
                self.model.eval()
                v_losses, v_correct, v_total = [], 0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        x, y = [b.to(self.config.device) for b in batch]
                        logits = self.model(x)
                        v_losses.append(criterion(logits, y).item())
                        if logits.dim() > 1 and logits.shape[1] > 1:
                            v_correct += (logits.argmax(1) == y).sum().item()
                            v_total += len(y)
                v_loss = np.mean(v_losses)
                v_acc = v_correct / v_total if v_total > 0 else 0
                self.history['val_loss'].append(v_loss)
                self.history['val_acc'].append(v_acc)

                if v_loss < best_val:
                    best_val = v_loss
                    patience_cnt = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_cnt += 1

                if verbose and (epoch + 1) % max(1, self.config.epochs // 5) == 0:
                    print(f"Epoch {epoch+1:4d}  train={t_loss:.4f}/{t_acc:.1%}  "
                          f"val={v_loss:.4f}/{v_acc:.1%}  "
                          f"lr={opt.param_groups[0]['lr']:.2e}")

                if patience_cnt >= self.config.patience:
                    if verbose:
                        print(f"早停于 epoch {epoch+1}")
                    break

            if sched and self.config.scheduler == 'step':
                sched.step()

        if best_state:
            self.model.load_state_dict(best_state)

        return self.history

    def plot_history(self, figsize=(12, 4)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for key, ax, title in [
            ('loss', axes[0], '损失曲线'),
            ('acc', axes[1], '准确率曲线'),
        ]:
            if self.history[f'train_{key}']:
                ax.plot(self.history[f'train_{key}'], label='训练', linewidth=1.5)
            if self.history[f'val_{key}']:
                ax.plot(self.history[f'val_{key}'], label='验证', linewidth=1.5)
            ax.set_title(f'{self.config.model_name} — {title}', fontsize=11)
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'history_{self.config.model_name}.png', dpi=120, bbox_inches='tight')
        plt.show()


# ─────────────────────────────────────────────────────────
# 一键对比多个模型
# ─────────────────────────────────────────────────────────

def compare_models_on_task(model_configs: list, train_loader, val_loader,
                            criterion=None, figsize=(14, 5)):
    """
    在同一任务上对比多个模型

    model_configs: [{'name': 'mlp', 'kwargs': {...}, 'label': '...'}]
    """
    all_histories = {}

    for cfg in model_configs:
        label = cfg.get('label', cfg['name'])
        print(f"\n训练: {label}")
        train_cfg = TrainConfig(
            model_name=cfg['name'],
            model_kwargs=cfg.get('kwargs', {}),
            epochs=cfg.get('epochs', 50),
            lr=cfg.get('lr', 1e-3),
        )
        trainer = ModularTrainer(train_cfg)
        history = trainer.fit(train_loader, val_loader, criterion=criterion, verbose=False)
        all_histories[label] = history
        final_val = history['val_loss'][-1] if history['val_loss'] else float('nan')
        final_acc = history['val_acc'][-1] if history['val_acc'] else float('nan')
        print(f"  最终验证: loss={final_val:.4f}  acc={final_acc:.1%}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for label, hist in all_histories.items():
        if hist['val_loss']:
            axes[0].plot(hist['val_loss'], linewidth=1.5, label=label, alpha=0.85)
        if hist['val_acc']:
            axes[1].plot(hist['val_acc'], linewidth=1.5, label=label, alpha=0.85)

    axes[0].set_title('验证损失对比', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].set_title('验证准确率对比', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('准确率')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.suptitle('多模型对比实验', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return all_histories


# ─────────────────────────────────────────────────────────
# 完整演示
# ─────────────────────────────────────────────────────────

def demo_modular_framework():
    torch.manual_seed(42)
    print("可用模型:", list(MODEL_REGISTRY.keys()))

    # 生成数据
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.randn(800, 2)
    y = ((X[:, 0]**2 + X[:, 1]**2) < 1).long()  # 圆形边界

    split = 640
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds   = TensorDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128)

    # 对比三种模型
    model_configs = [
        {'name': 'mlp', 'kwargs': {'input_dim': 2, 'hidden_dims': [32, 32], 'output_dim': 2},
         'label': 'MLP-小', 'epochs': 60},
        {'name': 'mlp', 'kwargs': {'input_dim': 2, 'hidden_dims': [128, 128, 64], 'output_dim': 2},
         'label': 'MLP-大', 'epochs': 60},
        {'name': 'mlp', 'kwargs': {'input_dim': 2, 'hidden_dims': [64, 64], 'output_dim': 2,
                                    'dropout': 0.3},
         'label': 'MLP+Dropout', 'epochs': 60},
    ]

    histories = compare_models_on_task(model_configs, train_loader, val_loader)
    return histories

histories = demo_modular_framework()
```

---

## 14.2 快速切换模型模板

```python
# ─────────────────────────────────────────────────────────
# 只需改这一行，其余代码完全不动
# ─────────────────────────────────────────────────────────

# 选项 A：MLP
config = TrainConfig(
    model_name='mlp',
    model_kwargs={'input_dim': 20, 'hidden_dims': [64, 64], 'output_dim': 2},
    epochs=50, lr=1e-3,
)

# 选项 B：CNN（改这一行）
# config = TrainConfig(
#     model_name='cnn',
#     model_kwargs={'in_channels': 1, 'num_classes': 10},
#     epochs=20, lr=1e-3,
# )

# 选项 C：LSTM（改这一行）
# config = TrainConfig(
#     model_name='lstm',
#     model_kwargs={'input_size': 10, 'hidden_size': 64, 'num_classes': 2},
#     epochs=30, lr=5e-4,
# )

# 选项 D：Transformer（改这一行）
# config = TrainConfig(
#     model_name='transformer_encoder',
#     model_kwargs={'vocab_size': 1000, 'd_model': 64, 'n_heads': 4, 'num_classes': 2},
#     epochs=30, lr=3e-4,
# )

# 训练代码完全相同
trainer = ModularTrainer(config)
# history = trainer.fit(train_loader, val_loader)
# trainer.plot_history()
print(f"模型: {config.model_name}")
print(f"参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")
```

---

## 小结

| 组件 | 作用 | 关键特性 |
|------|------|----------|
| MODEL_REGISTRY | 模型注册表 | `@register_model` 装饰器注册 |
| build_model | 工厂函数 | 一行代码构建任意注册模型 |
| TrainConfig | 配置 dataclass | 所有超参数集中管理 |
| ModularTrainer | 统一训练器 | 支持早停/调度/梯度裁剪 |
| compare_models_on_task | 多模型对比 | 同一任务自动对比所有模型 |
