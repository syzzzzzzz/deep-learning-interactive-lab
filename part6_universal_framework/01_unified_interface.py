"""
自动生成自: part6_universal_framework\01_unified_interface.md
可独立运行的 Python 源码
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any

# ─────────────────────────────────────────────────────────
# 通用数据集包装器
# ─────────────────────────────────────────────────────────

class TensorDatasetWrapper(Dataset):
    """
    将 numpy 数组或 torch 张量包装成 Dataset

    支持：
    - 自动类型转换
    - 可选的数据增强（transform）
    - 自动归一化

    使用方法：
        ds = TensorDatasetWrapper(X_np, y_np, normalize=True)
        train_ds, val_ds = ds.split(val_ratio=0.2)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    """

    def __init__(self, X, y,
                 x_dtype=torch.float32,
                 y_dtype=torch.float32,
                 normalize: bool = False,
                 transform: Optional[Callable] = None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        self.X = X.to(x_dtype)
        self.y = y.to(y_dtype)
        self.transform = transform

        if normalize:
            self.mean = self.X.mean(dim=0, keepdim=True)
            self.std = self.X.std(dim=0, keepdim=True) + 1e-8
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def split(self, val_ratio: float = 0.2, seed: int = 42):
        """按比例分割为训练集和验证集"""
        n_val = int(len(self) * val_ratio)
        n_train = len(self) - n_val
        generator = torch.Generator().manual_seed(seed)
        return random_split(self, [n_train, n_val], generator=generator)

    def get_loaders(self, batch_size: int = 32, val_ratio: float = 0.2,
                    num_workers: int = 0, seed: int = 42):
        """一步获取训练和验证 DataLoader"""
        train_ds, val_ds = self.split(val_ratio, seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2,
                                shuffle=False, num_workers=num_workers)
        return train_loader, val_loader


# ─────────────────────────────────────────────────────────
# 统一模型接口（Mixin）
# ─────────────────────────────────────────────────────────

class TrainableMixin:
    """
    为任意 nn.Module 添加便捷训练方法的 Mixin

    使用方法：
        class MyModel(nn.Module, TrainableMixin):
            ...

        model = MyModel()
        model.fit(train_loader, val_loader, epochs=50)
        model.save('best.pt')
        model.load('best.pt')
    """

    def fit(self, train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 50,
            lr: float = 1e-3,
            criterion=None,
            optimizer=None,
            scheduler=None,
            patience: int = 10,
            grad_clip: float = 1.0,
            device: str = 'auto',
            verbose: bool = True) -> Dict[str, list]:
        """
        一键训练

        返回训练历史字典：{'train_loss': [...], 'val_loss': [...], ...}
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        self.to(device)

        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)

        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # 训练阶段
            self.train()
            train_losses = []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = self(x)
                loss = criterion(out, y)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            # 验证阶段
            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        out = self(x)
                        loss = criterion(out, y)
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)
                history['val_loss'].append(val_loss)

                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = copy.deepcopy(self.state_dict())
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch+1:4d}/{epochs}  "
                          f"train={train_loss:.4f}  val={val_loss:.4f}  "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}")

                if patience_counter >= patience:
                    if verbose:
                        print(f"早停：{patience} 轮无改善，在 epoch {epoch+1} 停止")
                    break
            else:
                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch+1:4d}/{epochs}  train={train_loss:.4f}  "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}")

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loader is not None:
                        scheduler.step(val_loss)
                else:
                    scheduler.step()

        # 恢复最优权重
        if best_state is not None:
            self.load_state_dict(best_state)
            if verbose:
                print(f"已恢复最优模型（val_loss={best_val_loss:.4f}）")

        return history

    def save(self, path: str):
        """保存模型权重"""
        torch.save(self.state_dict(), path)
        print(f"模型已保存到 {path}")

    def load(self, path: str, device: str = 'cpu'):
        """加载模型权重"""
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"模型已从 {path} 加载")

    def predict(self, x, device: str = 'auto', batch_size: int = 256):
        """批量预测"""
        if device == 'auto':
            device = next(self.parameters()).device
        else:
            device = torch.device(device)
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        results = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size].to(device)
                results.append(self(batch).cpu())
        return torch.cat(results)

    def count_params(self) -> int:
        """统计可训练参数量"""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数量: {total:,}  ({total/1e6:.2f}M)")
        return total


# ─────────────────────────────────────────────────────────
# 预置常用模型（继承 TrainableMixin）
# ─────────────────────────────────────────────────────────

class MLP(nn.Module, TrainableMixin):
    """
    多层感知机（全连接网络）

    使用方法：
        model = MLP(input_dim=20, hidden_dims=[64, 64], output_dim=10)
        history = model.fit(train_loader, val_loader, epochs=50)
    """

    def __init__(self, input_dim: int,
                 hidden_dims: list,
                 output_dim: int,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 batch_norm: bool = False):
        super().__init__()

        act_map = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
        }
        Act = act_map.get(activation, nn.ReLU)

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module, TrainableMixin):
    """
    简单 CNN（适用于图像分类）

    使用方法：
        model = SimpleCNN(in_channels=1, num_classes=10)
        history = model.fit(train_loader, val_loader, epochs=20)
    """

    def __init__(self, in_channels: int = 1,
                 num_classes: int = 10,
                 base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c, c*2, 3, padding=1), nn.BatchNorm2d(c*2), nn.ReLU(),
            nn.Conv2d(c*2, c*2, 3, padding=1), nn.BatchNorm2d(c*2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c*2*16, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────
# 完整演示
# ─────────────────────────────────────────────────────────

def demo_unified_interface():
    """演示统一接口的使用"""
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("演示 1：MLP 分类（使用 TrainableMixin）")
    print("=" * 60)

    # 生成数据
    X = np.random.randn(1000, 20).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    # 一步创建 DataLoader
    ds = TensorDatasetWrapper(X, y, y_dtype=torch.long, normalize=True)
    train_loader, val_loader = ds.get_loaders(batch_size=64, val_ratio=0.2)

    # 创建模型
    model = MLP(input_dim=20, hidden_dims=[64, 64, 32], output_dim=2,
                activation='relu', dropout=0.2, batch_norm=True)
    model.count_params()

    # 一键训练
    history = model.fit(
        train_loader, val_loader,
        epochs=50, lr=1e-3,
        criterion=nn.CrossEntropyLoss(),
        patience=10,
        verbose=True,
    )

    # 保存和加载
    model.save('mlp_demo.pt')
    model.load('mlp_demo.pt')

    # 预测
    X_test = torch.randn(10, 20)
    preds = model.predict(X_test)
    print(f"\n预测输出形状: {preds.shape}")
    print(f"预测类别: {preds.argmax(dim=1).tolist()}")

    # 绘制训练曲线
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='训练损失')
    axes[0].plot(history['val_loss'], label='验证损失')
    axes[0].set_title('损失曲线', fontsize=11)
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['lr'])
    axes[1].set_title('学习率曲线（CosineAnnealing）', fontsize=11)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('学习率')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('统一接口训练演示', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('unified_interface_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    return model, history

if __name__ == "__main__":
    demo_unified_interface()

# ============================================================
# 代码段 2
# ============================================================

# ─────────────────────────────────────────────────────────
# 复制这段模板，5分钟搭建新项目
# ─────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import numpy as np

# 1. 准备数据
X_train = np.random.randn(800, 20).astype(np.float32)
y_train = (X_train[:, 0] > 0).astype(np.int64)
X_val   = np.random.randn(200, 20).astype(np.float32)
y_val   = (X_val[:, 0] > 0).astype(np.int64)

ds_train = TensorDatasetWrapper(X_train, y_train, y_dtype=torch.long, normalize=True)
ds_val   = TensorDatasetWrapper(X_val,   y_val,   y_dtype=torch.long, normalize=True)
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
val_loader   = torch.utils.data.DataLoader(ds_val,   batch_size=128)

# 2. 定义模型（继承 TrainableMixin 获得 .fit() 等方法）
class MyModel(nn.Module, TrainableMixin):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, x):
        return self.net(x)

# 3. 训练
# 取消注释下面几行即可运行这个快速模板。
# model = MyModel()
# model.count_params()
# history = model.fit(train_loader, val_loader, epochs=50, lr=1e-3, patience=10)

# 4. 保存
# model.save('my_model.pt')

# 5. 推理
# X_new = torch.randn(5, 20)
# preds = model.predict(X_new)
# print("预测结果:", preds.argmax(dim=1))
