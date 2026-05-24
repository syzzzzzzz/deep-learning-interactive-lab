# 迁移学习：预训练模型 / 微调策略 / 渐进式训练

## 1. 为什么迁移学习有效？

```
从零训练 CNN 需要：
- 数百万张标注图片
- 数百 GPU 小时
- 精心调参

迁移学习：
- 在大数据集（ImageNet）上预训练
- 在小数据集上微调
- 10-100x 更少的数据和计算

核心洞察：CNN 浅层学到的特征是通用的
┌─────────────────────────────────┐
│  深层：语义特征（物体、场景）    │  ← 需要微调
│  中层：纹理、形状、零件          │  ← 部分可复用
│  浅层：边缘、颜色、纹理          │  ← 完全可复用
│  输入：原始像素                  │
└─────────────────────────────────┘
```

---

## 2. 预训练模型库

### 2.1 PyTorch 内置模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt


def list_pretrained_models():
    """列出常用的预训练模型"""
    model_info = {
        'ResNet-18':    ('resnet18',    11.7,  69.8),
        'ResNet-50':    ('resnet50',    25.6,  76.1),
        'ResNet-101':   ('resnet101',   44.5,  77.4),
        'VGG-16':       ('vgg16',       138,   71.6),
        'MobileNetV3':  ('mobilenet_v3_small', 2.5, 67.4),
        'EfficientNet-B0': ('efficientnet_b0', 5.3, 77.1),
        'DenseNet-121':  ('densenet121', 8.0,  74.4),
    }

    print("常用预训练模型")
    print("=" * 65)
    print(f"{'模型':20s} {'参数量(M)':>10s} {'ImageNet Top-1':>15s}")
    print("-" * 65)
    for name, (fn, params, acc) in model_info.items():
        print(f"{name:20s} {params:10.1f} {acc:14.1f}%")

    return model_info


model_info = list_pretrained_models()
```

### 2.2 加载预训练模型

```python
def load_and_inspect_model():
    """加载 ResNet-18 并分析结构"""
    # 加载预训练权重
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.eval()

    # 查看模型结构
    print("\nResNet-18 结构：")
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            n_blocks = len(list(module.children()))
            print(f"  {name}: Sequential ({n_blocks} 个块)")
        else:
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {type(module).__name__} ({params:,} 参数)")

    # 查看最后全连接层
    print(f"\n分类头: {model.fc}")
    print(f"  输入维度: {model.fc.in_features}")
    print(f"  输出维度: {model.fc.out_features}  ← ImageNet 1000 类")

    # 模拟推理
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"\n输出形状: {tuple(out.shape)}")
    print(f"预测类别: {out.argmax(dim=1).item()}")

    return model


resnet18 = load_and_inspect_model()
```

---

## 3. 三种迁移学习策略

### 3.1 策略对比

```
策略 1：特征提取（冻结全部，只训练分类头）
  预训练层 [冻结] → 分类头 [训练]
  适用：数据很少（<1K 张），目标域与 ImageNet 相似

策略 2：微调最后几层
  预训练层 [冻结] → 中间层 [冻结] → 最后块 [训练] → 分类头 [训练]
  适用：数据中等（1K-10K 张），目标域与 ImageNet 略有不同

策略 3：全模型微调
  预训练层 [小学习率] → 分类头 [大学习率]
  适用：数据充足（>10K 张），目标域与 ImageNet 差异大
```

### 3.2 策略 1：特征提取

```python
def create_feature_extractor(num_classes=10):
    """
    策略 1：特征提取

    冻结所有预训练层，只替换分类头
    """
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换分类头
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # 只有 fc 层可训练
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"特征提取模式:")
    print(f"  可训练参数: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"  总参数: {total:,}")

    return model


model_fe = create_feature_extractor(num_classes=10)
```

### 3.3 策略 2：部分微调

```python
def create_partial_finetune(num_classes=10, unfreeze_from='layer4'):
    """
    策略 2：部分微调

    冻结浅层，只微调深层 + 分类头
    """
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻指定层及之后的所有层
    unfreeze = False
    for name, module in model.named_children():
        if name == unfreeze_from:
            unfreeze = True
        if unfreeze:
            for param in module.parameters():
                param.requires_grad = True

    # 替换分类头（总是可训练）
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"部分微调模式（解冻 {unfreeze_from} 起）:")
    print(f"  可训练参数: {trainable:,} ({trainable/total*100:.2f}%)")

    return model


model_pf = create_partial_finetune(unfreeze_from='layer4')
```

### 3.4 策略 3：差异化学习率微调

```python
def create_differential_lr(num_classes=10, base_lr=1e-3, head_lr=1e-2):
    """
    策略 3：差异化学习率

    预训练层用小学习率，分类头用大学习率
    """
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 替换分类头
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # 分组设置学习率
    pretrain_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'fc' in name:
            head_params.append(param)
        else:
            pretrain_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': pretrain_params, 'lr': base_lr},   # 预训练层：小学习率
        {'params': head_params, 'lr': head_lr},       # 分类头：大学习率
    ])

    print(f"差异化学习率:")
    print(f"  预训练层 lr={base_lr}")
    print(f"  分类头 lr={head_lr}")
    print(f"  学习率比: {head_lr/base_lr:.0f}x")

    return model, optimizer


model_dlr, optimizer_dlr = create_differential_lr()
```

---

## 4. 渐进式解冻训练

```python
class ProgressiveUnfreezer:
    """
    渐进式解冻

    训练流程：
    1. 冻结所有层，只训练分类头（epoch 1-3）
    2. 解冻最后一层（epoch 4-6）
    3. 解冻倒数第二层（epoch 7-9）
    4. 解冻所有层，降低学习率（epoch 10+）

    类似 FastAI 的策略
    """

    def __init__(self, model, num_classes=10):
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.freeze_schedule = [
            (0,  ['fc']),                          # 只训练分类头
            (3,  ['layer4', 'fc']),                # + 最后一个块
            (6,  ['layer3', 'layer4', 'fc']),      # + 倒数第二个块
            (9,  ['layer2', 'layer3', 'layer4', 'fc']),  # + 第三个块
            (12, 'all'),                           # 全部解冻
        ]

    def apply_freeze(self, epoch):
        """根据 epoch 决定哪些层解冻"""
        trainable_layers = ['fc']  # 分类头始终可训练

        for threshold, layers in self.freeze_schedule:
            if epoch >= threshold:
                if layers == 'all':
                    trainable_layers = 'all'
                else:
                    trainable_layers = layers

        # 应用冻结
        for name, param in self.model.named_parameters():
            if trainable_layers == 'all':
                param.requires_grad = True
            else:
                param.requires_grad = any(
                    name.startswith(layer) for layer in trainable_layers
                )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Epoch {epoch}: 解冻 {trainable_layers}, "
              f"可训练 {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")

        return trainable_layers

    def get_learning_rate(self, epoch):
        """随解冻程度降低学习率"""
        if epoch < 3:
            return 1e-2   # 只训练分类头，大学习率
        elif epoch < 6:
            return 5e-3
        elif epoch < 9:
            return 1e-3
        else:
            return 5e-4   # 全模型微调，小学习率


def demo_progressive_unfreeze():
    """演示渐进式解冻"""
    unfreezer = ProgressiveUnfreezer(num_classes=10)

    print("渐进式解冻计划")
    print("=" * 60)
    for epoch in range(15):
        lr = unfreezer.get_learning_rate(epoch)
        unfreezer.apply_freeze(epoch)
        print(f"  学习率: {lr}")


demo_progressive_unfreeze()
```

---

## 5. 迁移学习实验：策略对比

```python
def compare_transfer_strategies():
    """
    对比不同迁移学习策略的效果（示意）

    在 CIFAR-10 上的典型结果：
    - 从零训练: ~70% (5 epochs)
    - 特征提取: ~85% (5 epochs)
    - 部分微调: ~90% (5 epochs)
    - 渐进解冻: ~93% (15 epochs)
    """
    strategies = {
        '从零训练': {'acc': [35, 50, 58, 64, 70], 'color': '#C44E52'},
        '特征提取': {'acc': [65, 75, 80, 83, 85], 'color': '#4C72B0'},
        '部分微调': {'acc': [70, 80, 85, 88, 90], 'color': '#55A868'},
        '渐进解冻': {'acc': [65, 78, 85, 89, 93], 'color': '#DD8452'},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 准确率曲线
    epochs = range(1, 6)
    for name, info in strategies.items():
        axes[0].plot(epochs, info['acc'], 'o-', color=info['color'],
                     label=name, linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('测试准确率 (%)')
    axes[0].set_title('迁移学习策略对比\n（CIFAR-10 示意）', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(30, 100)

    # 数据量 vs 策略选择
    data_sizes = [100, 500, 1000, 5000, 10000, 50000]
    fe_acc =  [50, 70, 80, 85, 87, 88]
    ft_acc =  [30, 55, 75, 88, 92, 94]
    scratch = [10, 25, 40, 60, 72, 80]

    axes[1].plot(data_sizes, fe_acc, 'o-', color='#4C72B0', label='特征提取', linewidth=2)
    axes[1].plot(data_sizes, ft_acc, 's-', color='#55A868', label='微调', linewidth=2)
    axes[1].plot(data_sizes, scratch, '^-', color='#C44E52', label='从零训练', linewidth=2)
    axes[1].set_xlabel('训练样本数')
    axes[1].set_ylabel('测试准确率 (%)')
    axes[1].set_title('数据量 vs 策略选择', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    plt.tight_layout()
    plt.savefig('transfer_learning_comparison.png', dpi=150)
    plt.show()


compare_transfer_strategies()
```

---

## 6. 实用技巧

### 6.1 数据预处理对齐

```python
# 预训练模型期望的预处理（必须对齐！）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transfer_transforms(input_size=224):
    """
    迁移学习的数据预处理

    关键：必须使用 ImageNet 的均值和标准差
    """
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, val_transform
```

### 6.2 渐进式调整大小

```python
def progressive_resizing_schedule():
    """
    渐进式调整大小

    先用小图训练（快速），再逐步增大（精细）
    """
    schedule = [
        # (epoch_start, image_size, batch_size)
        (0,   128, 128),   # 小图，大批次
        (5,   192, 64),    # 中图
        (10,  224, 32),    # 原始大小
        (15,  288, 16),    # 超大图（可选）
    ]

    print("渐进式调整大小计划")
    print("=" * 50)
    print(f"{'Epoch':>6s} {'图像大小':>8s} {'批次':>6s} {'相对速度':>8s}")
    print("-" * 50)

    base_flops = 224 * 224
    for epoch, size, batch in schedule:
        relative_speed = base_flops / (size * size)
        print(f"{epoch:6d} {size:8d}×{size} {batch:6d} {relative_speed:7.1f}x")

    return schedule


progressive_resizing_schedule()
```

### 6.3 混合预训练权重

```python
def load_partial_weights(model, state_dict, skip_layers=None):
    """
    加载部分预训练权重

    场景：预训练模型和目标模型的某些层不匹配
    """
    if skip_layers is None:
        skip_layers = []

    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and not any(s in k for s in skip_layers)
    }

    print(f"预训练权重: {len(state_dict)} 个键")
    print(f"匹配并加载: {len(pretrained_dict)} 个键")
    print(f"跳过: {len(state_dict) - len(pretrained_dict)} 个键")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
```

---

## 7. 策略选择决策树

```
你的数据量是多少？
│
├─ < 1000 张
│   └─ 目标域与 ImageNet 相似？
│       ├─ 是 → 特征提取（策略1）
│       └─ 否 → 数据增强 + 特征提取，考虑 Few-shot 方法
│
├─ 1000 ~ 10000 张
│   └─ 目标域与 ImageNet 相似？
│       ├─ 是 → 部分微调（策略2）
│       └─ 否 → 差异化学习率微调（策略3）
│
└─ > 10000 张
    └─ 渐进式解冻 → 全模型微调
```

---

## 小结

| 策略 | 可训练层 | 学习率 | 数据需求 | 适用场景 |
|------|----------|--------|----------|----------|
| 特征提取 | 只分类头 | 大 | <1K | 相似域 |
| 部分微调 | 深层+分类头 | 中 | 1K-10K | 中等差异 |
| 差异化LR | 全部 | 分层 | >5K | 较大差异 |
| 渐进解冻 | 逐步增加 | 递减 | >1K | 最佳实践 |
