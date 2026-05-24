# 项目模板：比赛 / 作业 / 个人项目完整结构

## 1. 项目目录结构

```
deep_learning_project/
│
├── README.md                    ← 项目说明
├── requirements.txt             ← 依赖列表
├── .gitignore                   ← Git 忽略规则
├── config.yaml                  ← 实验配置（核心！）
│
├── data/
│   ├── raw/                     ← 原始数据（不入 Git）
│   ├── processed/               ← 预处理后的数据
│   └── __init__.py
│
├── models/
│   ├── __init__.py
│   ├── registry.py              ← 模型注册表
│   ├── cnn.py
│   ├── rnn.py
│   ├── transformer.py
│   └── mlp.py
│
├── datasets/
│   ├── __init__.py
│   ├── registry.py              ← 数据集注册表
│   ├── image_dataset.py
│   ├── text_dataset.py
│   └── sequence_dataset.py
│
├── tasks/
│   ├── __init__.py
│   ├── registry.py              ← 任务注册表
│   ├── classification.py
│   ├── regression.py
│   └── seq2seq.py
│
├── training/
│   ├── __init__.py
│   ├── runner.py                ← ExperimentRunner
│   ├── hooks.py                 ← 钩子系统
│   └── utils.py                 ← 梯度裁剪、学习率等
│
├── visualization/
│   ├── __init__.py
│   ├── curves.py                ← 训练曲线
│   ├── attention.py             ← 注意力可视化
│   └── gradcam.py               ← GradCAM
│
├── app.py                       ← Streamlit 演示
│
├── scripts/
│   ├── train.py                 ← 训练入口脚本
│   ├── evaluate.py              ← 评估脚本
│   ├── predict.py               ← 推理脚本
│   └── download_data.py         ← 数据下载
│
├── notebooks/
│   ├── 01_eda.ipynb             ← 探索性分析
│   ├── 02_baseline.ipynb        ← 基线模型
│   └── 03_analysis.ipynb        ← 结果分析
│
└── experiments/                 ← 实验输出（不入 Git）
    ├── exp_001/
    │   ├── config.json
    │   ├── best.pt
    │   ├── training_log.csv
    │   ├── training_curves.png
    │   └── final_result.json
    └── exp_002/
        └── ...
```

---

## 2. 核心文件内容

### 2.1 config.yaml（实验配置）

```yaml
# 实验配置 — 改这里就能切换所有设置

experiment:
  name: "mnist_cnn_baseline"
  seed: 42
  device: auto  # auto / cuda / cpu

model:
  name: cnn
  params:
    in_channels: 1
    num_classes: 10

dataset:
  name: mnist
  params:
    root: ./data
  batch_size: 64
  val_ratio: 0.1
  num_workers: 0

task:
  name: classification
  params: {}

training:
  epochs: 20
  optimizer: adam
  lr: 0.001
  weight_decay: 1.0e-4
  grad_clip: 1.0
  scheduler: cosine
  patience: 10

output:
  save_dir: ./experiments/exp_001
  save_best: true
  save_every: 10
  plot_curves: true
```

### 2.2 scripts/train.py（训练入口）

```python
#!/usr/bin/env python3
"""训练入口脚本"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='深度学习训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--model', type=str, default=None,
                        help='覆盖模型名称')
    parser.add_argument('--lr', type=float, default=None,
                        help='覆盖学习率')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖训练轮数')
    parser.add_argument('--gpu', type=int, default=None,
                        help='指定 GPU 编号')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 命令行覆盖
    if args.model:
        config['model']['name'] = args.model
    if args.lr:
        config['training']['lr'] = args.lr
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.gpu is not None:
        config['experiment']['device'] = f'cuda:{args.gpu}'

    # 设置随机种子
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 构建组件
    from models.registry import build_model
    from datasets.registry import build_dataset
    from tasks.registry import build_task
    from training.runner import ExperimentRunner

    model = build_model(config['model']['name'], **config['model'].get('params', {}))
    dataset = build_dataset(config['dataset']['name'], **config['dataset'].get('params', {}))
    task = build_task(config['task']['name'], **config['task'].get('params', {}))

    # 分割数据集
    from torch.utils.data import DataLoader, random_split
    val_ratio = config['dataset'].get('val_ratio', 0.1)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    batch_size = config['dataset'].get('batch_size', 64)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)

    # 运行训练
    runner = ExperimentRunner(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=task.loss_fn,
        metrics=task.metrics,
    )
    runner.run()


if __name__ == '__main__':
    main()
```

### 2.3 scripts/evaluate.py（评估脚本）

```python
#!/usr/bin/env python3
"""评估脚本"""

import argparse
import torch
import yaml
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, default=None, help='配置文件（默认从检查点加载）')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    # 加载检查点
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    # 加载配置
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = ckpt.get('config', {})

    # 构建模型
    from models.registry import build_model
    model = build_model(config['model']['name'], **config['model'].get('params', {}))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"模型: {config['model']['name']}")
    print(f"检查点 epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"检查点指标: {ckpt.get('metric', 'N/A')}")

    # TODO: 加载测试数据并评估


if __name__ == '__main__':
    main()
```

### 2.4 requirements.txt

```
torch>=2.0
torchvision>=0.15
numpy>=1.24
matplotlib>=3.7
pyyaml>=6.0
streamlit>=1.30
scikit-learn>=1.3
pandas>=2.0
tqdm>=4.65
tensorboard>=2.14
```

### 2.5 .gitignore

```gitignore
# 数据
data/raw/
data/processed/
*.csv
*.pkl

# 实验
experiments/
wandb/
runs/

# 模型权重
*.pt
*.pth
*.bin

# Python
__pycache__/
*.pyc
.venv/
env/

# IDE
.idea/
.vscode/
*.swp

# 系统
.DS_Store
Thumbs.db
```

---

## 3. 快速启动命令

```bash
# 1. 创建项目
cp -r deep_learning_project/ my_project/
cd my_project

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据
python scripts/download_data.py

# 4. 训练（默认配置）
python scripts/train.py --config config.yaml

# 5. 切换模型（只改一行）
python scripts/train.py --model lstm

# 6. 调学习率
python scripts/train.py --lr 0.0001

# 7. 指定 GPU
python scripts/train.py --gpu 0

# 8. 评估
python scripts/evaluate.py --checkpoint experiments/exp_001/best.pt

# 9. 启动演示
streamlit run app.py

# 10. Jupyter 分析
jupyter notebook notebooks/01_eda.ipynb
```

---

## 4. 比赛专用技巧

### 4.1 K-Fold 交叉验证

```python
from sklearn.model_selection import KFold

def kfold_train(config, n_folds=5):
    """K-Fold 训练（比赛常用）"""
    from datasets.registry import build_dataset

    dataset = build_dataset(config['dataset']['name'])
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n===== Fold {fold+1}/{n_folds} =====")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # 修改保存目录
        fold_config = {**config}
        fold_config['output'] = {**config['output'],
                                  'save_dir': f"{config['output']['save_dir']}/fold_{fold}"}

        runner = ExperimentRunner(fold_config, train_loader=..., val_loader=...)
        runner.run()
        fold_results.append(runner.best_val_metric)

    print(f"\nK-Fold 平均: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    return fold_results
```

### 4.2 模型集成

```python
def ensemble_predict(model_paths, dataloader, device='cuda'):
    """多模型集成预测（比赛常用）"""
    from models.registry import build_model

    all_probs = []

    for path in model_paths:
        ckpt = torch.load(path, map_location=device)
        config = ckpt['config']
        model = build_model(config['model']['name'], **config['model']['params'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for x, _ in dataloader:
                logits = model(x.to(device))
                probs = F.softmax(logits, dim=-1)
                fold_probs.append(probs.cpu())
        all_probs.append(torch.cat(fold_probs))

    # 平均概率
    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs.argmax(dim=-1)
```

---

## 小结

| 文件 | 作用 | 修改频率 |
|------|------|----------|
| config.yaml | 实验配置 | 每次实验改 |
| models/*.py | 模型定义 | 加新模型时改 |
| datasets/*.py | 数据集 | 加新数据时改 |
| scripts/train.py | 训练入口 | 很少改 |
| app.py | Streamlit 演示 | 很少改 |
| requirements.txt | 依赖 | 加新库时改 |
