"""
自动生成自: part6_universal_framework\07_project_template.md
可独立运行的 Python 源码
"""

#!/usr/bin/env python3
"""训练入口脚本"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path


def print_learning_guide():
    print("""
学习导读：项目模板的价值在于复现和交接，不只是把训练代码拆成几个文件。

1. 训练入口怎么看
   - parse_args 允许从命令行覆盖 config、model、lr、epochs 和 gpu，适合复现实验或临时消融。
   - train_main 负责读取配置、设置随机种子、构建模型/数据/任务，再交给 ExperimentRunner。
   - 训练入口不应该写大量模型细节；模型、数据、任务应由 registry 和 config 决定。

2. 评估脚本怎么看
   - evaluate_main 从 checkpoint 读取 config 和 model_state_dict，只做加载和评估，不再训练。
   - 评估脚本必须明确 split=train/val/test，避免把验证集或测试集混在一起。
   - 真实项目要把“加载测试数据并评估”的占位逻辑补成确定实现，否则报告无法复现。

3. K-Fold 和集成怎么看
   - K-Fold 用多次不同切分估计模型稳定性，适合小数据、比赛和高风险验证。
   - ensemble_predict 平均多个 checkpoint 的概率，通常能降低方差，但会增加推理成本。
   - 每个 fold 必须有独立 save_dir，否则 checkpoint 和日志会互相覆盖。

工程坑案例：
   我见过模板复制后没有补完整评估数据加载逻辑，训练能跑、评估却只打印 checkpoint 信息，最后团队误以为项目已经闭环。
   模板落地时要逐项检查 config、seed、数据切分、checkpoint、training_log、final_result 是否都能追溯。

进阶思考：
   为什么训练脚本和评估脚本要分开？如果线上指标和离线验证集冲突，你会先查数据切分、指标定义，还是模型结构？
""".strip())


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


def train_main():
    args = parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"配置文件不存在，跳过训练示例: {config_path}")
        return

    with open(config_path, 'r') as f:
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


if __name__ == '__main__' and False:
    train_main()

# ============================================================
# 代码段 2
# ============================================================

#!/usr/bin/env python3
"""评估脚本"""

import argparse
import torch
import yaml
import json
from pathlib import Path


def evaluate_main():
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


if __name__ == '__main__' and False:
    evaluate_main()

# ============================================================
# 代码段 3
# ============================================================

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

# ============================================================
# 代码段 4
# ============================================================

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


if __name__ == '__main__':
    print_learning_guide()
    print("Project template snippets loaded. Use train_main() or evaluate_main() with real project files.")
