#!/usr/bin/env python3
"""
OpenClaw Self-Trainer - SFT 训练脚本

使用 DeepSpeed 对清洗后的对话数据进行 SFT 微调。
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path=None):
    """加载配置"""
    default = Path(__file__).parent.parent / "config" / "defaults.yaml"
    path = Path(config_path) if config_path else default
    with open(path) as f:
        return yaml.safe_load(f)


def check_new_data(data_dir, min_samples=50):
    """检查是否有足够的新数据需要训练"""
    cleaned_dir = Path(data_dir) / "cleaned"
    if not cleaned_dir.exists():
        return 0

    # 查找上次训练时间之后的数据文件
    state_file = Path(data_dir) / "reports" / "training_state.json"
    last_train_time = 0
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            last_train_time = state.get("last_train_time", 0)

    new_count = 0
    for f in sorted(cleaned_dir.glob("*.jsonl")):
        if f.stat().st_mtime > last_train_time:
            with open(f) as fh:
                new_count += sum(1 for _ in fh)

    return new_count


def merge_training_data(data_dir):
    """合并所有清洗后的数据到训练集"""
    cleaned_dir = Path(data_dir) / "cleaned"
    all_data = []

    for f in sorted(cleaned_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                item = json.loads(line.strip())
                all_data.append(item)

    # 切分数据集（test set 锁定）
    test_file = Path(data_dir) / "test.jsonl"
    if test_file.exists():
        with open(test_file) as f:
            existing_test = set(line.strip() for line in f)
    else:
        existing_test = set()

    config = load_config()
    test_ratio = config.get("dataset", {}).get("test_ratio", 0.07)
    val_ratio = config.get("dataset", {}).get("val_ratio", 0.08)

    # 排除已有的 test 数据
    trainable = [d for d in all_data if json.dumps(d, ensure_ascii=False) not in existing_test]

    # 切分
    import random
    random.shuffle(trainable)
    n = len(trainable)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_data = trainable[:n_test]
    val_data = trainable[n_test:n_test + n_val]
    train_data = trainable[n_test + n_val:]

    # 写入文件
    def write_jsonl(path, data):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps({
                    "input": item["input"],
                    "output": item["output"],
                    "category": item.get("category", "其他"),
                }, ensure_ascii=False) + "\n")

    write_jsonl(Path(data_dir) / "test.jsonl", test_data)
    write_jsonl(Path(data_dir) / "val.jsonl", val_data)
    write_jsonl(Path(data_dir) / "train.jsonl", train_data)

    print(f"📊 数据集切分: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return len(train_data)


def sft_train(data_dir, config_path=None):
    """执行 SFT 训练"""
    config = load_config(config_path)
    training = config.get("training", {})

    print("🚀 开始 SFT 训练...")
    print(f"   Epochs: {training.get('epochs', 3)}")
    print(f"   Batch size: {training.get('batch_size', 4)}")
    print(f"   Learning rate: {training.get('learning_rate', '2e-5')}")

    # TODO: 实现实际的 DeepSpeed SFT 训练
    # 1. 加载基座模型和 tokenizer
    # 2. 加载训练集，格式化为 instruction-response pairs
    # 3. 使用 PEFT (LoRA) 进行参数高效微调
    # 4. DeepSpeed ZeRO-2 加速
    # 5. 保存 checkpoint

    print("⚠️ 训练逻辑待实现 — 需要 GPU 环境")

    # 模拟保存
    models_dir = Path(data_dir).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = models_dir / version
    checkpoint_dir.mkdir(exist_ok=True)
    print(f"   Checkpoint 将保存到: {checkpoint_dir}")
    return str(checkpoint_dir)


def run(data_dir=None, config_path=None, auto=False, **kwargs):
    """主入口"""
    config = load_config(config_path)
    data_dir = Path(data_dir or config.get("paths", {}).get("data_dir", "~/.cache/self-trainer/data"))
    data_dir = data_dir.expanduser()

    if auto:
        min_samples = config.get("training", {}).get("min_new_samples", 50)
        new_count = check_new_data(data_dir, min_samples)
        print(f"📊 新增数据: {new_count} 条（阈值: {min_samples}）")
        if new_count < min_samples:
            print(f"🟡 数据不够（{new_count} < {min_samples}），跳过训练")
            return

    # 合并数据
    train_count = merge_training_data(data_dir)
    if train_count < 20:
        print("❌ 训练数据太少，至少需要 20 条")
        return

    # 训练
    checkpoint = sft_train(data_dir, config_path)

    # 更新训练状态
    state_file = data_dir / "reports" / "training_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump({
            "last_train_time": datetime.now().timestamp(),
            "checkpoint": checkpoint,
            "train_samples": train_count,
        }, f, indent=2)

    print("✅ 训练完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenClaw Self-Trainer SFT 训练")
    parser.add_argument("--auto", action="store_true", help="自动模式")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        config_path=args.config,
        auto=args.auto,
    )
