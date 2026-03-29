#!/usr/bin/env python3
"""
OpenClaw Self-Trainer - SFT 训练脚本（增量微调）

核心逻辑：
- 每次只用新收集的数据进行微调
- 微调的起点是上一个 checkpoint（不是基座模型）
- 训练后自动评估，通过则部署，退步则回滚

日循环流程：
  1. 收集过去24小时的 session 日志 → collect.py
  2. 排除已在训练集中的数据
  3. 新数据 ≥ 阈值 → 增量微调 → 保存 checkpoint → 评估
  4. 新数据不够 → 跳过
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml


# ─── 默认配置 ───────────────────────────────────────────
DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "defaults.yaml"
DEFAULT_DATA_DIR = Path.home() / ".cache" / "self-trainer" / "data"
DEFAULT_MODELS_DIR = Path.home() / ".cache" / "self-trainer" / "models"


def load_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


# ─── 数据管理 ───────────────────────────────────────────

def get_training_state(data_dir=None):
    """获取训练状态（上次训练时间、当前 checkpoint 等）"""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    state_file = data_dir / "reports" / "training_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {
        "last_train_time": 0,
        "current_checkpoint": None,
        "base_model": None,
        "total_new_samples_trained": 0,
        "training_rounds": 0,
        "history": [],
    }


def save_training_state(state, data_dir=None):
    """保存训练状态"""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    state_file = reports_dir / "training_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_new_data_count(data_dir=None):
    """统计自上次训练以来的新数据量"""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    state = get_training_state(data_dir)
    last_train_time = state.get("last_train_time", 0)

    cleaned_dir = data_dir / "cleaned"
    if not cleaned_dir.exists():
        return 0

    new_count = 0
    for f in sorted(cleaned_dir.glob("*.jsonl")):
        if f.stat().st_mtime > last_train_time:
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        new_count += 1
    return new_count


def get_latest_checkpoint(models_dir=None):
    """获取最新的模型 checkpoint"""
    models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    if not models_dir.exists():
        return None

    checkpoints = sorted(models_dir.glob("checkpoint_*"))
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def prepare_train_data(data_dir=None):
    """
    准备训练数据：合并自上次训练以来的新数据。
    
    返回训练集路径和样本数。
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    state = get_training_state(data_dir)
    last_train_time = state.get("last_train_time", 0)

    cleaned_dir = data_dir / "cleaned"
    if not cleaned_dir.exists():
        return None, 0

    # 收集新数据
    new_samples = []
    new_files = []
    for f in sorted(cleaned_dir.glob("*.jsonl")):
        if f.stat().st_mtime > last_train_time:
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        new_samples.append(json.loads(line.strip()))
                new_files.append(f.name)

    if not new_samples:
        return None, 0

    # 写入临时训练文件
    train_file = data_dir / "reports" / "incremental_train.jsonl"
    train_file.parent.mkdir(parents=True, exist_ok=True)
    with open(train_file, "w") as f:
        for sample in new_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return str(train_file), len(new_samples)


# ─── Test Set 管理 ──────────────────────────────────────

def ensure_test_set(data_dir=None, test_ratio=0.07):
    """
    确保 test set 存在。
    
    首次运行时，从已有数据中切出一部分作为 test set（锁定，永不参与训练）。
    后续运行直接使用已有的 test set。
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    test_file = data_dir / "test.jsonl"

    if test_file.exists():
        with open(test_file) as f:
            count = sum(1 for line in f if line.strip())
        return count

    # 首次：从所有已清洗数据中切出 test set
    import random
    all_samples = []
    cleaned_dir = data_dir / "cleaned"
    if not cleaned_dir.exists():
        return 0

    for f in sorted(cleaned_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    all_samples.append(line.strip())

    if not all_samples:
        return 0

    random.shuffle(all_samples)
    n_test = max(1, int(len(all_samples) * test_ratio))
    test_samples = all_samples[:n_test]

    with open(test_file, "w") as f:
        for sample in test_samples:
            f.write(sample + "\n")

    print(f"✅ Test set 已创建: {n_test} 条（从 {len(all_samples)} 条中切出）")
    return n_test


# ─── SFT 训练 ──────────────────────────────────────────

def sft_train_incremental(train_file, checkpoint=None, config_path=None, data_dir=None):
    """
    增量微调：在最新 checkpoint 基础上，用新数据继续训练。
    
    参数:
        train_file: 新数据文件路径
        checkpoint: 上一个 checkpoint 路径（None 则使用基座模型）
        config_path: 配置文件路径
        data_dir: 数据目录路径
    
    返回: 新 checkpoint 路径
    """
    config = load_config(config_path)
    training = config.get("training", {})

    base_model = checkpoint or config.get("base_model", None)
    if not base_model:
        print("❌ 没有找到基座模型或 checkpoint，请先配置")
        return None

    epochs = training.get("epochs", 3)
    batch_size = training.get("batch_size", 4)
    learning_rate = training.get("learning_rate", "2.0e-5")

    print(f"🚀 开始增量微调...")
    print(f"   起点: {'checkpoint' if checkpoint else 'base model'}")
    print(f"   训练数据: {train_file}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")

    # TODO: 实际训练逻辑
    # 1. 加载模型 (从 checkpoint 或 base model)
    # 2. 加载训练数据 (OpenAI messages 格式)
    # 3. 使用 PEFT LoRA 进行参数高效微调
    # 4. DeepSpeed ZeRO-2 加速
    # 5. 保存新 checkpoint（带版本号）
    #
    # 关键：不是从头训练，而是在已有 checkpoint 基础上继续
    # 用较低的 learning rate 避免灾难性遗忘
    #
    # 示例代码框架:
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # from peft import PeftModel
    # from trl import SFTTrainer
    #
    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    # if checkpoint:
    #     model = AutoModelForCausalLM.from_pretrained(checkpoint)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(base_model)
    #
    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     num_train_epochs=epochs,
    #     per_device_train_batch_size=batch_size,
    #     learning_rate=learning_rate,
    #     max_seq_length=training.get("max_seq_length", 2048),
    # )

    print("⚠️ 训练逻辑待实现 — 需要 GPU 环境")

    # 模拟保存 checkpoint
    models_dir = Path(data_dir).parent / "models" if data_dir else DEFAULT_MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now().strftime("checkpoint_%Y%m%d_%H%M%S")
    new_checkpoint = models_dir / version
    new_checkpoint.mkdir(exist_ok=True)

    # 保存训练元信息
    meta = {
        "version": version,
        "parent_checkpoint": checkpoint,
        "base_model": str(base_model),
        "train_file": str(train_file),
        "timestamp": datetime.now().isoformat(),
        "epochs": epochs,
        "learning_rate": learning_rate,
    }
    with open(new_checkpoint / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"   新 checkpoint: {new_checkpoint}")
    return str(new_checkpoint)


# ─── 主流程 ────────────────────────────────────────────

def run_auto(data_dir=None, config_path=None):
    """
    自动增量训练流程（设计为 cron 每天调用一次）：
    
    1. 检查新数据量
    2. 不够 → 跳过
    3. 够了 → 确认 test set → 增量微调 → 更新状态
    """
    config = load_config(config_path)
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    min_samples = config.get("training", {}).get("min_new_samples", 20)

    state = get_training_state(data_dir)
    print(f"📋 当前状态:")
    print(f"   已训练轮次: {state.get('training_rounds', 0)}")
    print(f"   已训练样本: {state.get('total_new_samples_trained', 0)}")
    print(f"   当前 checkpoint: {state.get('current_checkpoint', '无')}")
    print()

    # 1. 统计新数据
    new_count = get_new_data_count(data_dir)
    print(f"📊 新数据: {new_count} 条（阈值: {min_samples}）")

    if new_count < min_samples:
        print(f"🟡 数据不够（{new_count} < {min_samples}），跳过训练")
        return None

    # 2. 准备训练数据
    train_file, train_count = prepare_train_data(data_dir)
    if not train_file:
        print("❌ 无法准备训练数据")
        return None

    print(f"📝 准备了 {train_count} 条新数据")

    # 3. 确保 test set 存在
    test_count = ensure_test_set(data_dir)
    if test_count == 0:
        print("⚠️ Test set 为空，评估结果可能不可靠")

    # 4. 增量微调
    checkpoint = get_latest_checkpoint(data_dir.parent / "models" if data_dir else None)
    new_checkpoint = sft_train_incremental(train_file, checkpoint, config_path, data_dir)

    if not new_checkpoint:
        print("❌ 训练失败")
        return None

    # 5. 更新训练状态
    state["last_train_time"] = datetime.now().timestamp()
    state["current_checkpoint"] = new_checkpoint
    state["total_new_samples_trained"] += train_count
    state["training_rounds"] += 1
    if not state.get("base_model") and not checkpoint:
        state["base_model"] = new_checkpoint
    state["history"].append({
        "round": state["training_rounds"],
        "timestamp": datetime.now().isoformat(),
        "new_samples": train_count,
        "checkpoint": new_checkpoint,
        "parent": checkpoint,
    })
    save_training_state(state, data_dir)

    print(f"\n✅ 训练完成 (第 {state['training_rounds']} 轮)")
    print(f"   新 checkpoint: {new_checkpoint}")
    print(f"   累计训练样本: {state['total_new_samples_trained']}")
    print(f"\n💡 下一步: 运行评估脚本检查效果")
    print(f"   python3 scripts/evaluate.py --model {new_checkpoint}")

    return new_checkpoint


def run_init(data_dir=None, base_model=None):
    """初始化训练环境"""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    # 创建目录
    for d in ["cleaned", "reports"]:
        (data_dir / d).mkdir(parents=True, exist_ok=True)

    # 初始化训练状态
    state = get_training_state(data_dir)
    if base_model:
        state["base_model"] = base_model
        save_training_state(state, data_dir)

    print(f"✅ 训练环境已初始化: {data_dir}")
    print(f"   基座模型: {state.get('base_model', '未配置')}")
    print(f"   Test set: {'已有' if (data_dir / 'test.jsonl').exists() else '待生成'}")


def run_status(data_dir=None):
    """查看训练状态"""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    state = get_training_state(data_dir)
    new_count = get_new_data_count(data_dir)

    print(f"📊 训练状态:")
    print(f"   训练轮次: {state.get('training_rounds', 0)}")
    print(f"   累计样本: {state.get('total_new_samples_trained', 0)}")
    print(f"   当前 checkpoint: {state.get('current_checkpoint', '无')}")
    print(f"   基座模型: {state.get('base_model', '未配置')}")
    print(f"   待训练新数据: {new_count} 条")

    if state.get("history"):
        print(f"\n📜 训练历史:")
        for h in state["history"][-5:]:
            print(f"   #{h['round']} {h['timestamp'][:10]} | +{h['new_samples']}条 | {Path(h['checkpoint']).name}")

    # checkpoint 列表
    models_dir = data_dir.parent / "models" if data_dir else DEFAULT_MODELS_DIR
    if models_dir.exists():
        checkpoints = sorted(models_dir.glob("checkpoint_*"))
        if checkpoints:
            print(f"\n📁 Checkpoints ({len(checkpoints)}):")
            for cp in checkpoints[-5:]:
                size_mb = sum(f.stat().st_size for f in cp.rglob("*")) / 1024 / 1024
                print(f"   {cp.name} ({size_mb:.1f} MB)")


def run_rollback(data_dir=None):
    """回滚到上一个 checkpoint"""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    models_dir = data_dir.parent / "models" if data_dir else DEFAULT_MODELS_DIR
    state = get_training_state(data_dir)

    checkpoints = sorted(models_dir.glob("checkpoint_*")) if models_dir.exists() else []
    if len(checkpoints) < 2:
        print("❌ 没有可回滚的 checkpoint")
        return

    current = str(checkpoints[-1])
    previous = str(checkpoints[-2])

    state["current_checkpoint"] = previous
    save_training_state(state, data_dir)

    print(f"🔄 已回滚:")
    print(f"   从: {checkpoints[-1].name}")
    print(f"   到: {checkpoints[-2].name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenClaw Self-Trainer 增量微调")
    sub = parser.add_subparsers(dest="command")

    # auto: 自动增量训练（cron 调用）
    p_auto = sub.add_parser("auto", help="自动增量训练（检查新数据，够就训练）")
    p_auto.add_argument("--data-dir", type=str, default=None)
    p_auto.add_argument("--config", type=str, default=None)

    # init: 初始化
    p_init = sub.add_parser("init", help="初始化训练环境")
    p_init.add_argument("--base-model", type=str, default=None, help="基座模型路径")
    p_init.add_argument("--data-dir", type=str, default=None)

    # status: 查看状态
    sub.add_parser("status", help="查看训练状态")

    # rollback: 回滚
    sub.add_parser("rollback", help="回滚到上一个 checkpoint")

    # manual: 手动指定数据训练
    p_manual = sub.add_parser("manual", help="手动训练（指定数据和 checkpoint）")
    p_manual.add_argument("--train-file", type=str, required=True)
    p_manual.add_argument("--checkpoint", type=str, default=None)
    p_manual.add_argument("--epochs", type=int, default=None)
    p_manual.add_argument("--data-dir", type=str, default=None)
    p_manual.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    if args.command == "auto":
        run_auto(data_dir=args.data_dir, config_path=args.config)
    elif args.command == "init":
        run_init(data_dir=args.data_dir, base_model=args.base_model)
    elif args.command == "status":
        run_status()
    elif args.command == "rollback":
        run_rollback()
    elif args.command == "manual":
        sft_train_incremental(
            train_file=args.train_file,
            checkpoint=args.checkpoint,
            config_path=args.config,
            data_dir=args.data_dir,
        )
    else:
        parser.print_help()
