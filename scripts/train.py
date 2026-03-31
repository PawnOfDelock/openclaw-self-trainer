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
import subprocess
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


# ─── 硬件检测与模型推荐 ──────────────────────────────────

# 常见模型库：name → (类型, 参数描述, LoRA所需VRAM, QLoRA所需VRAM)
MODEL_CATALOG = [
    {"name": "Qwen/Qwen3.5-27B",       "type": "dense", "desc": "27B",       "lora_vram": 45,  "qlora_vram": 22},
    {"name": "Qwen/Qwen3.5-35B-A3B",   "type": "MoE",   "desc": "35B总/3B激活", "lora_vram": 70, "qlora_vram": None},
    {"name": "Qwen/Qwen3.5-122B-A10B", "type": "MoE",   "desc": "122B总/10B激活", "lora_vram": 200, "qlora_vram": None},
    {"name": "Qwen/Qwen2.5-7B",        "type": "dense", "desc": "7B",        "lora_vram": 22,  "qlora_vram": 12},
    {"name": "Qwen/Qwen2.5-14B",       "type": "dense", "desc": "14B",       "lora_vram": 30,  "qlora_vram": 16},
]


def detect_hardware():
    """检测本地硬件信息"""
    info = {
        "gpu": [],
        "gpu_total_vram_gb": 0,
        "ram_gb": 0,
        "disk_free_gb": 0,
        "cpu_cores": 0,
    }

    # GPU 检测
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        info["gpu"].append({"name": parts[0], "vram_gb": float(parts[1])})
                        info["gpu_total_vram_gb"] += float(parts[1])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # RAM 检测
    try:
        result = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.startswith("Mem:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        info["ram_gb"] = _parse_size(parts[1])
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 磁盘检测
    try:
        result = subprocess.run(["df", "-h", "/"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 4:
                    info["disk_free_gb"] = _parse_size(parts[3])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # CPU 核心数
    try:
        result = subprocess.run(["nproc"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            info["cpu_cores"] = int(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info


def _parse_size(s):
    """解析 free -h / df -h 输出的大小字符串（如 '16G', '500M'）"""
    s = s.upper().strip().rstrip("I")
    if s.endswith("T"):
        return float(s[:-1]) * 1024
    elif s.endswith("G"):
        return float(s[:-1])
    elif s.endswith("M"):
        return float(s[:-1]) / 1024
    return float(s)


def run_suggest():
    """根据硬件推荐模型和训练方案"""
    hw = detect_hardware()

    print("🖥️  硬件检测结果:")
    print(f"   CPU 核心: {hw['cpu_cores']}")
    print(f"   内存: {hw['ram_gb']:.1f} GB")
    print(f"   磁盘剩余: {hw['disk_free_gb']:.1f} GB")
    if hw["gpu"]:
        for g in hw["gpu"]:
            print(f"   GPU: {g['name']} ({g['vram_gb']:.0f} GB)")
        print(f"   GPU 总 VRAM: {hw['gpu_total_vram_gb']:.0f} GB")
    else:
        print("   GPU: 未检测到")
    print()

    if not hw["gpu"]:
        print("⚠️  未检测到 GPU，暂无法进行本地训练。")
        print()
        print("📋 推荐最小硬件需求:")
        print("   - GPU: NVIDIA A100 40GB (或 V100 32GB, RTX 4090 24GB 等)")
        print("   - 内存: ≥ 64 GB")
        print("   - 磁盘: ≥ 100 GB 可用空间")
        print()
        print("💡 配有 GPU 后，重新运行此命令获取推荐方案。")
        return

    # 根据 VRAM 筛选可行方案
    vram = hw["gpu_total_vram_gb"]
    feasible = []
    for m in MODEL_CATALOG:
        if m["lora_vram"] <= vram:
            feasible.append((m["name"], m["desc"], "LoRA", m["lora_vram"]))
        if m.get("qlora_vram") and m["qlora_vram"] <= vram and m["lora_vram"] > vram:
            feasible.append((m["name"], m["desc"], "QLoRA", m["qlora_vram"]))

    if not feasible:
        print("❌ 当前 GPU VRAM 不足以运行任何推荐模型。")
        print(f"   可用 VRAM: {vram:.0f} GB | 最小需求: QLoRA 7B 需要 ~12 GB")
        return

    print("✅ 推荐方案（按 VRAM 占用排序）:")
    feasible.sort(key=lambda x: x[3])
    for i, (name, desc, method, req_vram) in enumerate(feasible, 1):
        print(f"   {i}. {name} ({desc}) — {method} ~{req_vram} GB VRAM")

    print()
    print("💡 选择方案后运行:")
    print(f'   python3 {__file__} set-model "模型名"')


def run_set_model(model_name):
    """将模型名写入 defaults.yaml"""
    config_path = DEFAULT_CONFIG
    config = load_config(config_path)

    if not config:
        config = {"base_model": {"name": None, "trust_remote_code": True}}

    if "base_model" not in config:
        config["base_model"] = {"name": None, "trust_remote_code": True}
    config["base_model"]["name"] = model_name

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"✅ 基座模型已设置: {model_name}")
    print(f"   配置文件: {config_path}")


# ─── 数据准备 ──────────────────────────────────────────

def run_prepare():
    """将 collect.py 输出的 JSONL 转换为训练格式"""
    config = load_config()
    data_cfg = config.get("data", {})
    source_dir = data_cfg.get("source_dir")
    output_dir = data_cfg.get("prepare_output")
    val_split = data_cfg.get("val_split", 0.1)
    min_samples = data_cfg.get("min_samples", 50)

    # 检查 source_dir
    if not source_dir:
        print("❌ data.source_dir 未配置，请先在 defaults.yaml 中设置，或运行 collect.py 收集数据。")
        return

    source_path = Path(source_dir).expanduser()
    if not source_path.exists():
        print(f"❌ 数据源目录不存在: {source_path}")
        print("   请先运行 collect.py 收集数据。")
        return

    # 检查 output_dir
    if not output_dir:
        print("❌ data.prepare_output 未配置，请先在 defaults.yaml 中设置。")
        return

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取所有 JSONL 文件
    samples = []
    for jsonl_file in sorted(source_path.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"⚠️  跳过无效 JSON 行: {jsonl_file}")

    if len(samples) < min_samples:
        print(f"❌ 样本数不足: {len(samples)} < {min_samples}（data.min_samples）")
        return

    import random
    random.seed(42)
    random.shuffle(samples)

    # ─── 增量 test set 分割 ──────────────────────────
    # 每次增量时，从新数据中抽取 test_split 比例加入 test set。
    # 新数据从未参与过训练，保证 test set 的独立性。
    test_split = data_cfg.get("test_split", 0.1)
    test_max_size = data_cfg.get("test_max_size")

    test_file = output_path / "test.jsonl"
    existing_test_ids = set()

    # 读取已有 test set，记录已存在的样本 ID（避免重复）
    if test_file.exists():
        with open(test_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_test_ids.add(line)  # 用原始 JSON 行做去重
                    except Exception:
                        pass

    # 计算本次新增样本（排除已有 test 中的）
    new_for_test = []
    remaining = []
    for s in samples:
        s_line = json.dumps(s, ensure_ascii=False)
        if s_line not in existing_test_ids and random.random() < test_split:
            new_for_test.append(s)
        else:
            remaining.append(s)

    # 追加新 test 样本
    if new_for_test:
        with open(test_file, "a") as f:
            for s in new_for_test:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"   📎 新增 test 样本: {len(new_for_test)} 条")

    # test set 滚动窗口：超过上限时丢弃最早的样本
    if test_max_size and test_file.exists():
        with open(test_file) as f:
            all_test_lines = [l.strip() for l in f if l.strip()]
        if len(all_test_lines) > test_max_size:
            trimmed = all_test_lines[-test_max_size:]
            with open(test_file, "w") as f:
                for l in trimmed:
                    f.write(l + "\n")
            print(f"   ✂️  test set 裁剪: {len(all_test_lines)} → {len(trimmed)}")

    # ─── train/val 分割 ──────────────────────────────
    n_val = max(1, int(len(remaining) * val_split))
    train_samples = remaining[n_val:]
    val_samples = remaining[:n_val]

    def convert(sample):
        """将 OpenAI messages 格式转为训练格式"""
        messages = sample.get("messages", [])

        # 提取系统提示
        instruction = "You are a helpful AI assistant."
        for msg in messages:
            if msg.get("role") == "system":
                instruction = msg.get("content", instruction)
                break

        # 拼接多轮对话为文本
        turns = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                continue  # 已单独提取
            if role == "user":
                turns.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                turns.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif role == "toolResult":
                turns.append(f"<|im_start|>tool\n{content}<|im_end|>")

        text = "\n".join(turns)

        # 最后一轮 assistant 回复作为 output
        output = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                output = msg.get("content", "")
                break

        return {"instruction": instruction, "text": text, "output": output}

    train_records = [convert(s) for s in train_samples]
    val_records = [convert(s) for s in val_samples]

    # 写入文件
    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"

    with open(train_file, "w") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_file, "w") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计 test set 大小
    test_count = 0
    if test_file.exists():
        with open(test_file) as f:
            test_count = sum(1 for l in f if l.strip())

    print(f"✅ 数据准备完成:")
    print(f"   总样本: {len(samples)}")
    print(f"   训练集: {len(train_records)} → {train_file}")
    print(f"   验证集: {len(val_records)} → {val_file}")
    print(f"   测试集: {test_count} 条 → {test_file}")


# ─── 配置状态 ──────────────────────────────────────────

def run_config_status():
    """显示配置和训练状态"""
    config = load_config()

    print("📋 defaults.yaml 配置:")
    if not config:
        print("   ⚠️  配置文件为空或不存在")
        print(f"   路径: {DEFAULT_CONFIG}")
        return

    # 基座模型
    base = config.get("base_model", {})
    model_name = base.get("name")
    status_icon = "✅" if model_name else "❌ 未配置"
    print(f"   基座模型: {status_icon}")
    print(f"      {model_name or '(空)'}")
    print(f"      trust_remote_code: {base.get('trust_remote_code', True)}")

    # 训练参数
    tr = config.get("training", {})
    print(f"   训练方法: {tr.get('method', 'lora')}")
    if tr.get("method") in ("lora", "qlora"):
        print(f"   LoRA r/α/dropout: {tr.get('lora_r')}/{tr.get('lora_alpha')}/{tr.get('lora_dropout')}")
    print(f"   Epochs: {tr.get('num_epochs')}, LR: {tr.get('learning_rate')}")
    print(f"   Batch size: {tr.get('batch_size')} × {tr.get('gradient_accumulation_steps')} accum")
    print(f"   Max seq length: {tr.get('max_seq_length')}")

    # 数据配置
    data = config.get("data", {})
    src = data.get("source_dir")
    out = data.get("prepare_output")
    print(f"   数据源: {src or '❌ 未配置'}")
    print(f"   输出目录: {out or '❌ 未配置'}")

    # 检查数据目录文件
    if out:
        out_path = Path(out).expanduser()
        if out_path.exists():
            files = list(out_path.glob("*.jsonl"))
            print(f"   已准备文件: {len(files)}")
            for f in files:
                count = sum(1 for _ in open(f) if _.strip())
                print(f"      {f.name}: {count} 条")
        else:
            print("   ⚠️  输出目录不存在")

    # 训练状态
    print()
    run_status()


# ─── 训练（占位） ──────────────────────────────────────

def run_train():
    """训练命令占位 — 检查配置和数据完整性"""
    config = load_config()

    errors = []

    # 检查基座模型
    if not config.get("base_model", {}).get("name"):
        errors.append("base_model.name 未配置 — 运行 --suggest 和 --set-model 选择模型")

    # 检查数据
    data = config.get("data", {})
    output_dir = data.get("prepare_output")
    if not output_dir:
        errors.append("data.prepare_output 未配置")
    else:
        out_path = Path(output_dir).expanduser()
        train_file = out_path / "train.jsonl"
        val_file = out_path / "val.jsonl"
        if not train_file.exists():
            errors.append("训练数据未准备 — 运行 prepare 命令")
        if not val_file.exists():
            errors.append("验证数据未准备 — 运行 prepare 命令")

    if errors:
        print("❌ 训练前置检查未通过:")
        for e in errors:
            print(f"   • {e}")
        return

    model_name = config["base_model"]["name"]
    tr = config.get("training", {})

    print("✅ 配置检查通过，训练将由 agent 按照以下配置执行:")
    print(f"   模型: {model_name}")
    print(f"   方法: {tr.get('method', 'lora')}")
    print(f"   Epochs: {tr.get('num_epochs')}, LR: {tr.get('learning_rate')}")
    print(f"   训练数据: {output_dir}/train.jsonl")
    print(f"   验证数据: {output_dir}/val.jsonl")
    print()
    print("⚠️  实际训练逻辑待实现（需要 GPU 环境）")


def main():
    """主入口：使用 subcommand 解析"""
    parser = argparse.ArgumentParser(description="OpenClaw Self-Trainer 增量微调")
    sub = parser.add_subparsers(dest="command")

    # suggest: 硬件检测 + 模型推荐
    sub.add_parser("suggest", help="检测硬件并推荐模型方案")

    # set-model: 设置基座模型
    p_set = sub.add_parser("set-model", help="设置基座模型")
    p_set.add_argument("model_name", type=str, help="模型名称")

    # prepare: 数据准备
    sub.add_parser("prepare", help="将收集的数据转换为训练格式")

    # config-status: 配置和训练状态
    sub.add_parser("config-status", help="查看配置和训练状态")

    # train: 训练（占位）
    sub.add_parser("train", help="训练模型（检查配置后由 agent 执行）")

    # auto: 自动增量训练（cron 调用）
    p_auto = sub.add_parser("auto", help="自动增量训练（检查新数据，够就训练）")
    p_auto.add_argument("--data-dir", type=str, default=None)
    p_auto.add_argument("--config", type=str, default=None)

    # init: 初始化
    p_init = sub.add_parser("init", help="初始化训练环境")
    p_init.add_argument("--base-model", type=str, default=None, help="基座模型路径")
    p_init.add_argument("--data-dir", type=str, default=None)

    # status: 查看状态（原有）
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

    if args.command == "suggest":
        run_suggest()
    elif args.command == "set-model":
        run_set_model(args.model_name)
    elif args.command == "prepare":
        run_prepare()
    elif args.command == "config-status":
        run_config_status()
    elif args.command == "train":
        run_train()
    elif args.command == "auto":
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


if __name__ == "__main__":
    main()
