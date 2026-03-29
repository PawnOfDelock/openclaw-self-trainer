#!/usr/bin/env python3
"""
OpenClaw Self-Trainer - 数据收集脚本

从 OpenClaw session 日志中提取对话，清洗元数据噪音，分类并追加到训练集。
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 默认配置路径
DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "defaults.yaml"
DEFAULT_DATA_DIR = Path.home() / ".cache" / "self-trainer" / "data"


def load_config(config_path=None):
    """加载配置文件"""
    import yaml
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def init_data_dir(data_dir=None):
    """初始化数据目录结构"""
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    dirs = ["raw", "cleaned", "reports"]
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)
    print(f"✅ 数据目录已初始化: {base}")
    return base


def collect_sessions(data_dir=None):
    """从 OpenClaw 收集当天的 session 日志"""
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    config = load_config()
    cleaning = config.get("cleaning", {})

    # TODO: 实现 OpenClaw session 日志的读取
    # 这里需要集成 OpenClaw 的 session 导出接口
    sessions = []

    print(f"📊 收集到 {len(sessions)} 条对话")

    # 清洗
    cleaned = []
    for s in sessions:
        input_text = s.get("input", "")
        output_text = s.get("output", "")

        # 去掉元数据前缀
        if cleaning.get("remove_metadata_prefix", True):
            input_text = clean_metadata(input_text)
            output_text = clean_metadata(output_text)

        # 长度过滤
        if len(input_text) < cleaning.get("min_input_length", 5):
            continue
        if len(output_text) < cleaning.get("min_output_length", 10):
            continue
        if len(output_text) > cleaning.get("max_output_length", 4096):
            continue

        cleaned.append({
            "input": input_text,
            "output": output_text,
            "category": classify(input_text),
            "timestamp": s.get("timestamp", ""),
        })

    # 保存
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = base / "cleaned" / f"{today}.jsonl"
    with open(out_path, "w") as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 清洗后保留 {len(cleaned)} 条，保存到 {out_path}")
    return cleaned


def clean_metadata(text):
    """清洗 OpenClaw 元数据前缀"""
    prefixes = [
        "Conversation info (untrusted metadata):\n",
        "Inbound Context (trusted metadata):\n",
        "System:",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def classify(text):
    """简单的关键词分类"""
    categories = {
        "技术开发": ["代码", "脚本", "编程", "debug", "git", "github", "部署", "训练", "模型", "deepseed", "openclaw", "api", "plugin"],
        "工具操作": ["搜索", "查", "帮我找", "发送", "生成", "下载", "上传", "天气"],
        "生活杂聊": ["REDACTED", "REDACTED", "REDACTED", "搞笑", "REDACTED", "电影", "音乐"],
        "知识讨论": ["什么是", "为什么", "怎么做", "原理", "区别", "对比"],
    }
    text_lower = text.lower()
    for category, keywords in categories.items():
        for kw in keywords:
            if kw in text_lower:
                return category
    return "其他"


def check_installation(data_dir=None):
    """检查安装状态"""
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    checks = []

    # 检查目录
    checks.append(("数据目录", base.exists()))
    for d in ["raw", "cleaned", "reports"]:
        checks.append((f"  {d}/", (base / d).exists()))

    # 检查依赖
    try:
        import deepspeed
        checks.append(("DeepSpeed", True))
    except ImportError:
        checks.append(("DeepSpeed", False))

    try:
        import torch
        checks.append((f"PyTorch (CUDA: {torch.cuda.is_available()})", True))
    except ImportError:
        checks.append(("PyTorch", False))

    all_passed = all(c[1] for c in checks)
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")

    if all_passed:
        print("✅ All checks passed")
    else:
        print("❌ Some checks failed")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenClaw Self-Trainer 数据收集")
    parser.add_argument("--init", action="store_true", help="初始化数据目录")
    parser.add_argument("--collect", action="store_true", help="收集并清洗数据")
    parser.add_argument("--check", action="store_true", help="检查安装状态")
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录路径")
    args = parser.parse_args()

    if args.init:
        init_data_dir(args.data_dir)
    elif args.collect:
        collect_sessions(args.data_dir)
    elif args.check:
        check_installation(args.data_dir)
    else:
        parser.print_help()
