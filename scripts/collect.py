#!/usr/bin/env python3
"""
OpenClaw Self-Trainer - 数据收集脚本

从 OpenClaw session 日志中提取对话，清洗元数据噪音，分类并追加到训练集。

Session 日志格式：
- 位置: ~/.openclaw/agents/main/sessions/*.jsonl
- 每行一个 JSON 对象，type 可以是: session, message, text, toolCall, custom 等
- message 中 role 可以是 user 或 assistant
- assistant 消息可能包含 text 和 toolCall
- user 消息包含 Slack 元数据前缀，需要清洗
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import yaml


# ─── 默认配置 ───────────────────────────────────────────
DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "defaults.yaml"
DEFAULT_SESSIONS_DIR = Path.home() / ".openclaw" / "agents" / "main" / "sessions"
DEFAULT_DATA_DIR = Path.home() / ".cache" / "self-trainer" / "data"


def load_config(config_path=None):
    import yaml
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


# ─── 数据清洗 ───────────────────────────────────────────

def clean_user_message(text):
    """
    清洗 OpenClaw user 消息中的元数据噪音。

    原始格式示例：
    ```
    System: [timestamp] Slack DM from user: 实际消息内容

    Conversation info (untrusted metadata):
    { ... }

    Sender (untrusted metadata):
    { ... }

    实际消息内容
    ```

    我们只保留最后一部分"实际消息内容"。
    """
    # 去掉开头的 "System: [timestamp] Slack DM from user: " 前缀
    # 格式示例: "System: [2026-03-09 12:47:01 GMT+8] Slack DM from user: 实际消息内容"
    text = re.sub(r'^System:\s*\[.*?\]\s*Slack\s+\w+\s+from\s+\S+:\s*', '', text)

    # 去掉 "Conversation info (untrusted metadata):" 及其后面的 json block
    text = re.sub(r'Conversation info \(untrusted metadata\):\s*\n```json\s*\n.*?```\s*\n', '', text, flags=re.DOTALL)

    # 去掉 "Sender (untrusted metadata):" 及其后面的 json block
    text = re.sub(r'Sender \(untrusted metadata\):\s*\n```json\s*\n.*?```\s*\n', '', text, flags=re.DOTALL)

    # 去掉 "Inbound Context (trusted metadata):" 及其后面的 json block
    text = re.sub(r'Inbound Context \(trusted metadata\):\s*\n```json\s*\n.*?```\s*\n', '', text, flags=re.DOTALL)

    return text.strip()


def clean_assistant_message(text):
    """清洗 assistant 消息，去掉系统标签"""
    # 去掉 NO_REPLY（单独的回复）
    if text.strip() == "NO_REPLY":
        return None

    # 去掉 HEARTBEAT_OK
    if text.strip() == "HEARTBEAT_OK":
        return None

    # 去掉常见的系统前缀
    for prefix in ["[[reply_to_current]] ", "[reply_to:", "[[reply_to:"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            # 如果是 [[reply_to_current]] 后面紧跟内容
            if text.startswith("]] "):
                text = text[3:]

    return text.strip() if text.strip() else None


def extract_tool_calls(message_content):
    """从 assistant 消息中提取工具调用信息"""
    tool_calls = []
    if not isinstance(message_content, list):
        return tool_calls

    for block in message_content:
        if isinstance(block, dict) and block.get("type") == "toolCall":
            tool_calls.append({
                "name": block.get("name", ""),
                "arguments": block.get("arguments", {}),
            })

    return tool_calls


# ─── 分类 ───────────────────────────────────────────────

# 分类关键词（中文 + 英文）
CATEGORY_KEYWORDS = {
    "技术开发": [
        # 中文
        "代码", "脚本", "编程", "debug", "部署", "训练", "模型", "微调", "sft",
        "gpu", "cuda", "显存", "loRA", "checkpoint", "权重", "epoch",
        "deepseed", "openclaw", "api", "plugin", "skill", "插件",
        "仓库", "repo", "commit", "pr", "issue", "branch", "merge",
        "服务器", "docker", "容器", "nginx", "ssl", "证书",
        "python", "javascript", "typescript", "node", "npm", "pip",
        "数据库", "sql", "redis", "缓存", "框架", "架构",
        # 英文
        "code", "script", "deploy", "train", "fine-tune", "model",
        "github", "git", "docker", "kubernetes", "k8s",
    ],
    "工具操作": [
        "搜索", "帮我找", "查一下", "查询", "发送", "帮我发",
        "生成", "下载", "上传", "转换", "翻译", "画一个",
        "天气", "日历", "提醒", "待办", "邮件", "文件",
        "截图", "浏览器", "打开网页",
        "search", "send", "generate", "download", "upload",
    ],
    "生活杂聊": [
        "电影", "音乐", "游戏", "美食", "旅游", "运动",
        "孩子", "家人",
        "开心", "无聊", "累了", "早安", "晚安", "周末",
    ],
    "知识讨论": [
        "什么是", "为什么", "怎么做", "原理", "区别", "对比",
        "如何看待", "你的观点", "分析一下", "解释一下",
        "思考", "理解", "概念", "定义", "历史", "趋势",
        "what is", "how to", "why", "explain", "analyze",
    ],
}


def classify(text):
    """基于关键词的文本分类"""
    text_lower = text.lower()
    scores = defaultdict(int)

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[category] += 1

    if not scores:
        return "其他"

    return max(scores, key=scores.get)


# ─── Session 解析 ──────────────────────────────────────

def parse_session(session_path):
    """
    解析单个 session 文件，提取对话对。

    返回: list of {
        "user_input": str,       # 清洗后的用户输入
        "assistant_text": str,   # assistant 的文本回复
        "tool_calls": list,      # assistant 的工具调用
        "category": str,         # 分类
        "timestamp": str,        # 时间戳
        "session_id": str,       # session ID
    }
    """
    messages = []
    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "message":
                    msg = obj.get("message", {})
                    messages.append(msg)
            except json.JSONDecodeError:
                continue

    # 提取 session ID（从文件名或第一条 session 记录）
    session_id = session_path.stem

    # 构建 user -> assistant 对话对
    conversations = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "user":
            # 提取用户文本
            user_text = ""
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        user_text = block.get("text", "")
                        break
            elif isinstance(content, str):
                user_text = content

            user_text = clean_user_message(user_text)
            if not user_text:
                i += 1
                continue

            # 找到下一个 assistant 回复
            j = i + 1
            assistant_texts = []
            assistant_tool_calls = []
            while j < len(messages) and messages[j].get("role") != "user":
                if messages[j].get("role") == "assistant":
                    content = messages[j].get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    cleaned = clean_assistant_message(block.get("text", ""))
                                    if cleaned:
                                        assistant_texts.append(cleaned)
                                elif block.get("type") == "toolCall":
                                    assistant_tool_calls.append({
                                        "name": block.get("name", ""),
                                        "args_keys": list(block.get("arguments", {}).keys()),
                                    })
                    elif isinstance(content, str):
                        cleaned = clean_assistant_message(content)
                        if cleaned:
                            assistant_texts.append(cleaned)
                j += 1

            if assistant_texts:
                combined_text = "\n".join(assistant_texts)
                conversations.append({
                    "user_input": user_text,
                    "assistant_text": combined_text,
                    "tool_calls": assistant_tool_calls,
                    "category": classify(user_text),
                    "timestamp": msg.get("timestamp", ""),
                    "session_id": session_id,
                })

            i = j
        else:
            i += 1

    return conversations


# ─── 主逻辑 ────────────────────────────────────────────

def init_data_dir(data_dir=None):
    """初始化数据目录结构"""
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    for d in ["raw", "cleaned", "reports"]:
        (base / d).mkdir(parents=True, exist_ok=True)
    print(f"✅ 数据目录已初始化: {base}")
    return base


def collect_sessions(sessions_dir=None, data_dir=None, since=None, dry_run=False):
    """从 OpenClaw session 日志收集对话"""
    sessions_dir = Path(sessions_dir) if sessions_dir else DEFAULT_SESSIONS_DIR
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    config = load_config()
    cleaning = config.get("cleaning", {})

    if not sessions_dir.exists():
        print(f"❌ Session 目录不存在: {sessions_dir}")
        return []

    # 收集所有 session 文件
    session_files = sorted(sessions_dir.glob("*.jsonl"))
    print(f"📂 找到 {len(session_files)} 个 session 文件")

    # 解析所有 session
    all_conversations = []
    for sf in session_files:
        try:
            convs = parse_session(sf)
            all_conversations.extend(convs)
        except Exception as e:
            print(f"  ⚠️ 解析失败 {sf.name}: {e}")

    print(f"📊 解析出 {len(all_conversations)} 条对话对")

    # 去重（基于 user_input 相似度）
    seen_inputs = set()
    unique_conversations = []
    for conv in all_conversations:
        # 简单去重：完全相同的 input
        input_key = conv["user_input"][:100]  # 取前100字符作为指纹
        if input_key not in seen_inputs:
            seen_inputs.add(input_key)
            unique_conversations.append(conv)

    dedup_count = len(all_conversations) - len(unique_conversations)
    if dedup_count > 0:
        print(f"🔄 去重后保留 {len(unique_conversations)} 条（去掉 {dedup_count} 条重复）")

    # 清洗：长度过滤
    cleaned = []
    for conv in unique_conversations:
        if len(conv["user_input"]) < cleaning.get("min_input_length", 5):
            continue
        if len(conv["assistant_text"]) < cleaning.get("min_output_length", 10):
            continue
        if len(conv["assistant_text"]) > cleaning.get("max_output_length", 4096):
            continue
        cleaned.append(conv)

    filter_count = len(unique_conversations) - len(cleaned)
    if filter_count > 0:
        print(f"🔍 长度过滤后保留 {len(cleaned)} 条（去掉 {filter_count} 条）")

    if dry_run:
        # 只打印统计，不写入文件
        print(f"\n📊 干运行结果：")
        print(f"   总对话对: {len(cleaned)}")
        category_counts = defaultdict(int)
        for c in cleaned:
            category_counts[c["category"]] += 1
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"   {cat}: {count} ({count/len(cleaned)*100:.1f}%)")
        return cleaned

    # 保存
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = data_dir / "cleaned" / f"{today}.jsonl"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "cleaned").mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for conv in cleaned:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"💾 保存到 {out_path}")

    # 打印分类统计
    category_counts = defaultdict(int)
    for c in cleaned:
        category_counts[c["category"]] += 1
    print(f"\n📈 分类统计:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"   {cat}: {count} ({count/len(cleaned)*100:.1f}%)")

    return cleaned


def check_installation(data_dir=None):
    """检查安装状态"""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    sessions_dir = DEFAULT_SESSIONS_DIR
    checks = []

    # 检查目录
    checks.append(("Session 目录", sessions_dir.exists()))
    if sessions_dir.exists():
        session_count = len(list(sessions_dir.glob("*.jsonl")))
        checks.append((f"  Session 文件数", True, f"{session_count}"))

    checks.append(("数据目录", data_dir.exists()))

    # 检查 Python 依赖
    deps = [("PyYAML", "yaml"), ("scikit-learn", "sklearn"), ("jieba", "jieba")]
    for name, module in deps:
        try:
            __import__(module)
            checks.append((name, True))
        except ImportError:
            checks.append((name, False))

    # 检查 GPU（可选）
    try:
        import torch
        checks.append((f"PyTorch CUDA", torch.cuda.is_available()))
    except ImportError:
        checks.append(("PyTorch", False))

    try:
        import deepspeed
        checks.append(("DeepSpeed", True))
    except ImportError:
        checks.append(("DeepSpeed (可选)", False))

    all_passed = all(c[1] for c in checks)
    for item in checks:
        name = item[0]
        ok = item[1]
        extra = f" ({item[2]})" if len(item) > 2 else ""
        print(f"  {'✅' if ok else '❌'} {name}{extra}")

    if all_passed:
        print("\n✅ All checks passed")
    else:
        print("\n⚠️ 部分检查未通过，核心功能仍可使用（数据收集不依赖 GPU）")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenClaw Self-Trainer 数据收集")
    parser.add_argument("--init", action="store_true", help="初始化数据目录")
    parser.add_argument("--collect", action="store_true", help="收集并清洗数据")
    parser.add_argument("--check", action="store_true", help="检查安装状态")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入文件")
    parser.add_argument("--sessions-dir", type=str, default=None, help="OpenClaw sessions 目录路径")
    parser.add_argument("--data-dir", type=str, default=None, help="数据输出目录路径")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    if args.init:
        init_data_dir(args.data_dir)
    elif args.collect:
        collect_sessions(
            sessions_dir=args.sessions_dir,
            data_dir=args.data_dir,
            dry_run=args.dry_run,
        )
    elif args.check:
        check_installation(args.data_dir)
    else:
        parser.print_help()
