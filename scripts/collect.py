#!/usr/bin/env python3
"""
OpenClaw Self-Trainer - 数据收集脚本

从 OpenClaw session 日志中提取完整的多轮交互链，输出 OpenAI messages 格式。

Session 日志格式：
- 位置: ~/.openclaw/agents/main/sessions/*.jsonl
- 每行一个 JSON 对象，type 可以是: session, message, custom 等
- message.role: user / assistant / toolResult
- assistant 消息可包含 text、toolCall、thinking block
- toolResult 消息包含工具执行的返回结果
- user 消息包含 Slack 元数据前缀，保留原始格式用于训练

输出格式 (OpenAI messages):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},    // 可含 think + toolCall + text
    {"role": "tool", "content": "...", "tool_call_id": "..."},
    {"role": "assistant", "content": "..."},    // 根据工具结果继续
    ...
  ],
  "metadata": {"category": "...", "session_id": "...", "timestamp": "..."}
}
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml


# ─── 默认配置 ───────────────────────────────────────────
DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "defaults.yaml"
DEFAULT_SESSIONS_DIR = Path.home() / ".openclaw" / "agents" / "main" / "sessions"
DEFAULT_DATA_DIR = Path.home() / ".cache" / "self-trainer" / "data"


def load_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


# ─── 数据清洗 ───────────────────────────────────────────

def clean_user_message(text):
    """清洗用户消息，仅用于分类统计（不影响训练数据中的原始格式）"""
    text = re.sub(r'^System:\s*\[.*?\]\s*Slack\s+\w+\s+from\s+\S+:\s*', '', text)
    text = re.sub(r'Conversation info \(untrusted metadata\):\s*\n```json\s*\n.*?```\s*\n', '', text, flags=re.DOTALL)
    text = re.sub(r'Sender \(untrusted metadata\):\s*\n```json\s*\n.*?```\s*\n', '', text, flags=re.DOTALL)
    text = re.sub(r'Inbound Context \(trusted metadata\):\s*\n```json\s*\n.*?```\s*\n', '', text, flags=re.DOTALL)
    return text.strip()


def should_skip_message(text):
    """检查是否应该跳过的系统消息"""
    stripped = text.strip()
    return stripped in ("NO_REPLY", "HEARTBEAT_OK", "")


def clean_assistant_text(text):
    """清洗 assistant 文本，去掉 reply_to 标签"""
    if should_skip_message(text):
        return None
    for prefix in ["[[reply_to_current]] ", "[[reply_to:"]:
        if text.startswith(prefix):
            idx = text.find("]] ")
            if idx != -1:
                text = text[idx + 3:]
            else:
                text = text[len(prefix):]
    return text.strip() if text.strip() else None


# ─── 分类 ───────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "技术开发": [
        "代码", "脚本", "编程", "debug", "部署", "训练", "模型", "微调", "sft",
        "gpu", "cuda", "显存", "loRA", "checkpoint", "权重", "epoch",
        "openclaw", "api", "plugin", "skill", "插件",
        "仓库", "repo", "commit", "pr", "issue", "branch", "merge",
        "服务器", "docker", "容器", "nginx", "ssl", "证书",
        "python", "javascript", "typescript", "node", "npm", "pip",
        "数据库", "sql", "redis", "缓存", "框架", "架构",
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
    return max(scores, key=scores.get) if scores else "其他"


# ─── 完整交互链解析 ────────────────────────────────────

def extract_text_from_content(content):
    """从消息 content 中提取纯文本"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts) if texts else ""
    return ""


def format_assistant_content(content_blocks):
    """
    将 assistant 消息的 content blocks 格式化为单个字符串。
    
    保留完整的思考链 + 工具调用 + 文本回复，用于 SFT 训练。
    
    格式：
    <think reasoning_content>
    思考内容...
    </think_text>
    
    {"type":"tool_call","name":"exec","arguments":{"command":"..."}}
    """
    if not isinstance(content_blocks, list):
        # 纯文本
        cleaned = clean_assistant_text(str(content_blocks))
        return cleaned or ""

    parts = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue

        btype = block.get("type", "")

        if btype == "text":
            text = clean_assistant_text(block.get("text", ""))
            if text:
                parts.append(text)

        elif btype == "toolCall":
            tool_call = {
                "type": "tool_call",
                "name": block.get("name", ""),
                "arguments": block.get("arguments", {}),
            }
            parts.append(json.dumps(tool_call, ensure_ascii=False))

    return "\n\n".join(parts)


def parse_session_to_conversations(session_path):
    """
    解析 session 文件，提取完整的多轮交互链（OpenAI messages 格式）。
    
    每个 conversation 是一轮 user -> [assistant + toolResult 交替] 的完整链。
    
    返回: list of {
        "messages": [...],  # OpenAI messages 格式
        "metadata": {
            "category": str,
            "session_id": str,
            "timestamp": str,
            "turns": int,         # user 轮数
            "tool_calls": int,    # 总工具调用数
        }
    }
    """
    raw_messages = []
    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "message":
                    raw_messages.append({
                        "role": obj.get("message", {}).get("role", ""),
                        "content": obj.get("message", {}).get("content", []),
                        "timestamp": obj.get("timestamp", ""),
                    })
            except json.JSONDecodeError:
                continue

    if not raw_messages:
        return []

    session_id = session_path.stem
    conversations = []

    # 按 user 消息分割成多个 conversation
    # 每个 conversation 从一个 user 消息开始，到下一个 user 消息之前结束
    i = 0
    while i < len(raw_messages):
        msg = raw_messages[i]
        if msg["role"] != "user":
            i += 1
            continue

        # 新 conversation 开始
        messages = []
        first_timestamp = msg["timestamp"]
        tool_call_count = 0

        # 添加 user 消息（保留原始格式，含元数据）
        user_text = extract_text_from_content(msg["content"])
        if not user_text.strip():
            i += 1
            continue
        messages.append({"role": "user", "content": user_text})

        # 收集后续的 assistant / toolResult 消息
        j = i + 1
        while j < len(raw_messages) and raw_messages[j]["role"] != "user":
            rmsg = raw_messages[j]
            role = rmsg["role"]

            if role == "assistant":
                # 格式化 assistant 完整内容（思考 + 工具调用 + 文本）
                content_blocks = rmsg["content"]
                if isinstance(content_blocks, list):
                    # 检查是否有文本（跳过纯 NO_REPLY）
                    has_text = any(
                        isinstance(b, dict) and b.get("type") == "text" and clean_assistant_text(b.get("text", ""))
                        for b in content_blocks if isinstance(b, dict)
                    )
                    has_tools = any(
                        isinstance(b, dict) and b.get("type") == "toolCall"
                        for b in content_blocks if isinstance(b, dict)
                    )
                    if not has_text and not has_tools:
                        j += 1
                        continue

                formatted = format_assistant_content(content_blocks)
                if formatted.strip():
                    messages.append({"role": "assistant", "content": formatted})
                    # 统计工具调用
                    if isinstance(content_blocks, list):
                        for b in content_blocks:
                            if isinstance(b, dict) and b.get("type") == "toolCall":
                                tool_call_count += 1

            elif role == "toolResult":
                # 工具返回结果
                result_text = extract_text_from_content(rmsg["content"])
                if result_text.strip():
                    messages.append({"role": "tool", "content": result_text})

            j += 1

        # 只保留有 assistant 回复的 conversation
        has_assistant = any(m["role"] == "assistant" for m in messages)
        if has_assistant and len(messages) >= 2:
            # 用第一条 user 消息的清洗版做分类
            clean_input = clean_user_message(user_text)
            messages_out = [{"role": "system", "content": "你是一个 AI 助手。"}] + messages

            conversations.append({
                "messages": messages_out,
                "metadata": {
                    "category": classify(clean_input),
                    "session_id": session_id,
                    "timestamp": first_timestamp,
                    "turns": sum(1 for m in messages if m["role"] == "user"),
                    "tool_calls": tool_call_count,
                },
            })

        i = j

    return conversations


# ─── 主逻辑 ────────────────────────────────────────────

def init_data_dir(data_dir=None):
    """初始化数据目录结构"""
    base = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    for d in ["raw", "cleaned", "reports"]:
        (base / d).mkdir(parents=True, exist_ok=True)
    print(f"✅ 数据目录已初始化: {base}")
    return base


def collect_sessions(sessions_dir=None, data_dir=None, dry_run=False, analyze_only=False):
    """
    从 OpenClaw session 日志收集完整交互链。
    
    输出 OpenAI messages 格式，包含：
    - system prompt
    - user 消息（保留原始元数据）
    - assistant 完整回复（思考链 + 工具调用 + 文本）
    - tool 返回结果
    """
    sessions_dir = Path(sessions_dir) if sessions_dir else DEFAULT_SESSIONS_DIR
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    config = load_config()

    if not sessions_dir.exists():
        print(f"❌ Session 目录不存在: {sessions_dir}")
        return []

    session_files = sorted(sessions_dir.glob("*.jsonl"))
    print(f"📂 找到 {len(session_files)} 个 session 文件")

    # 解析所有 session
    all_convs = []
    for sf in session_files:
        try:
            convs = parse_session_to_conversations(sf)
            all_convs.extend(convs)
        except Exception as e:
            print(f"  ⚠️ 解析失败 {sf.name}: {e}")

    print(f"📊 解析出 {len(all_convs)} 条交互链")

    # 去重（基于第一条 user 消息的清洗版）
    seen = set()
    unique = []
    for conv in all_convs:
        msgs = conv["messages"]
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        if user_msgs:
            key = clean_user_message(user_msgs[0])[:100]
            if key not in seen:
                seen.add(key)
                unique.append(conv)

    dedup_count = len(all_convs) - len(unique)
    if dedup_count > 0:
        print(f"🔄 去重后保留 {len(unique)} 条（去掉 {dedup_count} 条重复）")

    # 过滤：太短的交互链
    cleaned = []
    for conv in unique:
        msgs = conv["messages"]
        # 计算总 token 估算（粗略：1 token ≈ 2 字符）
        total_chars = sum(len(m["content"]) for m in msgs)
        if total_chars < 20:  # 太短
            continue
        cleaned.append(conv)

    filter_count = len(unique) - len(cleaned)
    if filter_count > 0:
        print(f"🔍 过滤后保留 {len(cleaned)} 条（去掉 {filter_count} 条过短）")

    # 统计
    category_counts = defaultdict(int)
    tool_call_counts = defaultdict(int)
    has_tool = 0
    multi_turn = 0
    for conv in cleaned:
        cat = conv["metadata"]["category"]
        category_counts[cat] += 1
        tc = conv["metadata"]["tool_calls"]
        tool_call_counts[cat] += tc
        if tc > 0:
            has_tool += 1
        if conv["metadata"]["turns"] > 1:
            multi_turn += 1

    if dry_run or analyze_only:
        mode = "分析" if analyze_only else "干运行"
        print(f"\n📊 {mode}结果：")
        print(f"   总交互链: {len(cleaned)}")
        print(f"   含工具调用: {has_tool} ({has_tool/len(cleaned)*100:.1f}%)" if cleaned else "")
        print(f"   多轮对话: {multi_turn} ({multi_turn/len(cleaned)*100:.1f}%)" if cleaned else "")
        print(f"\n   分类分布:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            bar = "█" * int(count / len(cleaned) * 20) if cleaned else ""
            print(f"   {cat:8s} {count:4d} ({count/len(cleaned)*100:5.1f}%) {bar}")
            if tool_call_counts[cat] > 0:
                print(f"           └─ 工具调用: {tool_call_counts[cat]}")
        return cleaned

    # 保存训练数据
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = data_dir / "cleaned" / f"{today}.jsonl"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "cleaned").mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for conv in cleaned:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"💾 保存到 {out_path}")
    print(f"\n📈 分类分布:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / len(cleaned) * 20) if cleaned else ""
        print(f"   {cat:8s} {count:4d} ({count/len(cleaned)*100:5.1f}%) {bar}")

    print(f"\n📊 工具调用统计:")
    for cat, tc in sorted(tool_call_counts.items(), key=lambda x: -x[1]):
        print(f"   {cat:8s} {tc} 次工具调用")

    return cleaned


def check_installation(data_dir=None):
    """检查安装状态"""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    sessions_dir = DEFAULT_SESSIONS_DIR
    checks = []

    checks.append(("Session 目录", sessions_dir.exists()))
    if sessions_dir.exists():
        session_count = len(list(sessions_dir.glob("*.jsonl")))
        checks.append((f"  Session 文件数", True, f"{session_count}"))
    checks.append(("数据目录", data_dir.exists()))

    deps = [("PyYAML", "yaml"), ("scikit-learn", "sklearn"), ("jieba", "jieba")]
    for name, module in deps:
        try:
            __import__(module)
            checks.append((name, True))
        except ImportError:
            checks.append((name, False))

    try:
        import torch
        checks.append(("PyTorch CUDA", torch.cuda.is_available()))
    except ImportError:
        checks.append(("PyTorch", False))

    try:
        import deepspeed
        checks.append(("DeepSpeed", True))
    except ImportError:
        checks.append(("DeepSpeed (可选)", False))

    for item in checks:
        name, ok = item[0], item[1]
        extra = f" ({item[2]})" if len(item) > 2 else ""
        print(f"  {'✅' if ok else '❌'} {name}{extra}")

    all_ok = all(c[1] for c in checks)
    print(f"\n{'✅ All checks passed' if all_ok else '⚠️ 部分检查未通过，核心功能仍可使用'}")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenClaw Self-Trainer 数据收集")
    parser.add_argument("--init", action="store_true", help="初始化数据目录")
    parser.add_argument("--collect", action="store_true", help="收集完整交互链，保存训练集")
    parser.add_argument("--analyze", action="store_true", help="只输出分类统计，不保存文件")
    parser.add_argument("--check", action="store_true", help="检查安装状态")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入文件")
    parser.add_argument("--sessions-dir", type=str, default=None, help="OpenClaw sessions 目录路径")
    parser.add_argument("--data-dir", type=str, default=None, help="数据输出目录路径")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    if args.init:
        init_data_dir(args.data_dir)
    elif args.collect:
        collect_sessions(sessions_dir=args.sessions_dir, data_dir=args.data_dir, dry_run=args.dry_run)
    elif args.analyze:
        collect_sessions(sessions_dir=args.sessions_dir, analyze_only=True)
    elif args.check:
        check_installation(args.data_dir)
    else:
        parser.print_help()
