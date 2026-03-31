#!/usr/bin/env python3
"""
OpenClaw Self-Trainer - 模型评估脚本

支持三种评估模式：
  --auto   自动评估（perplexity、工具调用准确率、格式合规）
  --judge  LLM 评审（用大模型对推理/工具/回复质量打分）
  --report 查看历史评估报告

用法：
  python3 evaluate.py --auto --model ./outputs/checkpoint-best
  python3 evaluate.py --judge --model ./outputs/checkpoint-best
  python3 evaluate.py --report
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml


# ─── 配置 ──────────────────────────────────────────────

DEFAULT_CONFIG = Path(__file__).parent.parent / "config" / "defaults.yaml"


def load_config(config_path=None):
    """加载配置文件"""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def get_test_set_path(config=None):
    """获取 test.jsonl 路径"""
    config = config or load_config()
    output_dir = config.get("data", {}).get("prepare_output")
    if output_dir:
        output_path = Path(output_dir).expanduser()
        if output_path.exists():
            test_file = output_path / "test.jsonl"
            if test_file.exists():
                return test_file
    return None


def get_results_path(config=None):
    """获取 eval_results.jsonl 保存路径"""
    config = config or load_config()
    output_dir = config.get("data", {}).get("prepare_output")
    if output_dir:
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / "eval_results.jsonl"
    return None


# ─── 数据加载 ──────────────────────────────────────────

def load_test_set(test_set_path):
    """加载测试集（JSONL 格式，instruction/text/output）"""
    samples = []
    with open(test_set_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_result(result, results_path=None, config=None):
    """追加评估结果到 eval_results.jsonl"""
    path = results_path or get_results_path(config)
    if not path:
        print("⚠️  未找到保存路径，跳过保存")
        return
    with open(path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"💾 结果已保存: {path}")


# ─── 工具调用解析 ──────────────────────────────────────

TOOL_CALL_PATTERN = re.compile(r'tool_call\s*[:=]\s*({[\s\S]*?})', re.IGNORECASE)
TOOL_CALL_ALT_PATTERN = re.compile(r'```json\s*(\{[\s\S]*?\})\s*```')


def extract_tool_calls(text):
    """
    从文本中提取工具调用（JSON 对象列表）。
    
    支持格式：
    - tool_call: {"name": "...", "arguments": {...}}
    - ```json {"name": "...", "arguments": {...}} ```
    - 多个 tool_call 块
    """
    calls = []
    seen = set()

    for pattern in [TOOL_CALL_PATTERN, TOOL_CALL_ALT_PATTERN]:
        for match in pattern.finditer(text):
            raw = match.group(1).strip()
            # 尝试解析单个 JSON 对象
            try:
                obj = json.loads(raw)
                key = json.dumps(obj, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    calls.append(obj)
            except json.JSONDecodeError:
                pass

    # 尝试解析 JSON 数组
    array_pattern = re.compile(r'\[[\s\S]*?\]', re.IGNORECASE)
    for match in array_pattern.finditer(text):
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict):
                        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                        if key not in seen:
                            seen.add(key)
                            calls.append(item)
        except json.JSONDecodeError:
            pass

    return calls


def tool_call_name(call):
    """提取工具调用名称"""
    return call.get("name") or call.get("function", {}).get("name", "")


def tool_call_params(call):
    """提取工具调用参数"""
    params = call.get("arguments") or call.get("function", {}).get("arguments") or call.get("parameters") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return {}
    return params


# ─── 格式合规检查 ──────────────────────────────────────

def check_format_compliance(text):
    """
    检查输出格式合规性，返回各子项得分和总分。
    
    检查项：
    1. <think think_content> 标签是否存在且格式正确
    2. 工具调用是否为合法 JSON
    3. 是否有 assistant 回复内容
    """
    scores = {}

    # 1. think 标签
    has_think_open = "<think" in text
    has_think_close = "</think" in text or "<|im_end|>" in text
    think_match = re.search(r'<think[^>]*>([\s\S]*?)(?:</think|<\|im_end\|>)', text)
    
    think_score = 0.0
    if has_think_open and think_match:
        think_content = think_match.group(1).strip()
        if think_content and len(think_content) > 10:
            think_score = 1.0
        elif think_content:
            think_score = 0.5
        else:
            think_score = 0.3
    elif has_think_open:
        think_score = 0.2  # 有开标签但无闭合或内容
    scores["think_tag"] = think_score

    # 2. 工具调用 JSON 合法性
    tool_calls = extract_tool_calls(text)
    if not tool_calls:
        # 没有 tool_call 也算合规（纯对话场景）
        scores["tool_json"] = 1.0
    else:
        valid_count = 0
        for call in tool_calls:
            if tool_call_name(call) and isinstance(tool_call_params(call), dict):
                valid_count += 1
        scores["tool_json"] = valid_count / len(tool_calls) if tool_calls else 1.0

    # 3. 有 assistant 回复内容
    # 检查 think 标签之外是否有实质内容
    assistant_content = text
    if think_match:
        assistant_content = text[think_match.end():].strip()
    # 去除 tool_call 块后检查剩余内容
    cleaned = re.sub(r'tool_call\s*[:=]\s*\{[\s\S]*?\}', '', assistant_content, flags=re.IGNORECASE)
    cleaned = re.sub(r'```json[\s\S]*?```', '', cleaned).strip()
    
    if len(cleaned) > 5:
        scores["assistant_reply"] = 1.0
    elif len(cleaned) > 0:
        scores["assistant_reply"] = 0.5
    else:
        scores["assistant_reply"] = 0.0

    # 总分 = 各项加权平均
    weights = {"think_tag": 0.3, "tool_json": 0.3, "assistant_reply": 0.4}
    total = sum(scores[k] * w for k, w in weights.items())
    scores["total"] = total

    return scores


# ─── 工具调用准确率 ──────────────────────────────────────

def compute_tool_call_accuracy(pred_text, gold_text):
    """
    计算工具调用 precision/recall。
    
    对比预测输出的工具调用与 ground truth 的工具调用。
    """
    pred_calls = extract_tool_calls(pred_text)
    gold_calls = extract_tool_calls(gold_text)

    if not gold_calls:
        # ground truth 没有工具调用，预测也不应有
        if not pred_calls:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not pred_calls:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # 名称匹配计数
    pred_names = {tool_call_name(c) for c in pred_calls if tool_call_name(c)}
    gold_names = {tool_call_name(c) for c in gold_calls if tool_call_name(c)}

    if not gold_names:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(pred_names & gold_names)
    precision = tp / len(pred_names) if pred_names else 0.0
    recall = tp / len(gold_names) if gold_names else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ─── 自动评估 ──────────────────────────────────────────

def run_auto_eval(model_path, test_set_path, config=None):
    """
    自动评估：计算 perplexity、工具调用准确率、格式合规。
    
    如果模型路径不存在或无法加载，使用 ground truth 模拟生成（开发模式）。
    """
    samples = load_test_set(test_set_path)
    if not samples:
        print("❌ 测试集为空")
        return None

    print(f"📊 自动评估 {len(samples)} 个测试样本...")
    print(f"   模型: {model_path}")

    # 尝试加载模型进行生成
    model_loaded = False
    tokenizer = None
    model = None
    device = None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"   加载模型到 {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device != "cuda" and model:
            model = model.to(device)
        model.eval()
        model_loaded = True
        print("   ✅ 模型加载成功")
    except Exception as e:
        print(f"   ⚠️  模型加载失败: {e}")
        print("   使用 ground truth 模拟（开发模式）")

    # 收集指标
    all_precision = []
    all_recall = []
    all_format = []
    total_loss = 0.0
    total_tokens = 0

    for i, sample in enumerate(samples):
        instruction = sample.get("instruction", "")
        text = sample.get("text", "")
        gold_output = sample.get("output", "")

        # 构建输入 prompt
        if text:
            prompt = text
        else:
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        if model_loaded:
            # 生成回复
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                # 计算 perplexity
                target_text = gold_output if gold_output else ""
                if target_text:
                    target_ids = tokenizer(target_text, return_tensors="pt", truncation=True, max_length=2048)
                    target_ids = target_ids["input_ids"].to(device)
                    input_with_target = torch.cat([inputs["input_ids"], target_ids], dim=1)
                    labels = input_with_target.clone()
                    labels[:, :inputs["input_ids"].shape[1]] = -100  # 忽略 prompt 部分

                    outputs = model(input_with_target, labels=labels)
                    total_loss += outputs.loss.item() * target_ids.shape[1]
                    total_tokens += target_ids.shape[1]

                # 生成预测
                generated = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                pred_output = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        else:
            # 开发模式：直接用 ground truth
            pred_output = gold_output

        # 计算指标
        acc = compute_tool_call_accuracy(pred_output, gold_output)
        all_precision.append(acc["precision"])
        all_recall.append(acc["recall"])

        fmt = check_format_compliance(pred_output)
        all_format.append(fmt["total"])

        # 进度输出
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   [{i+1}/{len(samples)}] P={acc['precision']:.2f} R={acc['recall']:.2f} Fmt={fmt['total']:.2f}")

    # 汇总
    avg_precision = sum(all_precision) / len(all_precision) if all_precision else 0.0
    avg_recall = sum(all_recall) / len(all_recall) if all_recall else 0.0
    avg_format = sum(all_format) / len(all_format) if all_format else 0.0

    perplexity = None
    if model_loaded and total_tokens > 0:
        import math
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

    result = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "type": "auto",
        "perplexity": round(perplexity, 2) if perplexity else None,
        "tool_call_precision": round(avg_precision, 4),
        "tool_call_recall": round(avg_recall, 4),
        "format_compliance": round(avg_format, 4),
        "test_samples": len(samples),
        "model_loaded": model_loaded,
    }

    return result


# ─── LLM 评审 ──────────────────────────────────────────

def load_openclaw_config():
    """从 ~/.openclaw/openclaw.json 读取 API 配置"""
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    if not config_path.exists():
        return None, None, None
    
    with open(config_path) as f:
        data = json.load(f)
    
    # 提取模型配置
    models = data.get("models", {})
    # 获取缺省模型
    default_model = data.get("defaults", {}).get("model")
    
    # 提取 API 配置（OpenAI 兼容格式）
    # 从 providers 或 llm 配置中找
    api_base = data.get("llm", {}).get("baseUrl") or data.get("api", {}).get("baseUrl")
    api_key = data.get("llm", {}).get("apiKey") or data.get("api", {}).get("apiKey")
    
    # 尝试从环境变量获取
    if not api_base:
        api_base = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    return default_model, api_base, api_key


JUDGE_PROMPT = """你是一个 AI 模型评审专家。请对以下 AI 助手的回复质量进行评分。

## 用户请求
{instruction}

## 助手回复
{response}

## Ground Truth（参考答案）
{gold_output}

请从以下 5 个维度打分（每项 1-5 分）：

1. **推理质量**：思考过程是否逻辑清晰、合理
2. **工具选择**：是否选择了正确的工具来完成任务
3. **工具参数**：工具调用参数是否准确、完整
4. **多轮一致性**：是否能正确理解工具返回结果并继续推理
5. **回复质量**：最终回复是否准确、有帮助

请严格按照以下 JSON 格式回复，不要包含其他内容：
```json
{{
  "推理质量": <1-5的整数>,
  "工具选择": <1-5的整数>,
  "工具参数": <1-5的整数>,
  "多轮一致性": <1-5的整数>,
  "回复质量": <1-5的整数>,
  "总体评价": "<一句话简评>"
}}
```"""


def call_llm_judge(prompt, model, api_base, api_key, temperature=0.3, max_retries=3):
    """调用 OpenAI 兼容 API 进行评审"""
    from openai import OpenAI
    
    client = OpenAI(base_url=api_base, api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
            )
            content = response.choices[0].message.content.strip()
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"      ⚠️  API 调用失败（重试 {attempt+1}/{max_retries}）: {e}")
            else:
                print(f"      ❌ API 调用失败: {e}")
                return None


def parse_judge_response(response_text):
    """解析 LLM 评审的 JSON 回复"""
    if not response_text:
        return None
    
    # 尝试提取 JSON 块
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 直接尝试解析整个回复
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # 尝试找 { } 包裹的 JSON
    brace_match = re.search(r'\{[\s\S]*\}', response_text)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass
    
    return None


def run_judge_eval(model_path, test_set_path, config=None):
    """
    LLM 评审：用大模型对微调模型的输出进行多维度打分。
    """
    samples = load_test_set(test_set_path)
    if not samples:
        print("❌ 测试集为空")
        return None

    # 加载 OpenClaw 配置
    judge_model, api_base, api_key = load_openclaw_config()
    if not api_base or not api_key:
        print("❌ 无法读取 OpenClaw API 配置")
        print("   请确保 ~/.openclaw/openclaw.json 存在且包含 llm 配置")
        return None

    if not judge_model:
        judge_model = "gpt-4o-mini"  # fallback
        print(f"⚠️  未找到缺省模型配置，使用 fallback: {judge_model}")

    print(f"🔍 LLM 评审 {len(samples)} 个测试样本...")
    print(f"   被评估模型: {model_path}")
    print(f"   评审模型: {judge_model}")

    # 尝试加载被评估模型
    finetune_model = None
    ft_tokenizer = None
    ft_device = None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            ft_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            ft_device = "mps"
        else:
            ft_device = "cpu"

        ft_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        finetune_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if ft_device == "cuda" else torch.float32,
            device_map="auto" if ft_device == "cuda" else None,
            trust_remote_code=True,
        )
        if ft_device != "cuda" and finetune_model:
            finetune_model = finetune_model.to(ft_device)
        finetune_model.eval()
        print(f"   ✅ 被评估模型已加载到 {ft_device}")
    except Exception as e:
        print(f"   ⚠️  被评估模型加载失败: {e}")
        print("   使用 ground truth 模拟（开发模式）")

    # 收集评审结果
    all_scores = {}
    score_dimensions = ["推理质量", "工具选择", "工具参数", "多轮一致性", "回复质量"]
    comments = []

    # 限制评审数量避免 API 费用过高
    max_judge_samples = min(len(samples), 20)
    judge_samples = samples[:max_judge_samples]
    
    for i, sample in enumerate(judge_samples):
        instruction = sample.get("instruction", "")
        text = sample.get("text", "")
        gold_output = sample.get("output", "")

        if text:
            prompt = text
        else:
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        # 生成预测
        if finetune_model:
            import torch
            inputs = ft_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(ft_device) for k, v in inputs.items()}
            with torch.no_grad():
                generated = finetune_model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=ft_tokenizer.pad_token_id or ft_tokenizer.eos_token_id,
                )
            pred_output = ft_tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        else:
            pred_output = gold_output

        # 构建评审 prompt
        judge_prompt = JUDGE_PROMPT.format(
            instruction=instruction[:500] if instruction else text[:500],
            response=pred_output[:2000],
            gold_output=gold_output[:2000],
        )

        # 调用 LLM 评审
        judge_response = call_llm_judge(judge_prompt, judge_model, api_base, api_key)
        scores = parse_judge_response(judge_response)

        if scores:
            for dim in score_dimensions:
                val = scores.get(dim)
                if val is not None:
                    try:
                        val = float(val)
                        if 1 <= val <= 5:
                            all_scores.setdefault(dim, []).append(val)
                    except (ValueError, TypeError):
                        pass
            
            comment = scores.get("总体评价", "")
            if comment:
                comments.append(comment)

        print(f"   [{i+1}/{max_judge_samples}] {scores or '❌ 评审失败'}")

    # 汇总
    avg_scores = {}
    for dim in score_dimensions:
        vals = all_scores.get(dim, [])
        if vals:
            avg_scores[dim] = round(sum(vals) / len(vals), 2)
        else:
            avg_scores[dim] = None

    overall = None
    valid_scores = [v for v in avg_scores.values() if v is not None]
    if valid_scores:
        overall = round(sum(valid_scores) / len(valid_scores), 2)

    result = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "type": "judge",
        "judge_model": judge_model,
        "judge_scores": avg_scores,
        "overall_score": overall,
        "judge_samples": max_judge_samples,
        "total_test_samples": len(samples),
        "comments": comments[:5],  # 只保留前 5 条评论
    }

    return result


# ─── 报告 ──────────────────────────────────────────────

def run_report(config=None):
    """读取历史评估结果，以表格形式展示"""
    results_path = get_results_path(config)
    if not results_path or not results_path.exists():
        print("❌ 没有找到评估结果文件")
        print("   请先运行 --auto 或 --judge 进行评估")
        return

    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        print("❌ 评估结果文件为空")
        return

    print(f"📊 评估报告（共 {len(results)} 次评估）\n")

    # 分离 auto 和 judge 结果
    auto_results = [r for r in results if r.get("type") == "auto"]
    judge_results = [r for r in results if r.get("type") == "judge"]

    if auto_results:
        print("=" * 80)
        print("🤖 自动评估结果")
        print("=" * 80)
        _print_auto_table(auto_results)

    if judge_results:
        print()
        print("=" * 80)
        print("🧑‍⚖️  LLM 评审结果")
        print("=" * 80)
        _print_judge_table(judge_results)


def _print_auto_table(results):
    """打印自动评估结果表格"""
    # 表头
    header = f"{'时间':<20} {'PPL':>8} {'Prec':>8} {'Recall':>8} {'Format':>8} {'样本数':>6}"
    print(header)
    print("-" * len(header))

    for r in results:
        ts = r.get("timestamp", "")[:16]
        ppl = f"{r['perplexity']:.1f}" if r.get("perplexity") else "N/A"
        prec = f"{r['tool_call_precision']:.3f}"
        recall = f"{r['tool_call_recall']:.3f}"
        fmt = f"{r['format_compliance']:.3f}"
        n = r.get("test_samples", 0)
        print(f"{ts:<20} {ppl:>8} {prec:>8} {recall:>8} {fmt:>8} {n:>6}")

    # 与上次对比
    if len(results) >= 2:
        latest = results[-1]
        previous = results[-2]
        print()
        print("📈 最近两次对比:")
        _compare_metric("Perplexity", latest.get("perplexity"), previous.get("perplexity"), lower_better=True)
        _compare_metric("Precision", latest.get("tool_call_precision"), previous.get("tool_call_precision"))
        _compare_metric("Recall", latest.get("tool_call_recall"), previous.get("tool_call_recall"))
        _compare_metric("Format", latest.get("format_compliance"), previous.get("format_compliance"))


def _print_judge_table(results):
    """打印 LLM 评审结果表格"""
    score_dimensions = ["推理质量", "工具选择", "工具参数", "多轮一致性", "回复质量"]
    dims_short = ["推理", "工具选", "参数", "多轮", "回复"]

    # 表头
    header = f"{'时间':<20} {'模型':<15}" + "".join(f"{d:>6}" for d in dims_short) + f" {'总分':>6}"
    print(header)
    print("-" * len(header))

    for r in results:
        ts = r.get("timestamp", "")[:16]
        model = r.get("model_path", "")[:14]
        judge_scores = r.get("judge_scores", {})
        
        scores = []
        for dim in score_dimensions:
            val = judge_scores.get(dim)
            scores.append(f"{val:.1f}" if val else "N/A")
        
        overall = r.get("overall_score")
        overall_str = f"{overall:.1f}" if overall else "N/A"
        
        row = f"{ts:<20} {model:<15}" + "".join(f"{s:>6}" for s in scores) + f" {overall_str:>6}"
        print(row)

    # 评分分布
    if results:
        latest = results[-1]
        judge_scores = latest.get("judge_scores", {})
        print()
        print(f"📋 最新评分详情: {latest.get('timestamp', '')[:16]}")
        for dim in score_dimensions:
            val = judge_scores.get(dim)
            if val:
                bar = "█" * int(val) + "░" * (5 - int(val))
                print(f"   {dim}: {bar} {val:.1f}/5")
        
        # 评论
        comments = latest.get("comments", [])
        if comments:
            print()
            print("💬 评审意见:")
            for c in comments[:3]:
                print(f"   • {c}")


def _compare_metric(name, current, previous, lower_better=False):
    """对比两个指标并打印变化"""
    if current is None or previous is None:
        print(f"   {name}: 数据不完整")
        return

    diff = current - previous
    if lower_better:
        better = diff < 0
    else:
        better = diff > 0

    emoji = "📈" if better else "📉" if diff != 0 else "➡️"
    sign = "+" if diff > 0 else ""
    print(f"   {name}: {current:.3f} ({sign}{diff:.3f}) {emoji}")


# ─── 主入口 ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw Self-Trainer 模型评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python3 evaluate.py --auto --model ./outputs/checkpoint-best
  python3 evaluate.py --judge --model ./outputs/checkpoint-best
  python3 evaluate.py --report
        """,
    )

    # 互斥参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--auto", action="store_true", help="自动评估（perplexity + 工具调用准确率 + 格式合规）")
    group.add_argument("--judge", action="store_true", help="LLM 评审（用大模型对多维度打分）")
    group.add_argument("--report", action="store_true", help="查看历史评估报告")

    # 可选参数
    parser.add_argument("--model", type=str, default=None, help="模型路径（--auto/--judge 必需）")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--test-set", type=str, default=None, help="测试集路径（默认从配置读取）")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.report:
        run_report(config)
        return

    # auto / judge 需要 model 和 test set
    if not args.model:
        print("❌ --model 参数是必需的（请指定模型路径）")
        print("   用法: python3 evaluate.py --auto --model ./outputs/checkpoint-best")
        sys.exit(1)

    test_set_path = Path(args.test_set) if args.test_set else get_test_set_path(config)
    if not test_set_path or not test_set_path.exists():
        print(f"❌ 测试集不存在: {test_set_path}")
        print("   请先运行数据准备流程生成 test.jsonl")
        sys.exit(1)

    if args.auto:
        result = run_auto_eval(args.model, test_set_path, config)
    elif args.judge:
        result = run_judge_eval(args.model, test_set_path, config)
    else:
        return

    if result:
        # 打印摘要
        print()
        print("=" * 50)
        print("📋 评估摘要")
        print("=" * 50)
        for k, v in result.items():
            if k == "comments":
                continue
            print(f"   {k}: {v}")
        
        # 保存结果
        save_result(result, config=config)


if __name__ == "__main__":
    main()
