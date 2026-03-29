#!/usr/bin/env python3
"""
OpenClaw Self-Trainer - 自动评估脚本

对比新旧模型在测试集上的表现，客观评估训练效果。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path=None):
    default = Path(__file__).parent.parent / "config" / "defaults.yaml"
    path = Path(config_path) if config_path else default
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(model_path, test_set_path, config=None):
    """
    评估模型在测试集上的表现

    评估维度：
    - 任务完成度 (40%) — 工具调用是否正确
    - 幻觉检测 (30%) — 是否编造信息
    - 指令遵循 (20%) — 是否正确理解用户意图
    - 格式正确性 (10%) — 输出格式是否规范
    """
    config = config or load_config()
    weights = config.get("evaluation", {}).get("weights", {
        "task_completion": 0.4,
        "hallucination": 0.3,
        "instruction_following": 0.2,
        "format_correctness": 0.1,
    })

    # 加载测试集
    test_cases = []
    with open(test_set_path) as f:
        for line in f:
            test_cases.append(json.loads(line.strip()))

    print(f"📊 评估 {len(test_cases)} 个测试用例...")

    # TODO: 实现实际评估逻辑
    # 1. 让被评估的模型对每个测试用例生成回答
    # 2. 对比工具调用序列（如果有）
    # 3. 检测幻觉（与上下文矛盾的内容）
    # 4. 评估指令遵循度
    # 5. 检查格式

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "test_cases": len(test_cases),
        "overall_score": 0.0,
        "category_scores": {},
        "details": {
            "task_completion": 0.0,
            "hallucination": 0.0,
            "instruction_following": 0.0,
            "format_correctness": 0.0,
        },
        "per_category": {},
    }

    print("⚠️ 评估逻辑待实现")
    return results


def make_decision(report, config=None):
    """根据评估结果做出部署决策"""
    config = config or load_config()
    thresholds = config.get("evaluation", {})

    deploy_threshold = thresholds.get("deploy_threshold", 0.80)
    rollback_threshold = thresholds.get("rollback_threshold", 0.70)

    score = report["overall_score"]

    # 加载当前版本分数
    state_file = Path.home() / ".cache" / "self-trainer" / "data" / "reports" / "current_deployed.json"
    current_score = 0.0
    if state_file.exists():
        with open(state_file) as f:
            current = json.load(f)
            current_score = current.get("overall_score", 0.0)

    if score >= deploy_threshold and score > current_score:
        decision = "DEPLOY"
        reason = f"准确率 {score:.1%} ≥ 阈值 {deploy_threshold:.0%} 且高于当前 {current_score:.1%}"
    elif score < rollback_threshold:
        decision = "ROLLBACK"
        reason = f"准确率 {score:.1%} < 回滚阈值 {rollback_threshold:.0%}"
    else:
        decision = "CONTINUE"
        reason = f"准确率 {score:.1%}，继续收集数据"

    return {
        "decision": decision,
        "reason": reason,
        "score": score,
        "current_score": current_score,
    }


def run(data_dir=None, config_path=None, model_path=None):
    config = load_config(config_path)
    data_dir = Path(data_dir or config.get("paths", {}).get("data_dir", "~/.cache/self-trainer/data")).expanduser()

    test_set = data_dir / "test.jsonl"
    if not test_set.exists():
        print("❌ 测试集不存在，请先运行数据收集和训练")
        sys.exit(1)

    if not model_path:
        # 使用最新的 checkpoint
        models_dir = data_dir.parent / "models"
        if not models_dir.exists():
            print("❌ 没有找到模型 checkpoint")
            sys.exit(1)
        checkpoints = sorted(models_dir.iterdir())
        if not checkpoints:
            print("❌ 没有找到模型 checkpoint")
            sys.exit(1)
        model_path = checkpoints[-1]

    report = evaluate(model_path, test_set, config)
    decision = make_decision(report, config)
    report["decision"] = decision

    # 保存报告
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_file = reports_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 更新 latest
    with open(reports_dir / "latest.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印结果
    emoji = {"DEPLOY": "✅", "ROLLBACK": "🔴", "CONTINUE": "🟡"}
    print(f"\n{emoji.get(decision['decision'], '❓')} 决策: {decision['decision']}")
    print(f"   {decision['reason']}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenClaw Self-Trainer 评估")
    parser.add_argument("--model", type=str, default=None, help="模型路径")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    run(data_dir=args.data_dir, config_path=args.config, model_path=args.model)
