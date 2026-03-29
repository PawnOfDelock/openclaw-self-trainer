# Self-Trainer Skill

OpenClaw 自我训练系统的日常使用指南。

## 描述

自动从 OpenClaw 对话日志中收集数据、训练本地小模型、评估效果、渐进式路由请求到小模型以降低云端 API 成本。

## 触发条件

当用户提到以下关键词时激活此 skill：
- "训练状态"、"训练报告"、"收集数据"、"开始训练"
- "self-trainer"、"小模型"、"路由状态"
- "模型准确率"、"训练进度"

## 数据收集

```bash
python3 ~/openclaw-self-trainer/scripts/collect.py
```

功能：
- 从 OpenClaw session 日志中提取对话
- 清洗元数据噪音（去掉 untrusted metadata 前缀等）
- 按类别分类（技术开发、工具操作、生活杂聊、知识讨论）
- 追加到训练集

## 训练

```bash
python3 ~/openclaw-self-trainer/scripts/train.py [--auto]
```

参数：
- `--auto` — 自动模式，检查新数据量，够 50 条才训练
- `--epochs N` — 训练轮数（默认 3）
- `--base-model PATH` — 指定基座模型路径

流程：
1. 合并新数据到训练集
2. 从训练集切出 val set（test set 已锁定）
3. DeepSpeed SFT 微调
4. 自动评估
5. 生成报告到 `~/.cache/self-trainer/data/reports/`

## 评估

```bash
python3 ~/openclaw-self-trainer/scripts/evaluate.py
```

评估维度：
- **任务完成度 (40%)** — 工具调用是否正确
- **幻觉检测 (30%)** — 是否编造信息
- **指令遵循 (20%)** — 是否正确理解用户意图
- **格式正确性 (10%)** — 输出格式是否规范

决策阈值：
- 准确率 ≥ 80% 且高于当前版本 → 部署
- 准确率 < 70% → 自动回滚
- 其他 → 继续收集数据

## 路由规则

路由规则存储在 `~/.cache/self-trainer/config/routing.json`：

```json
{
  "rules": [
    {"category": "工具操作", "accuracy": 0.85, "route": "small_model"},
    {"category": "生活闲聊", "accuracy": 0.91, "route": "small_model"},
    {"category": "技术开发", "accuracy": 0.72, "route": "large_model"},
    {"category": "default", "route": "large_model"}
  ],
  "fallback_to_large": true
}
```

## 模型管理

```bash
# 查看版本
python3 ~/openclaw-self-trainer/scripts/model.py --list

# 回滚
python3 ~/openclaw-self-trainer/scripts/model.py --rollback

# 切换部署版本
python3 ~/openclaw-self-trainer/scripts/model.py --deploy VERSION
```

模型版本存储在 `~/.cache/self-trainer/models/`，每次训练生成一个带时间戳的 checkpoint。
