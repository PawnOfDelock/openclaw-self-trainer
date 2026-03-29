# Self-Trainer Setup Guide

本文档面向 OpenClaw AI 助手。当用户要求安装 openclaw-self-trainer 时，请按以下步骤执行。

## 前置检查

按顺序执行以下检查，任何一项失败则告知用户并提供解决方案：

1. **GPU 检查**
   ```bash
   nvidia-smi
   ```
   - 需要至少一张 GPU，显存 ≥ 8GB（用于 3B 模型推理和训练）
   - 如果失败：告知用户需要一台有 GPU 的机器

2. **Python 环境**
   ```bash
   python3 --version
   pip3 --version
   ```
   - Python ≥ 3.10
   - 如果失败：引导安装

3. **DeepSpeed**
   ```bash
   deepspeed --version
   ```
   - 如果失败：`pip install deepspeed`

4. **磁盘空间**
   ```bash
   df -h ~
   ```
   - 至少需要 20GB 可用空间（模型权重 + 训练数据 + checkpoint）
   - 如果失败：告知用户清理空间

5. **OpenClaw 版本**
   ```bash
   openclaw --version
   ```
   - 确认 OpenClaw 正常运行

## 安装步骤

### Step 1: 安装 Python 依赖

```bash
cd ~/openclaw-self-trainer
pip install -r requirements.txt
```

### Step 2: 下载基座模型

询问用户选择基座模型，然后下载到 `~/.cache/self-trainer/base-model/`：

```bash
mkdir -p ~/.cache/self-trainer/base-model
# 根据用户选择的模型执行对应的下载命令
```

推荐基座模型：
- **GLM-4-9B** — 如果显存充足
- **Qwen2.5-3B** — 显存紧张时
- **Llama-3.2-3B** — 通用选择

### Step 3: 初始化数据目录

```bash
python3 scripts/collect.py --init
```

这会创建 `~/.cache/self-trainer/data/` 目录结构：
```
data/
├── raw/          # 原始 session 日志
├── cleaned/      # 清洗后的数据
├── train.jsonl   # 训练集
├── val.jsonl     # 验证集
├── test.jsonl    # 测试集（锁定，永不参与训练）
└── reports/      # 评估报告
```

### Step 4: 配置定时任务

添加两个 cron 任务：

**每日数据收集**（每天凌晨 3 点）：
```bash
openclaw cron add --schedule "0 3 * * *" --prompt "运行 self-trainer 数据收集：执行 python3 ~/openclaw-self-trainer/scripts/collect.py --collect。如果新增数据 ≥ 50 条，执行 python3 ~/openclaw-self-trainer/scripts/train.py --auto。完成后发送 Slack 通知。"
```

**每周评估报告**（每周日早上 8 点）：
```bash
openclaw cron add --schedule "0 8 * * 0" --prompt "生成 self-trainer 周报：执行 python3 ~/openclaw-self-trainer/scripts/report.py。发送报告到 Slack。"
```

### Step 5: 更新 HEARTBEAT.md

在 OpenClaw 的 HEARTBEAT.md 中添加：

```markdown
## Self-Trainer 状态检查
- 检查 ~/.cache/self-trainer/data/reports/latest.json
- 如果有新的评估结果或异常告警，立即通知用户
```

## 验证安装

安装完成后，执行验证：

```bash
python3 scripts/collect.py --check
```

如果输出 `✅ All checks passed`，则安装成功。

告知用户：
> ✅ openclaw-self-trainer 安装完成！
> - 每天凌晨 3 点自动收集数据
> - 累计 ≥ 50 条新数据后自动触发训练
> - 每周日生成训练报告
> 你也可以随时说"训练状态"查看当前进度。

## 手动触发命令

安装完成后，用户可以使用以下命令：

- "收集数据" → 运行 collect.py
- "开始训练" → 运行 train.py
- "训练状态" → 查看最新报告
- "训练报告" → 生成并发送详细报告
- "回滚模型" → 回滚到上一个版本
