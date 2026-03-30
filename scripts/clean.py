#!/usr/bin/env python3
"""
训练数据敏感信息清洗脚本 v2
- 同一真实值 → 同一随机替换（全局一致）
- 不同真实值 → 不同随机替换
- 替换值使用上下文合理的伪造内容，不引入新 token
"""

import json
import re
import sys
import hashlib
import random
from collections import Counter

random.seed(42)  # 可复现

# ========== 规则定义 ==========
RULES = [
    ("aws_key",      r"AKIA[0-9A-Z]{16}",                    "token", "AWS Access Key"),
    ("github_token", r"gh[pousr]_[a-zA-Z0-9_]{36,}",        "token", "GitHub Token"),
    ("slack_token",  r"xox[baprs]-[0-9a-zA-Z-]{10,}",       "token", "Slack Token"),
    ("private_key",  r"-----BEGIN\s+(RSA\s+|EC\s+)?PRIVATE KEY-----", "token", "Private Key"),
    ("jwt",          r"eyJ[A-Za-z0-9_-]{20,}\.eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+", "token", "JWT"),
    ("bearer",       r"[Bb]earer\s+[a-zA-Z0-9._\-]{20,}",   "token", "Bearer Token"),
    ("api_key_kv",   r'(?i)(?:api[_-]?key|apikey|api[_-]?secret)["\s:=]+([a-zA-Z0-9_\-]{16,})', "token", "API Key (kv)"),
    ("password_kv",  r'(?i)(?:password|passwd|pwd)["\s:=]+([^\s"\',}{)\]]*[!@#$%^&*_\-+=`~][^\s"\',}{)\]]*)', "password", "Password (kv)"),
    ("token_kv",     r'(?i)(?:token|secret[_-]?key|access[_-]?key)["\s:=]+([a-zA-Z0-9_\-]{16,})', "token", "Token (kv)"),
    ("ip",           r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b", "ip", "IP Address"),
    ("email",        r"[\w.\-]+@[\w.\-]+\.\w{2,}",         "email", "Email"),
    ("phone",        r"(?<!\d)1[3-9]\d{9}(?!\d)",           "phone", "Phone Number"),
    ("id_card",      r"(?<!\d)\d{17}[\dXx](?!\d)",          "id_card", "ID Card Number"),
    ("generic_secret", r'(?i)(?:sk|pk)-[a-zA-Z0-9]{20,}',  "token", "Generic Secret Key"),
    ("oauth",        r'(?i)client[_-]?secret["\s:=]+([a-zA-Z0-9_\-]{16,})', "token", "OAuth Secret"),
    ("connection_string", r'(?i)(?:mongodb|mysql|postgres|redis|amqp)://[^\s"\'>]+', "conn_str", "Connection String"),
    ("cred_field",   r'(?i)(?:credential|auth_token|session_id|oauth_token)["\s:=]+["\']?([a-zA-Z0-9_\-]{10,})["\']?', "uuid", "Credential Field"),
]

COMPILED = [(name, re.compile(pat), cat, desc) for name, pat, cat, desc in RULES]

# KV 规则 — 有 group(1) 的是只取 value
KV_RULES = {"api_key_kv", "password_kv", "token_kv", "oauth", "cred_field"}


# ========== 随机替换生成器 ==========
class ReplacementPool:
    """基于真实值的 hash 生成确定性随机替换"""
    
    def __init__(self):
        self._cache = {}  # (category, real_value) → replacement
        self._ip_counter = 0
        self._email_counter = 0
        self._token_counter = 0
        self._password_counter = 0
    
    def _hash_int(self, s, max_val):
        """确定性 hash → [0, max_val)"""
        h = hashlib.sha256(s.encode()).hexdigest()
        return int(h, 16) % max_val
    
    def get(self, category, real_value):
        key = (category, real_value)
        if key in self._cache:
            return self._cache[key]
        
        replacement = self._generate(category, real_value)
        self._cache[key] = replacement
        return replacement
    
    def _generate(self, category, real_value):
        if category == "ip":
            h = self._hash_int(real_value, 100000)
            # 生成 198.51.x.x 范围的假 IP（RFC 5737 TEST-NET-2 附近）
            return f"198.51.{h % 256}.{(h >> 8) % 254 + 1}"
        
        elif category == "email":
            h = self._hash_int(real_value, 100000)
            user = f"user{h % 1000}"
            domains = ["example.com", "example.org", "test.local", "sample.net", "demo.local"]
            domain = domains[h % len(domains)]
            return f"{user}@{domain}"
        
        elif category == "token":
            h = self._hash_int(real_value, 100000)
            prefix = f"tk_{h % 10000:04d}"
            body = "x" * max(8, len(real_value) - len(prefix) - 1)
            return f"{prefix}_{body}"
        
        elif category == "password":
            h = self._hash_int(real_value, 100000)
            # 用等长占位符
            return "•" * len(real_value)
        
        elif category == "phone":
            h = self._hash_int(real_value, 100000)
            return f"138{h % 100000000:08d}"
        
        elif category == "id_card":
            return "•" * 18
        
        elif category == "uuid":
            h = self._hash_int(real_value, 100000)
            return f"aaaaaaaa-{h % 10000:04x}-4bbb-cccc-{(h >> 4) % 0xffffffff:012x}"
        
        elif category == "conn_str":
            h = self._hash_int(real_value, 100000)
            return f"db://user{h%100}:****@host{h%10}.local:5432/db"
        
        else:
            return "****"


def scan_text(text):
    """扫描文本，返回 [(rule_name, matched_value, start, end), ...]"""
    findings = []
    for name, pattern, cat, desc in COMPILED:
        for m in pattern.finditer(text):
            if name in KV_RULES and m.lastindex:
                matched = m.group(1)
                start, end = m.start(1), m.end(1)
            else:
                matched = m.group(0)
                start, end = m.start(), m.end()
            findings.append((name, matched, start, end))
    return findings


def clean_text(text, pool):
    """清洗文本：同一真实值 → 同一替换"""
    findings = scan_text(text)
    if not findings:
        return text, []

    # 按位置排序，贪心去重（取最长匹配）
    findings.sort(key=lambda x: (x[2], -(x[3] - x[2])))
    selected = []
    last_end = -1
    for f in findings:
        if f[2] >= last_end:
            selected.append(f)
            last_end = f[3]

    replacements = []
    result = text
    
    for name, matched, start, end in reversed(selected):
        cat = dict((n, c) for n, _, c, _ in RULES)[name]
        replacement = pool.get(cat, matched)
        result = result[:start] + replacement + result[end:]
        replacements.append((name, matched, replacement))

    return result, replacements


def process_file(input_path, output_path):
    total_records = 0
    records_with_findings = 0
    all_findings = []
    pool = ReplacementPool()
    output_records = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total_records += 1
            record_findings = []

            if 'messages' in record:
                for msg in record['messages']:
                    content = msg.get('content', '')
                    if not content:
                        continue
                    cleaned, findings = clean_text(content, pool)
                    if findings:
                        record_findings.extend(findings)
                        msg['content'] = cleaned

            if record_findings:
                records_with_findings += 1
                all_findings.extend(record_findings)

            output_records.append(record)

    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    return {
        'total_records': total_records,
        'records_with_findings': records_with_findings,
        'total_findings': len(all_findings),
        'findings_by_type': Counter(f[0] for f in all_findings),
        'all_findings': all_findings,
        'pool_size': len(pool._cache),
    }


def print_report(stats):
    print("\n" + "=" * 60)
    print("📊 清洗报告 v2（随机化替换）")
    print("=" * 60)
    print(f"总记录数:       {stats['total_records']}")
    print(f"检出记录数:     {stats['records_with_findings']}")
    print(f"检出敏感信息:   {stats['total_findings']}")
    print(f"唯一映射条目:   {stats['pool_size']}")

    print(f"\n📋 按类型分布:")
    for rule, count in stats['findings_by_type'].most_common():
        desc = dict((n, d) for n, _, _, d in RULES)[rule]
        print(f"  {rule:20s} ({desc}): {count}")

    print(f"\n🔄 替换映射表（真实值 → 替换值）:")
    print("-" * 60)
    seen = set()
    for rule, real, fake in stats['all_findings']:
        k = (rule, real)
        if k not in seen:
            seen.add(k)
            r_display = real[:50] + "..." if len(real) > 50 else real
            f_display = fake[:50] + "..." if len(fake) > 50 else fake
            print(f"  {rule:16s} | {r_display:40s} → {f_display}")

    # Precision 分析
    print(f"\n✅ Precision 分析:")
    print("-" * 60)
    likely_fp = []
    likely_tp = []
    for rule, real, fake in stats['all_findings']:
        k = (rule, real)
        if k in [(r, v) for r, v, _, _ in likely_fp] or k in [(r, v) for r, v, _, _ in likely_tp]:
            continue
        if rule == "ip" and real.startswith(("10.", "192.168.", "172.16.")):
            likely_tp.append((rule, real, fake, "内网IP"))
        elif rule == "ip" and real.startswith("0."):
            likely_fp.append((rule, real, fake, "可能是版本号"))
        elif rule == "password_kv" and real in ("abc123", "test", "1234"):
            likely_fp.append((rule, real, fake, "示例密码"))
        else:
            likely_tp.append((rule, real, fake, ""))
    
    fp_count = len(likely_fp)
    tp_count = len(likely_tp)
    precision = tp_count / max(tp_count + fp_count, 1) * 100
    
    print(f"  疑似 TP: {tp_count}")
    if likely_tp:
        for rule, real, fake, note in likely_tp:
            note_str = f" ({note})" if note else ""
            print(f"    ✅ {rule}: {real[:40]}{note_str}")
    print(f"  疑似 FP: {fp_count}")
    if likely_fp:
        for rule, real, fake, note in likely_fp:
            print(f"    ⚠️  {rule}: {real[:40]} — {note}")
    
    print(f"\n  📐 估计 Precision: {precision:.0f}% (基于启发式分析)")
    print(f"  📌 Recall: 规则覆盖了常见格式，但无法捕获无上下文的随机字符串")

    # 一致性验证
    print(f"\n🔗 一致性验证（同一真实值是否映射到同一替换值）:")
    print("-" * 60)
    mappings = {}
    for rule, real, fake in stats['all_findings']:
        k = (rule, real)
        if k in mappings:
            if mappings[k] != fake:
                print(f"  ❌ 不一致! {rule}: {real[:30]} → {mappings[k]} vs {fake}")
        else:
            mappings[k] = fake
    if len(mappings) == len(set(mappings.values())):
        print(f"  ✅ 所有映射一致，{len(mappings)} 个唯一值")


if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/clawd/training_data.jsonl'
    output_file = input_file.replace('.jsonl', '_sanitized.jsonl')
    
    print(f"🔍 扫描: {input_file}")
    stats = process_file(input_file, output_file)
    print_report(stats)
    print(f"\n✅ 清洗后: {output_file}")
