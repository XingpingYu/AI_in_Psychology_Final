"""按模型 + condition 统计 story_analogy 的 A/B/both/NONE 分布。"""
import collections
import json
from pathlib import Path

results = {}
for model_dir in Path("results/raw").iterdir():
    if not model_dir.is_dir():
        continue
    jf = model_dir / "story_analogy.jsonl"
    if not jf.exists():
        continue
    cnt = collections.Counter()
    correct = collections.Counter()
    total = collections.Counter()
    for line in jf.open("r", encoding="utf-8"):
        r = json.loads(line)
        parsed = r.get("parsed")
        cat = "NONE" if parsed is None else ("both" if parsed == "both" else parsed.upper())
        cond = r["subtype"]
        cnt[(cond, cat)] += 1
        total[cond] += 1
        if r.get("correct"):
            correct[cond] += 1
    results[model_dir.name] = (cnt, correct, total)

hdr = "model".ljust(25) + "cond".ljust(12) + "A".rjust(4) + "B".rjust(5) + "both".rjust(6) + "NONE".rjust(6) + "  acc"
print(hdr)
for m in sorted(results):
    cnt, correct, total = results[m]
    for cond in ["similarity", "analogy"]:
        a = cnt[(cond, "A")]
        b = cnt[(cond, "B")]
        both = cnt[(cond, "both")]
        non = cnt[(cond, "NONE")]
        acc = correct[cond] / total[cond] if total[cond] else 0
        print(f"{m:<25}{cond:<12}{a:>4}{b:>5}{both:>6}{non:>6}  {acc:.3f}")
