"""按模型列出 token 预估和美元成本。

token 估计取自 DeepSeek 跑出来的实测平均(每题 input/output token)。
价格表来自 OpenAI / DeepSeek 公开 pricing(单位 $/M tokens)。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# 每模型: (input_$_per_M, output_$_per_M, is_reasoning, note)
PRICES = {
    # OpenAI
    "gpt-3.5-turbo-instruct": (1.50, 2.00, False, "GPT-3 proxy (completion-style)"),
    "gpt-4o-mini":            (0.15, 0.60, False, ""),
    "gpt-4o":                 (2.50, 10.00, False, ""),
    "gpt-4.1-nano":           (0.10, 0.40, False, "scale-small"),
    "gpt-4.1-mini":           (0.40, 1.60, False, "scale-mid"),
    "gpt-4.1":                (2.00, 8.00, False, "scale-large"),
    "gpt-5-nano":             (0.05, 0.40, False, "GPT-5 small (est)"),
    "gpt-5-mini":             (0.25, 2.00, False, "GPT-5 mid (est)"),
    "gpt-5":                  (1.25, 10.00, False, "GPT-5 frontier (est)"),
    "o3-mini":                (1.10, 4.40, True, "reasoning"),
    "o4-mini":                (1.10, 4.40, True, "reasoning"),
    # DeepSeek (off-peak, USD; on-peak ~50% higher) - reference only, already run
    "deepseek-chat":          (0.07, 0.27, False, "(already done, listed for ref)"),
    "deepseek-reasoner":      (0.07, 0.55, True, "(already done, listed for ref)"),
}


# 任务: items per model + per-item avg in/out tokens (chat / reasoning)
# 由 DeepSeek 实测得到
TASKS = {
    "digit_matrix":  {"n": 314, "in_avg": 150, "out_chat": 30,  "out_reasoning": 1400},
    "letter_string": {"n": 224, "in_avg": 90,  "out_chat": 30,  "out_reasoning": 1400},
    "verbal_analogy":{"n": 80,  "in_avg": 60,  "out_chat": 8,   "out_reasoning": 400},
}


def estimate_one(model: str, in_per_m: float, out_per_m: float, reasoning: bool):
    total_in = 0; total_out = 0; total_cost = 0.0
    rows = []
    for task, t in TASKS.items():
        n = t["n"]
        in_tok = n * t["in_avg"]
        out_per_item = t["out_reasoning"] if reasoning else t["out_chat"]
        out_tok = n * out_per_item
        cost = in_tok * in_per_m / 1e6 + out_tok * out_per_m / 1e6
        rows.append((task, n, in_tok, out_tok, cost))
        total_in += in_tok; total_out += out_tok; total_cost += cost
    return rows, total_in, total_out, total_cost


def main():
    print(f"{'model':<25} {'task':<16} {'N':>4} {'in_tok':>9} {'out_tok':>9} {'cost_$':>9}")
    print("-" * 80)
    grand_total = 0.0
    for model, (in_p, out_p, rsn, note) in PRICES.items():
        rows, ti, to_, tc = estimate_one(model, in_p, out_p, rsn)
        for task, n, in_t, out_t, c in rows:
            print(f"{model:<25} {task:<16} {n:>4} {in_t:>9,} {out_t:>9,} {c:>9.4f}")
        print(f"{model:<25} {'TOTAL':<16} {'':>4} {ti:>9,} {to_:>9,} {tc:>9.4f}  ({note})")
        print()
        if model not in ("deepseek-chat", "deepseek-reasoner"):
            grand_total += tc
    print("-" * 80)
    print(f"{'GRAND TOTAL (excluding already-done DeepSeek)':<60} ${grand_total:.4f}")
    print()
    # 时长粗估:chat ~1s/item, reasoning ~10s/item, concurrency=4 → /4
    print("[time, concurrency=4]")
    for model, (in_p, out_p, rsn, note) in PRICES.items():
        n_items = sum(t["n"] for t in TASKS.values())
        per_item_s = 10.0 if rsn else 1.0
        wall_s = n_items * per_item_s / 4
        print(f"  {model:<25} ~ {wall_s/60:.1f} min")


if __name__ == "__main__":
    main()
