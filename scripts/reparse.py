"""离线重新解析 raw jsonl(不重新调用 API)。
两个用途:
  1) 当 response_text 为空(reasoner 超长 thinking 被截断)时,从 reasoning_text 末尾
     回退抽取答案。
  2) 改进 parser 后重新打分所有题目,无需重花 API 钱。

用法:
  python scripts/reparse.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_experiment, load_models
from src.utils.io import jsonl_read

from src.tasks import digit_matrix as t_dm
from src.tasks import letter_string as t_ls
from src.tasks import verbal_analogy as t_va
from src.tasks.base import TaskItem


TASK_PARSERS = {
    "digit_matrix": (t_dm.parse_answer, t_dm.score, t_dm.load_items),
    "letter_string": (t_ls.parse_answer, t_ls.score, t_ls.load_items),
    "verbal_analogy": (t_va.parse_answer, t_va.score, t_va.load_items),
}


def reconstruct_item(task: str, rec: dict) -> TaskItem | None:
    """从 raw jsonl 一条记录里重建一个最小可评分 TaskItem。
    我们只需要 payload 中评分所需字段(correct_answer / tgt_B / D / D_distractor)。
    """
    if task == "digit_matrix":
        return TaskItem(
            item_id=rec["item_id"], task=task, subtype=rec["subtype"],
            payload={
                "correct_answer": rec.get("correct_answer", []),
                "perm_invariant": str(rec.get("subtype", "")).endswith("_permuted")
                                  or rec.get("subtype") in {"c1_set_union", "c2_set_union",
                                                            "c3_set_union", "AND", "XOR"},
            },
        )
    if task == "letter_string":
        # 需要 tgt_B,从原题库里反查
        # 简化:从 item_id 反查
        items = _ls_lookup()
        return items.get(rec["item_id"])
    if task == "verbal_analogy":
        items = _va_lookup()
        return items.get(rec["item_id"])
    return None


_LS_CACHE: dict[str, TaskItem] | None = None
_VA_CACHE: dict[str, TaskItem] | None = None


def _ls_lookup() -> dict[str, TaskItem]:
    global _LS_CACHE
    if _LS_CACHE is None:
        _LS_CACHE = {it.item_id: it for it in t_ls.load_items(n_per_subtype=None)}
    return _LS_CACHE


def _va_lookup() -> dict[str, TaskItem]:
    global _VA_CACHE
    if _VA_CACHE is None:
        items = t_va.load_items(n_max=None)
        # 为了让 score 知道正确选项,先调一次 format_prompt 让 _correct_letter 写入
        for it in items:
            t_va.format_prompt(it)
        _VA_CACHE = {it.item_id: it for it in items}
    return _VA_CACHE


def recover_from_reasoning(rec: dict, task: str) -> str | None:
    """response_text 为空 / unparseable 时,从 reasoning_text 末尾尝试找一个候选答案。"""
    rtxt = rec.get("reasoning_text") or ""
    if not rtxt:
        return None
    tail = rtxt[-1500:]  # 看末尾 1500 字符,通常结论在末尾
    # 找最后一个 [ ... ] 形式的答案
    import re
    matches = re.findall(r"\[([^\[\]]{1,80})\]", tail)
    if matches:
        cand = "[" + matches[-1] + "]"
        return cand
    return None


def main():
    exp = load_experiment()
    raw_dir = ROOT / exp["paths"]["raw_results_dir"]
    n_files = 0
    n_recovered = 0
    n_changed_score = 0
    n_total = 0
    for model_dir in raw_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for jf in model_dir.glob("*.jsonl"):
            task = jf.stem
            if task not in TASK_PARSERS:
                continue
            parse_fn, score_fn, _ = TASK_PARSERS[task]
            recs = jsonl_read(jf)
            new_recs = []
            for r in recs:
                n_total += 1
                item = reconstruct_item(task, r)
                if item is None:
                    new_recs.append(r)
                    continue
                resp_text = r.get("response_text") or ""
                used_fallback = False
                # 1. 主解析
                pa = parse_fn(resp_text, item)
                # 2. 若 unparseable + 有 reasoning_text,尝试从 reasoning 末尾恢复
                if pa.unparseable:
                    cand = recover_from_reasoning(r, task)
                    if cand:
                        pa2 = parse_fn(cand, item)
                        if not pa2.unparseable:
                            pa = pa2
                            used_fallback = True
                            n_recovered += 1
                sc = score_fn(pa, item)
                # 更新记录
                old_correct = r.get("correct", False)
                old_unparseable = r.get("unparseable", True)
                r["correct"] = sc["correct"]
                r["unparseable"] = pa.unparseable
                r["parse_note"] = (pa.note or "") + (" [recovered_from_reasoning]"
                                                     if used_fallback else "")
                if r["correct"] != old_correct or r["unparseable"] != old_unparseable:
                    n_changed_score += 1
                new_recs.append(r)
            # 覆写
            with jf.open("w", encoding="utf-8") as f:
                for r in new_recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_files += 1
    print(f"Reparsed {n_files} files, {n_total} records.")
    print(f"  recovered from reasoning_text: {n_recovered}")
    print(f"  records whose correct/unparseable changed: {n_changed_score}")


if __name__ == "__main__":
    main()
