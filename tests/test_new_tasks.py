"""离线自检:加载 + format + 用 golden 答案验证 parse/score。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tasks import digit_matrix as dm
from src.tasks import letter_string as ls
from src.tasks import verbal_analogy as va
from src.tasks import story_analogy as st


def _ok(c, m):
    if not c:
        raise AssertionError(m)


def test_completion_styles():
    item = dm.load_items(n_per_subtype=1)[0]
    p_chat = dm.format_prompt(item)
    p_compl = dm.format_prompt_completion(item)
    _ok("Reply with ONLY" in p_chat, "chat prompt missing instruction")
    _ok("Reply with ONLY" not in p_compl, "completion should have no instruction")
    _ok(p_compl.endswith("[") or p_compl.endswith("[\n"), "completion should end with open `[`")
    # letter_string completion
    li = ls.load_items(n_per_subtype=1)[0]
    p_ls_compl = ls.format_prompt_completion(li)
    _ok(p_ls_compl.endswith("["), "letter_string completion should end with open `[`")
    print("  completion-style formats OK")


def test_sternberg():
    items = va.load_items(dataset="sternberg", n_max=3)
    _ok(len(items) > 0, "no Sternberg loaded")
    p = va.format_prompt(items[0])
    _ok("(1)" in p and "(2)" in p, "Sternberg should be 2-AFC")
    # golden test
    correct = items[0].payload["_correct_letter"]
    sc = va.score(va.parse_answer(correct, items[0]), items[0])
    _ok(sc["correct"], "Sternberg golden score failed")
    print(f"  sternberg loaded {len(items)} items, scoring OK")


def test_kmiecik():
    items = va.load_items(dataset="kmiecik", n_max=12, stratify=True)
    _ok(len(items) > 0, "no Kmiecik loaded")
    p = va.format_prompt(items[0])
    _ok("yes" in p.lower() and "no" in p.lower(), "Kmiecik should be yes/no")
    # golden
    label = items[0].payload["label"]
    sc = va.score(va.parse_answer(label, items[0]), items[0])
    _ok(sc["correct"], "Kmiecik golden failed")
    # stratification
    cells = set()
    for it in items:
        cells.add((it.meta["near_far"], it.payload["label"]))
    print(f"  kmiecik loaded {len(items)} items across {len(cells)} stratified cells")


def test_story():
    items = st.load_items(n_sets=3)
    _ok(len(items) == 12, f"expected 12 trials (3 sets x 4), got {len(items)}")
    p = st.format_prompt(items[0])
    _ok("Story 1" in p and "Story A" in p and "Story B" in p, "story prompt missing parts")
    # golden
    correct = items[0].payload["correct"]
    sc = st.score(st.parse_answer(f"Story {correct}", items[0]), items[0])
    _ok(sc["correct"], "story golden score failed")
    # 'both' should be unparseable-but-parsed-as-both, count as wrong
    sc_both = st.score(st.parse_answer("both are equally analogous", items[0]), items[0])
    _ok(not sc_both["correct"], "'both' should not count as correct")
    print(f"  story loaded {len(items)} trials, 'both' handled correctly")


def test_conditions_balance():
    items = st.load_items(n_sets=18)
    by_cond = {}
    for it in items:
        by_cond.setdefault(it.payload["condition"], 0)
        by_cond[it.payload["condition"]] += 1
    _ok(by_cond.get("analogy") == 36 and by_cond.get("similarity") == 36,
        f"expected 36+36 trials, got {by_cond}")
    print(f"  story full set: {by_cond}")


if __name__ == "__main__":
    test_completion_styles()
    test_sternberg()
    test_kmiecik()
    test_story()
    test_conditions_balance()
    print("All new-task offline tests passed.")
