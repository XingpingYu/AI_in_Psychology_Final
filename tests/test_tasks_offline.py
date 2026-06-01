"""离线(不打 API)校验:加载题集、format prompt、用'伪答案'校验解析与打分。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tasks import digit_matrix as dm
from src.tasks import letter_string as ls
from src.tasks import verbal_analogy as va


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_digit_matrix():
    items = dm.load_items(n_per_subtype=3)
    _check(len(items) > 0, "no digit_matrix items loaded")
    sample = items[0]
    prompt = dm.format_prompt(sample)
    _check("[?]" in prompt, "prompt missing [?] placeholder")
    # 用正确答案回填 + 任意噪声前缀,看解析能不能恢复
    correct = sample.payload["correct_answer"]
    fake_response = (f"After analysis, the answer is "
                     f"[{ ' '.join(str(x) for x in correct) }]")
    parsed = dm.parse_answer(fake_response, sample)
    _check(not parsed.unparseable, "parse failed on golden")
    sc = dm.score(parsed, sample)
    _check(sc["correct"], f"score failed on golden answer for {sample.subtype}")
    # 错答
    bad_resp = "the answer is [9 9 9 9 9 9 9 9]"
    bad_parsed = dm.parse_answer(bad_resp, sample)
    bad_sc = dm.score(bad_parsed, sample)
    _check(bad_sc["correct"] is False, "should not be correct on garbage")

    # 逻辑题(permutation invariant)
    logic_items = [it for it in items if it.payload["perm_invariant"]]
    if logic_items:
        lit = logic_items[0]
        correct = lit.payload["correct_answer"]
        if len(correct) > 1:
            shuffled = list(reversed(correct))
            resp = f"[{ ' '.join(str(x) for x in shuffled) }]"
            ps = dm.score(dm.parse_answer(resp, lit), lit)
            _check(ps["correct"], "perm_invariant: reversed answer should still be correct")
    print("  digit_matrix OK")


def test_letter_string():
    items = ls.load_items(n_per_subtype=2)
    _check(len(items) > 0, "no letter_string items loaded")
    sample = items[0]
    prompt = ls.format_prompt(sample)
    _check("[?]" in prompt, "prompt missing [?] placeholder")
    tgt = sample.payload["tgt_B"]
    resp = f"[{ ' '.join(tgt) }]"
    parsed = ls.parse_answer(resp, sample)
    sc = ls.score(parsed, sample)
    _check(sc["correct"], "letter_string golden failed")
    # 错答
    bad = ls.score(ls.parse_answer("[z z z]", sample), sample)
    _check(bad["correct"] is False, "wrong answer should not be correct")
    # noprompt variant
    p2 = ls.format_prompt(sample, variant="noprompt")
    _check("Let's try" not in p2, "noprompt should drop intro")
    print("  letter_string OK")


def test_verbal_analogy():
    items = va.load_items()
    _check(len(items) > 0, "no UCLA VAT items loaded")
    sample = items[0]
    prompt = va.format_prompt(sample)
    correct_letter = sample.payload["_correct_letter"]
    resp = f"{correct_letter}"
    sc = va.score(va.parse_answer(resp, sample), sample)
    _check(sc["correct"], "verbal golden failed")
    wrong_letter = "2" if correct_letter == "1" else "1"
    bad_sc = va.score(va.parse_answer(wrong_letter, sample), sample)
    _check(bad_sc["correct"] is False, "verbal wrong should be wrong")
    print("  verbal_analogy OK")


if __name__ == "__main__":
    test_digit_matrix()
    test_letter_string()
    test_verbal_analogy()
    print("All offline task tests passed.")
