"""Letter String Analogies 任务(原论文 Exp 2)。

数据: data/repo_original/letter_string/all_prob.npz
- 28 个子类型,每个含 prob[t] = [[src_A, tgt_A], [src_B, tgt_B]]
- 'trans' 字段记录 transformation 类型,'gen' 字段记录 generalization 级别。
- 评测: 生成式,模型补 tgt_B,精确匹配。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .base import TaskItem, ParsedAnswer
from ..utils.config import project_root


REPO_DIR = project_root() / "data" / "repo_original" / "letter_string"
DATA_DEFAULT = REPO_DIR / "all_prob.npz"


def load_items(path: str | Path = DATA_DEFAULT,
               n_per_subtype: int | None = None,
               include_subtypes: Iterable[str] | None = None) -> list[TaskItem]:
    blob = np.load(path, allow_pickle=True)["all_prob"].item()
    items: list[TaskItem] = []
    for subtype, body in blob.items():
        if include_subtypes is not None and subtype not in include_subtypes:
            continue
        prob_arr = body["prob"]                  # list of [[src_A,tgt_A],[src_B,tgt_B]]
        trans_arr = body.get("trans", [subtype] * len(prob_arr))
        gen_arr = body.get("gen", [[]] * len(prob_arr))
        n_total = len(prob_arr)
        n_take = n_total if n_per_subtype is None else min(n_per_subtype, n_total)
        for t in range(n_take):
            ex = prob_arr[t]
            src_A = list(ex[0][0])
            tgt_A = list(ex[0][1])
            src_B = list(ex[1][0])
            tgt_B = list(ex[1][1])
            this_trans = trans_arr[t] if t < len(trans_arr) else subtype
            this_gen = list(gen_arr[t]) if t < len(gen_arr) and gen_arr[t] is not None else []
            items.append(TaskItem(
                item_id=f"{subtype}__{t:03d}",
                task="letter_string",
                subtype=subtype,
                payload={
                    "src_A": src_A, "tgt_A": tgt_A,
                    "src_B": src_B, "tgt_B": tgt_B,
                },
                meta={"trans": str(this_trans),
                      "gen": this_gen,                          # list
                      "gen_level": len(this_gen)},
            ))
    return items


# ------- prompt -------
def _fmt(seq: list) -> str:
    return "[" + " ".join(str(x) for x in seq) + "]"


def _join(seq) -> str:
    return " ".join(str(x) for x in seq)


def format_prompt_completion(item: TaskItem) -> str:
    """Completion-style prompt(为 gpt-3.5-turbo-instruct 设计):
    给 src_A, tgt_A, src_B 后跟开放 `[`,让模型补 `t1 t2 ... ]`。
    完全模仿原仓库 eval_GPT3_letterstring_prob.py 的 default 格式(去掉指令)。
    某些 generalization(letter↔num)的元素是 numpy.int64,要 str(),否则 join 报错。
    """
    p = item.payload
    s = lambda lst: " ".join(str(x) for x in lst)
    return f"[{s(p['src_A'])}] [{s(p['tgt_A'])}]\n[{s(p['src_B'])}] ["


def format_prompt(item: TaskItem, variant: str = "default") -> str:
    """
    variant:
      - default: 原论文的两行格式 + "Let's try to complete the pattern:" prompt
      - sentence: '改写为 If A changes to B, then C should change to ?'
      - noprompt: 不带任何引导句
    """
    p = item.payload
    if variant == "sentence":
        prompt = ("If " + _join(p["src_A"]) + " changes to " + _join(p["tgt_A"])
                  + ", then " + _join(p["src_B"]) + " should change to ?\n"
                  "Reply with ONLY the answer as space-separated tokens in square brackets, e.g. [a b c]. "
                  "No explanation.")
        return prompt
    if variant == "noprompt":
        header = ""
    else:
        header = "Let's try to complete the pattern:\n\n"
    body = (f"{_fmt(p['src_A'])} {_fmt(p['tgt_A'])}\n"
            f"{_fmt(p['src_B'])} [?]")
    tail = ("\n\nReply with ONLY the missing target on a single line in the same "
            "[t1 t2 ...] format. Do not include any explanation.")
    return header + body + tail


# ------- 解析 -------
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9\-]+")   # 字母 / 数字 / 短横(real-world variants 可能有)


def parse_answer(text: str, item: TaskItem) -> ParsedAnswer:
    raw = text.strip()
    matches = _BRACKET_RE.findall(raw)
    candidate = None
    for m in reversed(matches):
        if m.strip() and "?" not in m:
            candidate = m
            break
    if candidate is None:
        # 退而求其次: 最后一行
        last = raw.splitlines()[-1] if raw else ""
        candidate = last.strip()
    tokens = _TOKEN_RE.findall(candidate)
    if not tokens:
        return ParsedAnswer(parsed=None, raw=raw, unparseable=True, note="no_tokens")
    tokens = [t.lower() for t in tokens]
    return ParsedAnswer(parsed=tokens, raw=raw, unparseable=False)


def score(parsed: ParsedAnswer, item: TaskItem) -> dict:
    out = {"correct": False, "unparseable": parsed.unparseable}
    if parsed.unparseable or parsed.parsed is None:
        return out
    target = [str(t).lower() for t in item.payload["tgt_B"]]
    out["correct"] = list(parsed.parsed) == target
    return out
