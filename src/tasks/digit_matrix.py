"""Digit Matrices 任务(原论文 Exp 1 主任务)。

- 数据源: data/repo_original/digit_mat/all_problems.npz
- 32 个子类型:
    transformation 类(perm_invariant=False, 答案顺序也要对):
       row_constant, col_constant, dist3_diag1, dist3_diag2,
       prog_size1, prog_size2,
       two_rule_comb0..5, three_rule_comb0..9
    logic 类(perm_invariant=True, 集合一致即可):
       c1/c2/c3_set_union (OR), AND, XOR, 及对应 _permuted 变体
- 评测: 生成式 - 模型补全最后一格 [..],解析数字序列。
- chat 协议: 把矩阵以原论文相同的 '[a b c] [d e f] [g h i]\n' 格式呈现,
  最后一行末尾留空,要求模型只输出最后一格 '[...]' 内的数字序列。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .base import TaskItem, ParsedAnswer
from ..utils.config import project_root


REPO_DIR = project_root() / "data" / "repo_original" / "digit_mat"
DATA_DEFAULT = REPO_DIR / "all_problems.npz"


# 论文中区分 problem class:
TRANS_TYPES = {
    "row_constant", "col_constant", "dist3_diag1", "dist3_diag2",
    "prog_size1", "prog_size2",
    "two_rule_comb0", "two_rule_comb1", "two_rule_comb2",
    "two_rule_comb3", "two_rule_comb4", "two_rule_comb5",
    "three_rule_comb0", "three_rule_comb1", "three_rule_comb2",
    "three_rule_comb3", "three_rule_comb4", "three_rule_comb5",
    "three_rule_comb6", "three_rule_comb7", "three_rule_comb8",
    "three_rule_comb9",
}
LOGIC_TYPES = {
    "c1_set_union", "c2_set_union", "c3_set_union",
    "AND", "XOR",
    "c1_set_union_permuted", "c2_set_union_permuted", "c3_set_union_permuted",
    "AND_permuted", "XOR_permuted",
}


def problem_class(subtype: str) -> str:
    if subtype in LOGIC_TYPES:
        return "logic"
    if subtype.startswith("one_") or subtype in {"row_constant", "col_constant",
                                                 "dist3_diag1", "dist3_diag2",
                                                 "prog_size1", "prog_size2"}:
        return "one_rule"
    if subtype.startswith("two_"):
        return "two_rule"
    if subtype.startswith("three_"):
        return "three_rule"
    return "other"


def has_progression(subtype: str) -> bool:
    return subtype.startswith("prog_")


# ---------- 加载 ----------
def load_items(path: str | Path = DATA_DEFAULT,
               n_per_subtype: int | None = None,
               include_subtypes: Iterable[str] | None = None,
               rng_seed: int = 42) -> list[TaskItem]:
    """按子类型加载题目。
    n_per_subtype=None 取全部;否则取前 n 道(保持原顺序以保证可复现)。
    """
    rng = np.random.default_rng(rng_seed)  # noqa: F841 (reserved for future sampling)
    blob = np.load(path, allow_pickle=True)
    all_prob = blob["all_problems"].item()

    items: list[TaskItem] = []
    for subtype, body in all_prob.items():
        if include_subtypes is not None and subtype not in include_subtypes:
            continue
        prob_arr = body["prob"]
        ans_arr = body["answer_choices"]
        correct_idx_arr = body["correct_ind"]
        perm_inv = bool(body["perm_invariant"])
        n_total = len(prob_arr)
        n_take = n_total if n_per_subtype is None else min(n_per_subtype, n_total)
        for i in range(n_take):
            prob = prob_arr[i]
            answer_choices = ans_arr[i]
            correct_ind = int(correct_idx_arr[i])
            correct_answer = answer_choices[correct_ind]
            # 转 python list 以便 json 序列化
            prob_list = _to_listish(prob)
            ans_list = [_to_listish(a) for a in answer_choices]
            corr_list = _to_listish(correct_answer)
            items.append(TaskItem(
                item_id=f"{subtype}__{i:04d}",
                task="digit_matrix",
                subtype=subtype,
                payload={
                    "prob": prob_list,            # 3x3, 最后一格不展示
                    "answer_choices": ans_list,
                    "correct_ind": correct_ind,
                    "correct_answer": corr_list,
                    "perm_invariant": perm_inv,
                },
                meta={
                    "problem_class": problem_class(subtype),
                    "has_progression": has_progression(subtype),
                },
            ))
    return items


def _to_listish(arr) -> Any:
    """递归把 numpy 数组转成普通 list,便于 json/jsonl 落盘。"""
    if isinstance(arr, np.ndarray):
        return [_to_listish(x) for x in arr.tolist()] if arr.ndim > 1 else arr.tolist()
    if isinstance(arr, (list, tuple)):
        return [_to_listish(x) for x in arr]
    return arr


# ---------- prompt 构造 ----------
def _format_cell(cell: Any) -> str:
    """单元格 -> '[a b c]' 字符串。逻辑题里 cell 是 list[int],
    transformation 题里 cell 是 list[int](rules 数 = 内层长度)。"""
    # cell 可能是 [int] 或 [[int,...],...] 取决于 task layout
    # 原仓库的统一处理是 flatten 后输出
    flat = _flatten(cell)
    items = []
    for v in flat:
        if v == -1:
            items.append(" ")    # 论文中的空白占位
        else:
            items.append(str(int(v)))
    return "[" + " ".join(items) + "]"


def _flatten(x: Any) -> list:
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [x]


def format_prompt(item: TaskItem) -> str:
    """chat-style prompt: 给前 8 格,要求只输出最后一格 [...]."""
    prob = item.payload["prob"]
    lines = []
    for r in range(3):
        row_cells = []
        for c in range(3):
            if r == 2 and c == 2:
                row_cells.append("[?]")
            else:
                row_cells.append(_format_cell(prob[r][c]))
        lines.append(" ".join(row_cells))
    matrix_str = "\n".join(lines)

    prompt = (
        "You are solving a 3x3 matrix reasoning problem. "
        "Each cell of the matrix is a set of digits enclosed in square brackets. "
        "The bottom-right cell is missing (marked as [?]). "
        "Find the cell that completes the matrix following the same pattern as the other rows/columns.\n\n"
        f"{matrix_str}\n\n"
        "Reply with ONLY the missing cell on a single line in the format [d1 d2 ...]. "
        "Do not include any explanation."
    )
    return prompt


def format_prompt_completion(item: TaskItem) -> str:
    """Completion-style prompt(为 gpt-3.5-turbo-instruct / davinci-002 等设计)。
    完全模仿原仓库 eval_gpt_matprob.py:仅给 8 个 cell,最后一格以开放 `[` 结尾,
    让模型自然补全 `d1 d2 ... ]`。没有任何指令文本。
    """
    prob = item.payload["prob"]
    out = ""
    for r in range(3):
        for c in range(3):
            out += "["
            if not (r == 2 and c == 2):
                cell = prob[r][c]
                flat = _flatten(cell)
                tokens = [(" " if v == -1 else str(int(v))) for v in flat]
                out += " ".join(tokens) + "]"
                if c < 2:
                    out += " "
                else:
                    out += "\n"
    return out


# ---------- 答案解析 ----------
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
_DIGIT_RE = re.compile(r"-?\d+")


def parse_answer(text: str, item: TaskItem) -> ParsedAnswer:
    """从模型输出抽出数字序列。优先取**最后一个**方括号内容
    (兼顾模型先复述再答的情形);若没有方括号,从全文末尾抽数字。"""
    raw = text.strip()
    expected_len = len(item.payload["correct_answer"])

    # 1) 找方括号
    matches = _BRACKET_RE.findall(raw)
    candidate_str = None
    if matches:
        # 取最后一个非空匹配
        for m in reversed(matches):
            if any(ch.isdigit() for ch in m):
                candidate_str = m
                break
    # 2) 否则取最后一行的数字
    if candidate_str is None:
        last_line = raw.splitlines()[-1] if raw else ""
        if any(ch.isdigit() for ch in last_line):
            candidate_str = last_line

    if candidate_str is None:
        return ParsedAnswer(parsed=None, raw=raw, unparseable=True,
                            note="no_digits_found")

    nums = [int(s) for s in _DIGIT_RE.findall(candidate_str)]
    if not nums:
        return ParsedAnswer(parsed=None, raw=raw, unparseable=True,
                            note="bracket_no_digits")
    return ParsedAnswer(parsed=nums, raw=raw, unparseable=False,
                        note=f"expected_len={expected_len}")


def score(parsed: ParsedAnswer, item: TaskItem) -> dict:
    """返回 {'correct': bool, 'unparseable': bool}.
    transformation: 序列+顺序都要对
    logic (perm_invariant): 集合一致即可
    """
    out = {"correct": False, "unparseable": parsed.unparseable}
    if parsed.unparseable or parsed.parsed is None:
        return out
    correct = item.payload["correct_answer"]
    pred = list(parsed.parsed)
    if item.payload["perm_invariant"]:
        out["correct"] = sorted(pred) == sorted(correct)
    else:
        out["correct"] = pred == list(correct)
    return out
