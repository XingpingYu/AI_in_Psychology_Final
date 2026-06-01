"""Story Analogies 任务(对齐 Webb 2023 + Gentner, Rattermann & Forbus 1993)。

数据源:`data/AnalogyInventory/Public/Cognitive Psychology.xlsx`
里的 `Rattermann` sheet,18 个故事集 × 5 个变体:
   Base / Literally similar / True Analogy / False Analogy / Mere-Appearance / New Mere-Appearance

评测设计(照搬 `data/repo_original/story_analogies/eval_GPT3_story_analogies.py`):
- analogy condition (analogy_vs_similarity=1):
    True analogy vs False analogy  → correct = True analogy
    每个 source 跑 2 trial(A/B 顺序交换)
- similarity condition (analogy_vs_similarity=0):
    Literal similarity vs Mere appearance → correct = Literal similarity
    每个 source 跑 2 trial

因此每个 source 总共 4 trial,18 个 source → 72 题/模型。

每 trial 的 prompt(完全照原文字面):
```
Consider the following story:

Story 1: <source>

Now consider two more stories:

Story A: <option_A>

Story B: <option_B>

Which of Story A and Story B is a better analogy to Story 1?
Is the best answer Story A, Story B, or both are equally analogous?
```

评分:模型回答里抽出 'A' / 'B' / 'both'。chance 取 50%(把 'both' 视作非正确)。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .base import TaskItem, ParsedAnswer
from ..utils.config import project_root


XLSX_DEFAULT = project_root() / "data" / "AnalogyInventory" / "Public" / "Cognitive Psychology.xlsx"


def _load_rattermann_sets(path: Path = XLSX_DEFAULT) -> list[dict[str, str]]:
    """读取 Rattermann sheet 的 18 个 story set。
    原 eval 脚本里用 [1:19] 跳过第 0 行(列说明),即第 1-18 行为 18 个 set。
    """
    df = pd.read_excel(path, sheet_name="Rattermann")
    sets = []
    # 数据行从 idx=1 起(idx=0 是 schema 行说明);跑到 idx=18(含)
    for i in range(1, 19):
        if i >= len(df):
            break
        row = df.iloc[i]
        try:
            sets.append({
                "id": int(row["Set #"]),
                "base": str(row["Base"]).strip(),
                "literal": str(row["Literally similar story"]).strip(),
                "true_analogy": str(row["True Analogy Story"]).strip(),
                "false_analogy": str(row["False Analogy Story"]).strip(),
                "mere_appearance": str(row["Mere-Appearance Match"]).strip(),
            })
        except Exception:
            continue
    return sets


def load_items(path: Path = XLSX_DEFAULT,
               n_sets: int | None = None) -> list[TaskItem]:
    """生成 18 × 4 = 72 trial(分 analogy 与 similarity 两个条件,各 2 trial 控位置)。"""
    sets = _load_rattermann_sets(path)
    if n_sets is not None:
        sets = sets[:n_sets]
    items: list[TaskItem] = []
    for s in sets:
        sid = s["id"]
        # ----- analogy condition (analogy_vs_similarity=1):
        # True analogy vs False analogy
        items.append(TaskItem(
            item_id=f"story__set{sid:02d}__analogy_A",
            task="story_analogy",
            subtype="analogy",
            payload={
                "source": s["base"],
                "option_A": s["true_analogy"],
                "option_B": s["false_analogy"],
                "correct": "A",
                "condition": "analogy",   # near/structural
                "set_id": sid,
            },
            meta={"position_swap": False},
        ))
        items.append(TaskItem(
            item_id=f"story__set{sid:02d}__analogy_B",
            task="story_analogy",
            subtype="analogy",
            payload={
                "source": s["base"],
                "option_A": s["false_analogy"],
                "option_B": s["true_analogy"],
                "correct": "B",
                "condition": "analogy",
                "set_id": sid,
            },
            meta={"position_swap": True},
        ))
        # ----- similarity condition (analogy_vs_similarity=0):
        # Literal similarity vs Mere appearance
        items.append(TaskItem(
            item_id=f"story__set{sid:02d}__similarity_A",
            task="story_analogy",
            subtype="similarity",
            payload={
                "source": s["base"],
                "option_A": s["literal"],
                "option_B": s["mere_appearance"],
                "correct": "A",
                "condition": "similarity",
                "set_id": sid,
            },
            meta={"position_swap": False},
        ))
        items.append(TaskItem(
            item_id=f"story__set{sid:02d}__similarity_B",
            task="story_analogy",
            subtype="similarity",
            payload={
                "source": s["base"],
                "option_A": s["mere_appearance"],
                "option_B": s["literal"],
                "correct": "B",
                "condition": "similarity",
                "set_id": sid,
            },
            meta={"position_swap": True},
        ))
    return items


def format_prompt(item: TaskItem) -> str:
    p = item.payload
    prompt = (
        "Consider the following story:\n\n"
        f"Story 1: {p['source']}\n\n"
        "Now consider two more stories:\n\n"
        f"Story A: {p['option_A']}\n\n"
        f"Story B: {p['option_B']}\n\n"
        "Which of Story A and Story B is a better analogy to Story 1? "
        "Is the best answer Story A, Story B, or both are equally analogous?"
    )
    return prompt


_CHOICE_RE = re.compile(r"\b(?:story\s+)?([ab])\b", re.IGNORECASE)
_BOTH_RE = re.compile(r"\bboth\b|\bequally\b", re.IGNORECASE)


def parse_answer(text: str, item: TaskItem) -> ParsedAnswer:
    raw = (text or "").strip()
    if not raw:
        return ParsedAnswer(parsed=None, raw=raw, unparseable=True, note="empty")
    # 优先看是否出现"Story A" / "Story B" 字样,后退到首字母
    if _BOTH_RE.search(raw):
        return ParsedAnswer(parsed="both", raw=raw, unparseable=False, note="both/equal")
    m = _CHOICE_RE.search(raw)
    if not m:
        return ParsedAnswer(parsed=None, raw=raw, unparseable=True, note="no_AB_found")
    return ParsedAnswer(parsed=m.group(1).upper(), raw=raw, unparseable=False)


def score(parsed: ParsedAnswer, item: TaskItem) -> dict:
    out = {"correct": False, "unparseable": parsed.unparseable}
    if parsed.unparseable or parsed.parsed is None:
        return out
    # 把 'both' 视作没明确选,按 chance=50% 处理 → 算错(原 paper 也是这么处理)
    if parsed.parsed == "both":
        out["correct"] = False
        out["both"] = True
        return out
    out["correct"] = (str(parsed.parsed).upper() == str(item.payload["correct"]).upper())
    return out
