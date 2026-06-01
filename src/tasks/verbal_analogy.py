"""Four-term Verbal Analogies。

数据集:
  - UCLA VAT (80 题): `data/repo_original/UCLA_VAT/UCLA_VAT.xlsx`
    论文 Webb 2023 主用数据集之一,与原文有逐题 GPT-3 / 人类参照
  - Sternberg & Nigro 1980 (197 题):`data/AnalogyInventory/Public/Cognitive Psychology.xlsx`
    Webb 2023 引用同一来源,但原文未释出 per-item GPT-3 结果,只能聚合对照
  - Kmiecik (≈ Jones et al. 2022)(720 题,Webb 引用):同上 xlsx,Kmiecik sheet
    格式不同:**判断 A:B :: C:D 是否为有效类比**(Yes / No)

所有数据集的题目内容由 `dataset` 参数指定;不同数据集复用同样的 format/parse/score 逻辑。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .base import TaskItem, ParsedAnswer
from ..utils.config import project_root


REPO_DIR = project_root() / "data" / "repo_original" / "UCLA_VAT"
DATA_DEFAULT = REPO_DIR / "UCLA_VAT.xlsx"
INVENTORY = project_root() / "data" / "AnalogyInventory" / "Public" / "Cognitive Psychology.xlsx"


def load_items(dataset: str = "ucla_vat",
               n_max: int | None = None,
               **kwargs) -> list[TaskItem]:
    """根据 dataset 选 loader。
    dataset 取值:
      - 'ucla_vat'  → UCLA_VAT.xlsx (2-AFC: D vs D')
      - 'sternberg' → Sternberg sheet  (2-AFC, 同 UCLA_VAT 格式)
      - 'kmiecik'  → Kmiecik sheet (yes/no judgment),
                     额外 kwarg: stratify=True/False, seed=int
    """
    if dataset == "ucla_vat":
        return _load_ucla_vat(n_max=n_max)
    if dataset == "sternberg":
        return _load_sternberg(n_max=n_max)
    if dataset == "kmiecik":
        return _load_kmiecik(n_max=n_max, **kwargs)
    raise ValueError(f"unknown dataset: {dataset}")


def _load_ucla_vat(n_max: int | None = None,
                   path: str | Path = DATA_DEFAULT) -> list[TaskItem]:
    df = pd.read_excel(path, sheet_name=0)
    df["Relation"] = df["Relation"].ffill()
    df = df.dropna(subset=["A", "B", "C", "D"])
    if "D'" not in df.columns:
        for c in df.columns:
            if str(c).strip() == "D'":
                df = df.rename(columns={c: "D'"})
                break
    items: list[TaskItem] = []
    for i, row in df.iterrows():
        items.append(TaskItem(
            item_id=f"vat__{i:04d}",
            task="verbal_analogy",
            subtype=str(row["Relation"]).strip(),
            payload={
                "A": str(row["A"]).strip(),
                "B": str(row["B"]).strip(),
                "C": str(row["C"]).strip(),
                "D": str(row["D"]).strip(),
                "D_distractor": str(row["D'"]).strip(),
                "format": "2afc",
            },
            meta={"dataset": "ucla_vat"},
        ))
        if n_max is not None and len(items) >= n_max:
            break
    return items


def _load_sternberg(n_max: int | None = None,
                    path: str | Path = INVENTORY) -> list[TaskItem]:
    """Sternberg & Nigro 1980 - 197 题, 同 UCLA_VAT 格式 (D vs D')。"""
    df = pd.read_excel(path, sheet_name="Sternberg")
    df["Relation"] = df["Relation"].ffill()
    df = df.dropna(subset=["A", "B", "C", "D"]).copy()
    if "D'" not in df.columns:
        for c in df.columns:
            if str(c).strip() == "D'":
                df = df.rename(columns={c: "D'"})
                break
    items: list[TaskItem] = []
    for i, row in df.iterrows():
        items.append(TaskItem(
            item_id=f"stern__{i:04d}",
            task="verbal_analogy",
            subtype=str(row["Relation"]).strip(),
            payload={
                "A": str(row["A"]).strip(),
                "B": str(row["B"]).strip(),
                "C": str(row["C"]).strip(),
                "D": str(row["D"]).strip(),
                "D_distractor": str(row["D'"]).strip(),
                "format": "2afc",
            },
            meta={"dataset": "sternberg"},
        ))
        if n_max is not None and len(items) >= n_max:
            break
    return items


def _load_kmiecik(n_max: int | None = None,
                  stratify: bool = True,
                  seed: int = 42,
                  path: str | Path = INVENTORY) -> list[TaskItem]:
    """Kmiecik(Webb 引用为 Jones et al. 2022)— 判断 A:B :: C:D 是否为有效类比。

    数据有 4 个 stratification cells: {Near, Far} x {True, False}
    每 cell 各 180 题。stratify=True 时,从每 cell 等量抽样 n_max/4 题。
    """
    df = pd.read_excel(path, sheet_name="Kmiecik")
    df = df.dropna(subset=["A", "B", "C", "D"]).copy()
    df["T/F"] = df["T/F"].astype(str).str.strip()
    df["Near/Far"] = df["Near/Far"].astype(str).str.strip().str.lower()
    df = df[df["T/F"].isin(["True", "False"])]
    df = df[df["Near/Far"].isin(["near", "far"])]
    import numpy as np
    rng = np.random.default_rng(seed)
    if n_max is None or n_max >= len(df):
        df_sample = df
    elif stratify:
        per_cell = max(1, n_max // 4)
        chunks = []
        for tf in ["True", "False"]:
            for nf in ["near", "far"]:
                sub = df[(df["T/F"] == tf) & (df["Near/Far"] == nf)]
                take = min(per_cell, len(sub))
                pick = rng.choice(sub.index.values, size=take, replace=False)
                chunks.append(sub.loc[pick])
        df_sample = pd.concat(chunks).sort_index()
    else:
        pick = rng.choice(df.index.values, size=n_max, replace=False)
        df_sample = df.loc[sorted(pick)]
    items: list[TaskItem] = []
    for i, row in df_sample.iterrows():
        is_valid = (row["T/F"] == "True")
        items.append(TaskItem(
            item_id=f"kmie__{i:04d}",
            task="verbal_analogy",
            subtype=f"kmie_{row['Near/Far']}_{'T' if is_valid else 'F'}",
            payload={
                "A": str(row["A"]).strip(),
                "B": str(row["B"]).strip(),
                "C": str(row["C"]).strip(),
                "D": str(row["D"]).strip(),
                "label": "yes" if is_valid else "no",
                "format": "yesno",
            },
            meta={"dataset": "kmiecik", "near_far": str(row["Near/Far"]),
                  "relation_ab": str(row.get("relation AB", "")),
                  "relation_cd": str(row.get("relation CD", ""))},
        ))
    return items


def format_prompt(item: TaskItem) -> str:
    fmt = item.payload.get("format", "2afc")
    if fmt == "2afc":
        return _format_2afc(item)
    if fmt == "yesno":
        return _format_yesno(item)
    raise ValueError(f"unknown format: {fmt}")


def _format_2afc(item: TaskItem) -> str:
    """两选一(D 正确 vs D' 干扰)。位置按 item_id 末位奇偶轮换以消位置偏置。"""
    A, B, C, D, Dp = (item.payload["A"], item.payload["B"], item.payload["C"],
                      item.payload["D"], item.payload["D_distractor"])
    flip = int(item.item_id.split("__")[-1]) % 2 == 1
    if flip:
        opt1, opt2, correct = Dp, D, "2"
    else:
        opt1, opt2, correct = D, Dp, "1"
    item.payload["_correct_letter"] = correct
    return (
        "Complete the verbal analogy by choosing the best option.\n\n"
        f"{A} : {B} :: {C} : ?\n\n"
        f"(1) {opt1}\n(2) {opt2}\n\n"
        "Reply with ONLY the number 1 or 2. Do not include any explanation."
    )


def _format_yesno(item: TaskItem) -> str:
    """Kmiecik 风格:判断 A:B :: C:D 是否为合理类比。"""
    A, B, C, D = (item.payload["A"], item.payload["B"], item.payload["C"], item.payload["D"])
    return (
        "Decide whether the following is a valid analogy:\n\n"
        f"{A} : {B} :: {C} : {D}\n\n"
        "Reply with ONLY 'yes' or 'no'. Do not include any explanation."
    )


_YES_RE = re.compile(r"\b(yes|true|valid)\b", re.IGNORECASE)
_NO_RE  = re.compile(r"\b(no|false|invalid)\b", re.IGNORECASE)


def parse_answer(text: str, item: TaskItem) -> ParsedAnswer:
    raw = (text or "").strip()
    fmt = item.payload.get("format", "2afc")
    if fmt == "yesno":
        if _YES_RE.search(raw) and not _NO_RE.search(raw):
            return ParsedAnswer(parsed="yes", raw=raw, unparseable=False)
        if _NO_RE.search(raw) and not _YES_RE.search(raw):
            return ParsedAnswer(parsed="no", raw=raw, unparseable=False)
        # 都没匹配 / 两个都匹配 → unparseable
        return ParsedAnswer(parsed=None, raw=raw, unparseable=True,
                            note="no_clear_yesno")
    # 2afc
    m = re.search(r"\(?([12])\)?", raw)
    if not m:
        return ParsedAnswer(parsed=None, raw=raw, unparseable=True, note="no_12_found")
    return ParsedAnswer(parsed=m.group(1), raw=raw, unparseable=False)


def score(parsed: ParsedAnswer, item: TaskItem) -> dict:
    out = {"correct": False, "unparseable": parsed.unparseable}
    if parsed.unparseable or parsed.parsed is None:
        return out
    fmt = item.payload.get("format", "2afc")
    if fmt == "yesno":
        out["correct"] = str(parsed.parsed).lower() == str(item.payload["label"]).lower()
        return out
    correct = item.payload.get("_correct_letter", "1")
    out["correct"] = str(parsed.parsed) == str(correct)
    return out
