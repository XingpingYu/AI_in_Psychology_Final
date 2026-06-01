"""把 raw jsonl 汇总成长表 DataFrame,后续 stats/plot 都基于它。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils.io import jsonl_read


def load_all_raw(raw_dir: Path, tasks: Iterable[str] | None = None) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(Path(raw_dir).iterdir()):
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name
        for jf in sorted(model_dir.glob("*.jsonl")):
            task = jf.stem
            if tasks is not None and task not in tasks:
                continue
            for r in jsonl_read(jf):
                tokens = (r.get("usage") or {})
                rows.append({
                    "model": model_id,
                    "task": r.get("task", task),
                    "subtype": r.get("subtype"),
                    "item_id": r.get("item_id"),
                    "correct": bool(r.get("correct", False)),
                    "unparseable": bool(r.get("unparseable", False)),
                    "elapsed_s": r.get("elapsed_s", 0.0),
                    "prompt_tokens": tokens.get("prompt_tokens", 0),
                    "completion_tokens": tokens.get("completion_tokens", 0),
                    "total_tokens": tokens.get("total_tokens", 0),
                    "has_error": "error" in r,
                    "reasoning_text_len": len(r.get("reasoning_text") or ""),
                    "payload_meta": r.get("payload_meta", {}),
                })
    return pd.DataFrame(rows)


def attach_problem_class(df: pd.DataFrame) -> pd.DataFrame:
    """给 digit_matrix 加 problem_class 列(one/two/three_rule/logic)。"""
    from ..tasks.digit_matrix import problem_class, has_progression
    is_dm = df["task"] == "digit_matrix"
    df = df.copy()
    df.loc[is_dm, "problem_class"] = df.loc[is_dm, "subtype"].apply(problem_class)
    df.loc[is_dm, "has_progression"] = df.loc[is_dm, "subtype"].apply(has_progression)
    df.loc[is_dm, "is_permuted"] = df.loc[is_dm, "subtype"].str.endswith("_permuted")
    return df
