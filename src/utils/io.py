"""结果落盘与断点续跑工具。
约定:每个 (model_id, task, item_id) 的原始响应单独一行 jsonl,
重跑时按 item_id 去重跳过。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def jsonl_append(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    # 使用 ensure_ascii=False 保留中文便于人工核查
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def jsonl_read(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def completed_item_ids(path: str | Path) -> set[str]:
    """读取 jsonl 中已完成的 item_id 集合用于断点续跑。"""
    return {r["item_id"] for r in jsonl_read(path) if "item_id" in r}


def raw_results_path(raw_dir: str | Path, model_id: str, task: str) -> Path:
    raw_dir = Path(raw_dir)
    p = raw_dir / model_id / f"{task}.jsonl"
    ensure_dir(p.parent)
    return p


def hash_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
