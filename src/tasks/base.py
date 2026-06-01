"""任务接口。
每个任务模块对外暴露:
  - load_items(...) -> list[TaskItem]  题目集
  - format_prompt(item) -> str         构造 chat 题面(prompt 协议下)
  - parse_answer(raw_text, item) -> ParsedAnswer
  - score(parsed, item) -> bool / 分数
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskItem:
    """一道题。item_id 必须在该 task 内唯一,用于断点续跑。"""
    item_id: str
    task: str
    subtype: str                  # 用于分组统计的子类型(如 'row_constant' / 'far')
    payload: dict[str, Any] = field(default_factory=dict)   # 任务相关原始数据
    meta: dict[str, Any] = field(default_factory=dict)      # 附加(N_unique_rules 等)


@dataclass
class ParsedAnswer:
    parsed: Any                                    # 抽取出来的答案(list[int] / str / int)
    raw: str                                       # 模型原始文本
    unparseable: bool = False
    note: str = ""
