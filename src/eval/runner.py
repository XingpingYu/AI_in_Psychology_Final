"""通用评测驱动:对一个 (client, task) 组合跑题集,支持:
- 断点续跑(读 raw jsonl 的已完成 item_id 跳过)
- 单条调用失败时记录 error,不阻塞批次
- 元数据(模型/参数/温度/时间)落盘,便于事后审计
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from ..clients.base import BaseLLMClient
from ..tasks.base import TaskItem, ParsedAnswer
from ..utils.io import jsonl_append, completed_item_ids, raw_results_path


# jsonl_append 在并发场景下需要互斥
_WRITE_LOCK = threading.Lock()


@dataclass
class TaskSpec:
    name: str                                                     # e.g. 'digit_matrix'
    items: list[TaskItem]
    format_prompt: Callable[[TaskItem], str]
    parse_answer: Callable[[str, TaskItem], ParsedAnswer]
    score: Callable[[ParsedAnswer, TaskItem], dict]
    max_tokens: int = 256
    prompt_variant: str | None = None    # 给 letter_string 等任务的 prompt 变体
    # 若不为空且 client.completion_style=True,会改用 completion-style format
    format_prompt_completion: Callable[[TaskItem], str] | None = None


def _one_item(client: BaseLLMClient, task: TaskSpec, item: TaskItem,
              raw_path: Path) -> tuple[bool, bool]:
    """跑单条题目并落盘。返回 (success, failed)."""
    # completion-style 模型(gpt-3.5-turbo-instruct / davinci-002):
    # 如果 TaskSpec 提供了 format_prompt_completion,优先用它(原仓库格式)。
    use_completion = getattr(client, "completion_style", False) and \
                     task.format_prompt_completion is not None
    if use_completion:
        prompt = task.format_prompt_completion(item)
    elif task.prompt_variant is None:
        prompt = task.format_prompt(item)
    else:
        prompt = task.format_prompt(item, variant=task.prompt_variant)
    record: dict[str, Any] = {
        "item_id": item.item_id,
        "task": task.name,
        "subtype": item.subtype,
        "model": client.model,
        "provider": client.provider,
        "timestamp": time.time(),
        "prompt": prompt,
        "payload_meta": item.meta,
        "correct_answer": _safe_serializable(item.payload.get("correct_answer")),
    }
    success = False
    failed = False
    try:
        effective_max = max(task.max_tokens, client.max_tokens)
        resp = client.generate(prompt, max_tokens=effective_max)
        parsed = task.parse_answer(resp.text, item)
        sc = task.score(parsed, item)
        record.update({
            "response_text": resp.text,
            "reasoning_text": resp.reasoning_text,
            "elapsed_s": resp.elapsed_s,
            "usage": resp.usage,
            "parsed": _safe_serializable(parsed.parsed),
            "unparseable": parsed.unparseable,
            "parse_note": parsed.note,
            "correct": sc.get("correct", False),
        })
        success = True
    except Exception as e:
        record.update({
            "error": repr(e),
            "traceback": traceback.format_exc()[-1200:],
            "correct": False,
            "unparseable": True,
        })
        failed = True
    with _WRITE_LOCK:
        jsonl_append(raw_path, record)
    return success, failed


def run_task(client: BaseLLMClient,
             task: TaskSpec,
             raw_dir: Path,
             model_id: str,
             progress_desc: str = "",
             dry_run: bool = False,
             concurrency: int = 1) -> dict[str, Any]:
    """执行评测;支持并发。返回汇总 dict(亦写入 raw jsonl)。"""
    raw_path = raw_results_path(raw_dir, model_id, task.name)
    done_ids = completed_item_ids(raw_path)

    n_total = len(task.items)
    n_skip = sum(1 for it in task.items if it.item_id in done_ids)
    pending = [it for it in task.items if it.item_id not in done_ids]

    print(f"  task={task.name} model={model_id}: {n_total} items "
          f"({n_skip} cached, {len(pending)} pending, concurrency={concurrency})")

    if dry_run:
        print("  [dry_run] skip API calls.")
        return {"task": task.name, "model": model_id,
                "n_total": n_total, "n_pending": len(pending),
                "dry_run": True}

    successes = 0
    failures = 0
    desc = progress_desc or f"{model_id}/{task.name}"
    if concurrency <= 1:
        for it in tqdm(pending, desc=desc, ncols=100, leave=False):
            s, f = _one_item(client, task, it, raw_path)
            successes += int(s)
            failures += int(f)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_one_item, client, task, it, raw_path) for it in pending]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc=desc, ncols=100, leave=False):
                s, f = fut.result()
                successes += int(s)
                failures += int(f)

    return {
        "task": task.name,
        "model": model_id,
        "n_total": n_total,
        "n_done": n_total - len(pending) + successes,
        "n_failed": failures,
        "raw_path": str(raw_path),
    }


def _safe_serializable(v: Any) -> Any:
    """递归把 numpy 类型转成 python 原生,便于 json 序列化。"""
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return [_safe_serializable(x) for x in v]
    if isinstance(v, dict):
        return {k: _safe_serializable(x) for k, x in v.items()}
    return v
