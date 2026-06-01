"""主评测入口。
按 config/experiment.yaml + config/models.yaml,对每个 enabled 模型
依次跑指定 task,raw 结果写 jsonl,跑完汇总到 csv。

Usage:
  python scripts/run_eval.py --tasks digit_matrix              # 单任务
  python scripts/run_eval.py --tasks digit_matrix letter_string verbal_analogy
  python scripts/run_eval.py --tasks digit_matrix --models deepseek-chat
  python scripts/run_eval.py --tasks all
  python scripts/run_eval.py --tasks all --dry-run             # 不打 API,只看预算

ENV:
  EXP_MODE=full python scripts/run_eval.py ...   # 覆盖 yaml 里的 mode
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import load_env
from src.utils.config import (load_models, load_experiment, get_active_models,
                              task_sample_size, project_root)
from src.utils.logging_setup import get_logger
from src.utils.io import ensure_dir, jsonl_read
from src.clients import build_client
from src.eval.runner import TaskSpec, run_task

from src.tasks import digit_matrix as t_dm
from src.tasks import letter_string as t_ls
from src.tasks import verbal_analogy as t_va
from src.tasks import story_analogy as t_st


TASK_BUILDERS = {
    "digit_matrix":    "_build_dm",
    "letter_string":   "_build_ls",
    "verbal_ucla_vat":  "_build_vat",
    "verbal_sternberg": "_build_sternberg",
    "verbal_kmiecik":   "_build_kmiecik",
    "verbal_jurgens":   "_build_jurgens",
    "story_analogy":   "_build_story",
}


def _build_dm(exp: dict, mode: str) -> TaskSpec:
    n = task_sample_size(exp, "digit_matrix", "n_per_subtype")
    items = t_dm.load_items(n_per_subtype=n)
    return TaskSpec(
        name="digit_matrix",
        items=items,
        format_prompt=t_dm.format_prompt,
        format_prompt_completion=t_dm.format_prompt_completion,  # gpt-3.5-instruct 用
        parse_answer=t_dm.parse_answer,
        score=t_dm.score,
        max_tokens=64,
    )


def _build_ls(exp: dict, mode: str) -> TaskSpec:
    n = task_sample_size(exp, "letter_string", "n_per_cell")
    items = t_ls.load_items(n_per_subtype=n)
    return TaskSpec(
        name="letter_string",
        items=items,
        format_prompt=t_ls.format_prompt,
        format_prompt_completion=t_ls.format_prompt_completion,
        parse_answer=t_ls.parse_answer,
        score=t_ls.score,
        max_tokens=64,
    )


def _build_vat(exp: dict, mode: str) -> TaskSpec:
    n = task_sample_size(exp, "verbal_ucla_vat", "n_max")
    items = t_va.load_items(dataset="ucla_vat", n_max=n)
    return TaskSpec(name="verbal_ucla_vat", items=items,
                    format_prompt=t_va.format_prompt,
                    parse_answer=t_va.parse_answer,
                    score=t_va.score, max_tokens=16)


def _build_sternberg(exp: dict, mode: str) -> TaskSpec:
    n = task_sample_size(exp, "verbal_sternberg", "n_max")
    items = t_va.load_items(dataset="sternberg", n_max=n)
    return TaskSpec(name="verbal_sternberg", items=items,
                    format_prompt=t_va.format_prompt,
                    parse_answer=t_va.parse_answer,
                    score=t_va.score, max_tokens=16)


def _build_kmiecik(exp: dict, mode: str) -> TaskSpec:
    n = task_sample_size(exp, "verbal_kmiecik", "n_max")
    items = t_va.load_items(dataset="kmiecik", n_max=n,
                            stratify=True, seed=42)
    return TaskSpec(name="verbal_kmiecik", items=items,
                    format_prompt=t_va.format_prompt,
                    parse_answer=t_va.parse_answer,
                    score=t_va.score, max_tokens=8)


def _build_jurgens(exp: dict, mode: str) -> TaskSpec:
    # placeholder (本次禁用)
    raise NotImplementedError("verbal_jurgens disabled in this run")


def _build_story(exp: dict, mode: str) -> TaskSpec:
    n = task_sample_size(exp, "story_analogy", "n_sets")
    items = t_st.load_items(n_sets=n)
    return TaskSpec(name="story_analogy", items=items,
                    format_prompt=t_st.format_prompt,
                    parse_answer=t_st.parse_answer,
                    score=t_st.score, max_tokens=64)


def estimate_calls(specs: list[TaskSpec], models: list[dict]) -> dict:
    """估计调用数量(没考虑断点续跑里的 already done)。"""
    total = 0
    by_task = {}
    for s in specs:
        n_items = len(s.items)
        n_models = len(models)
        total += n_items * n_models
        by_task[s.name] = n_items
    return {"total_calls": total, "by_task": by_task,
            "n_models": len(models)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["all"],
                    help="task names or 'all'")
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these model ids (default: all enabled)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan but do not call APIs")
    ap.add_argument("--mode", default=None, choices=["smoke", "full"],
                    help="override experiment.mode")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="concurrent API calls per model (default 4)")
    args = ap.parse_args()

    log = get_logger("analogy.run", logfile=ROOT / "logs" / "run_eval.log")
    load_env(verbose=False)

    exp = load_experiment()
    if args.mode:
        exp["mode"] = args.mode
    elif os.environ.get("EXP_MODE"):
        exp["mode"] = os.environ["EXP_MODE"]
    mode = exp["mode"]
    log.info(f"mode={mode}")

    # build task specs
    if "all" in args.tasks:
        tasks_to_run = list(TASK_BUILDERS.keys())
    else:
        tasks_to_run = list(args.tasks)
    specs: list[TaskSpec] = []
    for tname in tasks_to_run:
        if tname not in TASK_BUILDERS:
            print(f"[warn] unknown task: {tname}, skipped")
            continue
        builder = globals()[TASK_BUILDERS[tname]]
        s = builder(exp, mode)
        specs.append(s)
        print(f"  built task {tname}: {len(s.items)} items")

    # select models
    all_models = get_active_models()
    if args.models:
        sel = [m for m in all_models if m["id"] in args.models]
        missing = set(args.models) - {m["id"] for m in sel}
        if missing:
            print(f"[warn] models not enabled/found: {missing}")
        models = sel
    else:
        models = all_models
    if not models:
        print("no models selected. abort.")
        return 2

    # estimate
    est = estimate_calls(specs, models)
    print(f"\nEstimated raw calls (no resume): {est['total_calls']} "
          f"({est['n_models']} models x tasks)")
    for t, n in est["by_task"].items():
        print(f"  - {t}: {n} items x {est['n_models']} models = "
              f"{n * est['n_models']}")
    if args.dry_run:
        print("DRY RUN: not calling APIs.")
        return 0

    # run
    raw_dir = ROOT / exp["paths"]["raw_results_dir"]
    cache_dir = ROOT / exp["paths"]["cache_dir"]
    ensure_dir(raw_dir); ensure_dir(cache_dir)
    call_cfg = exp.get("call", {})
    summary_rows = []
    t0 = time.time()
    for m in models:
        print(f"\n=== Model: {m['id']}  (provider={m['provider']}) ===")
        try:
            client = build_client(m, cache_dir=cache_dir, call_cfg=call_cfg)
        except Exception as e:
            print(f"  [skip] build_client failed: {e!r}")
            continue
        for s in specs:
            result = run_task(client, s, raw_dir=raw_dir, model_id=m["id"],
                              concurrency=args.concurrency)
            summary_rows.append(result)
    dt = time.time() - t0
    print(f"\nAll done in {dt:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
