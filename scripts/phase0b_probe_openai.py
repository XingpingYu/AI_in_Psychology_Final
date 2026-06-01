"""Phase 0b: 对 models.yaml 里所有 OpenAI / DeepSeek 模型逐个 hello-world,
确认它们都能用同一份 client 封装跑通。Qwen / OpenRouter 跳过(等用户加 key)。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.env import load_env
from src.utils.config import load_models, load_experiment
from src.clients import build_client


HELLO = "Complete the analogy. Reply with one word.\nhot : cold :: tall : ?"


def main():
    load_env(verbose=False)
    exp = load_experiment()
    cache_dir = ROOT / exp["paths"]["cache_dir"]
    call_cfg = exp.get("call", {})
    rows = []
    for m in load_models():
        if not m.get("enabled", False):
            continue
        if m["provider"] == "openrouter":
            import os
            if not os.environ.get("OPENROUTER_API_KEY"):
                print(f"  [skip] {m['id']:25s} no OPENROUTER_API_KEY yet")
                continue
        out = {"id": m["id"], "provider": m["provider"], "model": m["model"]}
        try:
            cli = build_client(m, cache_dir=cache_dir, call_cfg=call_cfg)
            # reasoner / gpt-5 给充足 budget
            max_t = 4096 if (m.get("reasoning") or m["model"].startswith(("o1","o3","o4","gpt-5"))) else 16
            r = cli.generate(HELLO, max_tokens=max_t, use_cache=False)
            out.update(text=r.text[:60], usage=r.usage, status="ok")
        except Exception as e:
            out.update(status="FAIL", err=repr(e)[:240])
        rows.append(out)
        if out["status"] == "ok":
            print(f"  [OK] {out['id']:25s} -> {out['text']!r}  tokens={out.get('usage',{}).get('total_tokens','?')}")
        else:
            print(f"  [FAIL] {out['id']:25s} {out['err']}")
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_total = len(rows)
    print(f"\nProbed {n_total} models, {n_ok} OK, {n_total - n_ok} failed.")


if __name__ == "__main__":
    main()
