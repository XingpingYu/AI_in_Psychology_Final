"""Phase 0 自检脚本

功能:
  1) 加载 .env 并报告各 key 是否存在(不打印任何 key 内容)。
  2) 通过 OpenAI API 列出账户可用模型(取前若干个 'gpt' / 'o' 系列)。
  3) DeepSeek 没有标准 list models;直接对 deepseek-chat 跑 1 token 探活。
  4) 对每个 enabled 模型跑一次 "hello world" 类比题,验证 client 封装可用。
     (会消耗极少 token,温度=0,max_tokens 设置极小)

运行:
    python scripts/phase0_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能 import src/
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.utils.env import load_env
from src.utils.config import load_models, load_experiment, get_active_models
from src.utils.logging_setup import get_logger
from src.clients import build_client


def list_openai_models(top_k: int = 30) -> list[str]:
    """直接调 OpenAI 的 /models 端点,把账户可见的模型按 id 排序打印若干个。"""
    from openai import OpenAI, AuthenticationError, APIError
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []
    cli = OpenAI(api_key=api_key, timeout=30)
    try:
        page = cli.models.list()
    except (AuthenticationError, APIError) as e:
        print(f"  [warn] list openai models failed: {e!r}")
        return []
    ids = sorted([m.id for m in page.data])
    # 只挑跟本项目相关的前缀,避免被嵌入/语音模型刷屏
    keep = [i for i in ids if any(i.startswith(p) for p in
            ("gpt-", "o1", "o3", "o4", "chatgpt-", "text-davinci"))]
    return keep[:top_k]


HELLO_PROMPT = "Complete the analogy. Answer with one word only.\n\nhot : cold :: tall : ?"


def hello_world_one(entry: dict, cache_dir: Path, call_cfg: dict) -> dict:
    out = {"id": entry["id"], "provider": entry["provider"], "model": entry["model"]}
    try:
        client = build_client(entry, cache_dir=cache_dir, call_cfg=call_cfg)
    except Exception as e:
        out.update(status="build_fail", err=repr(e))
        return out
    try:
        # 用 8 token 上限把成本压到最低;reasoning 模型给 256 兜底
        max_t = 256 if entry.get("reasoning") else 16
        resp = client.generate(HELLO_PROMPT, max_tokens=max_t, use_cache=True)
        out.update(
            status="ok",
            text=resp.text[:80],
            has_reasoning=bool(resp.reasoning_text),
            tokens=resp.usage,
            elapsed_s=round(resp.elapsed_s, 2),
        )
    except Exception as e:
        out.update(status="call_fail", err=repr(e))
    return out


def main() -> int:
    log = get_logger("analogy.phase0", logfile=ROOT / "logs" / "phase0.log")
    log.info("== Phase 0 probe ==")

    # 1) env
    print("\n[1] Env check")
    env_status = load_env(verbose=True)

    # 2) OpenAI list models
    print("\n[2] OpenAI models visible to your account (filtered to gpt-/o*/davinci):")
    if env_status.get("OPENAI_API_KEY"):
        ids = list_openai_models()
        if ids:
            for i in ids:
                print(f"  - {i}")
        else:
            print("  (list returned empty)")
    else:
        print("  (skip - no OPENAI_API_KEY)")

    # 3) configs
    print("\n[3] Active models (from config/models.yaml):")
    exp = load_experiment()
    active = get_active_models()
    for m in active:
        print(f"  - id={m['id']:20s} provider={m['provider']:10s} model={m['model']}")

    # 4) hello world calls
    print("\n[4] Hello-world calls (1 prompt each, low token):")
    cache_dir = ROOT / exp["paths"]["cache_dir"]
    call_cfg = exp.get("call", {})
    print(f"  (cache_dir={cache_dir})")
    results = []
    for m in active:
        r = hello_world_one(m, cache_dir=cache_dir, call_cfg=call_cfg)
        results.append(r)
        if r["status"] == "ok":
            extra = "[+reasoning]" if r.get("has_reasoning") else ""
            tok = r.get("tokens", {})
            print(f"  [OK] {r['id']:20s} -> '{r['text']!s}'  "
                  f"tokens={tok.get('total_tokens','?')} {r['elapsed_s']}s {extra}")
        else:
            print(f"  [FAIL] {r['id']:20s} status={r['status']} err={r.get('err','')[:160]}")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nResult: {n_ok}/{len(results)} models passed hello-world.")
    return 0 if n_ok == len(results) and results else 1


if __name__ == "__main__":
    sys.exit(main())
