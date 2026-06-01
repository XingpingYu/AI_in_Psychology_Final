"""根据 models.yaml 的 entry 构造对应 client。
评测主循环只通过 build_client(entry) 拿 client,不感知 provider。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseLLMClient, FatalClientError
from .openai_client import OpenAICompatClient


_SUPPORTED = {"openai", "deepseek", "openrouter", "dashscope"}


def list_supported_providers() -> set[str]:
    return set(_SUPPORTED)


def build_client(entry: dict[str, Any],
                 cache_dir: str | Path | None = None,
                 call_cfg: dict[str, Any] | None = None) -> BaseLLMClient:
    provider = entry["provider"]
    common = dict(
        temperature=entry.get("temperature", 0.0),
        max_tokens=entry.get("max_tokens", 1024),
        cache_dir=cache_dir,
        is_reasoning=entry.get("reasoning", False),
    )
    if call_cfg:
        common.update(dict(
            max_retries=call_cfg.get("max_retries", 5),
            base_backoff_s=call_cfg.get("base_backoff_s", 2.0),
            request_timeout_s=call_cfg.get("request_timeout_s", 90.0),
            rate_limit_qps=call_cfg.get("rate_limit_qps"),
        ))

    if provider == "openai":
        import os
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        return OpenAICompatClient(
            model=entry["model"], provider="openai",
            api_key_env="OPENAI_API_KEY", base_url=base_url,
            completion_style=entry.get("completion_style", False),
            **common,
        )
    if provider == "deepseek":
        return OpenAICompatClient(
            model=entry["model"], provider="deepseek",
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            **common,
        )
    if provider == "openrouter":
        # OpenRouter 推荐设置 HTTP-Referer / X-Title 便于他们分流量
        headers = {
            "HTTP-Referer": "https://github.com/local/emergent-analogies-repro",
            "X-Title": "emergent-analogies-repro",
        }
        return OpenAICompatClient(
            model=entry["model"], provider="openrouter",
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers, **common,
        )
    if provider == "dashscope":
        # 阿里云 DashScope(OpenAI 兼容模式)
        return OpenAICompatClient(
            model=entry["model"], provider="dashscope",
            api_key_env="QWEN_API_KEY",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            **common,
        )
    raise FatalClientError(f"provider '{provider}' not yet supported. "
                           f"Supported: {sorted(_SUPPORTED)}")
