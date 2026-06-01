"""LLM client 抽象基类与公共数据结构。
所有 provider 适配器都返回统一的 LLMResponse 结构,
评测主循环只面向 BaseLLMClient 编写,不感知具体 provider。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt,
    wait_exponential, before_sleep_log,
)
import logging

from ..utils.io import ensure_dir, hash_key

logger = logging.getLogger("analogy.client")


class ClientError(Exception):
    """非致命错误,触发重试。"""


class FatalClientError(Exception):
    """致命错误(如 auth 失败/题目不合规),不重试。"""


@dataclass
class LLMResponse:
    text: str                          # 模型主回答文本(reasoning 模型已剥离 thinking)
    raw: dict[str, Any] = field(default_factory=dict)   # provider 原始返回的可序列化片段
    logprobs: Optional[list[dict[str, Any]]] = None     # 若支持
    reasoning_text: Optional[str] = None                # CoT/reasoning tokens 文本(若有)
    model: str = ""
    provider: str = ""
    usage: dict[str, Any] = field(default_factory=dict) # prompt/completion tokens
    elapsed_s: float = 0.0
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseLLMClient:
    """统一接口。子类只需实现 _generate / (可选) _logprobs_for_completion。

    Parameters
    ----------
    model : str
        Provider 真实模型名
    provider : str
        provider 名(openai / deepseek / ...)
    temperature, max_tokens : 默认参数,可在每次调用覆盖
    cache_dir : 若提供则按 (prompt, params) 缓存原始响应,避免重复消费 token
    rate_limit_qps : 简单的客户端 QPS 上限
    """

    def __init__(self,
                 model: str,
                 provider: str,
                 temperature: float = 0.0,
                 max_tokens: int = 1024,
                 cache_dir: str | Path | None = None,
                 rate_limit_qps: float | None = None,
                 max_retries: int = 5,
                 base_backoff_s: float = 2.0,
                 request_timeout_s: float = 90.0,
                 supports_logprobs: bool = False,
                 is_reasoning: bool = False,
                 **kwargs: Any) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            ensure_dir(self.cache_dir)
        self.rate_limit_qps = rate_limit_qps
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.request_timeout_s = request_timeout_s
        self.supports_logprobs = supports_logprobs
        self.is_reasoning = is_reasoning
        self._last_call_ts: float = 0.0

    # ---------- 子类需实现 ----------
    def _generate(self, prompt: str, **params: Any) -> LLMResponse:
        raise NotImplementedError

    def logprob_of_completion(self, prompt: str, completion: str, **params: Any) -> float:
        """给定 prompt 和候选 completion,返回 completion 的(平均)log-prob。
        多选评测的'logprob 协议'用它。不支持的 client 抛 NotImplementedError。
        """
        raise NotImplementedError("This client does not support log-probability scoring.")

    # ---------- 公共逻辑 ----------
    def _throttle(self) -> None:
        if not self.rate_limit_qps:
            return
        min_gap = 1.0 / self.rate_limit_qps
        now = time.time()
        gap = now - self._last_call_ts
        if gap < min_gap:
            time.sleep(min_gap - gap)
        self._last_call_ts = time.time()

    def _cache_path(self, prompt: str, params: dict[str, Any]) -> Path | None:
        if not self.cache_dir:
            return None
        key = hash_key(self.provider, self.model,
                       json.dumps(params, sort_keys=True, ensure_ascii=False),
                       prompt)
        return self.cache_dir / f"{self.provider}__{self.model}__{key}.json"

    def _load_cache(self, p: Path | None) -> LLMResponse | None:
        if not p or not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return LLMResponse(**data)
        except Exception as e:
            logger.warning("cache_load_fail path=%s err=%s", p, e)
            return None

    def _save_cache(self, p: Path | None, resp: LLMResponse) -> None:
        if not p:
            return
        try:
            p.write_text(json.dumps(resp.to_dict(), ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning("cache_save_fail path=%s err=%s", p, e)

    def generate(self, prompt: str, use_cache: bool = True, **overrides: Any) -> LLMResponse:
        params = self._merged_params(overrides)
        cache_p = self._cache_path(prompt, params) if use_cache else None
        cached = self._load_cache(cache_p)
        if cached is not None:
            logger.debug("cache_hit model=%s", self.model)
            return cached

        self._throttle()
        # 调用 retry 包装版本
        t0 = time.time()
        resp = self._generate_with_retry(prompt, **params)
        resp.elapsed_s = time.time() - t0
        resp.provider = self.provider
        resp.model = self.model
        self._save_cache(cache_p, resp)
        return resp

    def _merged_params(self, overrides: dict[str, Any]) -> dict[str, Any]:
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        params.update({k: v for k, v in overrides.items() if v is not None})
        return params

    # tenacity 的 decorator 不方便用 self.max_retries 等动态值,
    # 这里用一个简单的手动循环包装更直观、且能尊重 FatalClientError。
    def _generate_with_retry(self, prompt: str, **params: Any) -> LLMResponse:
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._generate(prompt, **params)
            except FatalClientError:
                raise
            except Exception as e:  # ClientError or unexpected
                last_err = e
                if attempt == self.max_retries:
                    break
                backoff = self.base_backoff_s * (2 ** (attempt - 1))
                logger.warning("call_retry model=%s attempt=%d err=%s sleep=%.1fs",
                               self.model, attempt, e, backoff)
                time.sleep(backoff)
        raise ClientError(f"all retries failed for model={self.model}: {last_err}")
