"""OpenAI / DeepSeek 适配器(都是 OpenAI 兼容 API)。
DeepSeek 仅需把 base_url 指向 https://api.deepseek.com 即可复用 openai SDK。
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI, BadRequestError, AuthenticationError, RateLimitError, APIError

from .base import BaseLLMClient, LLMResponse, ClientError, FatalClientError


# DeepSeek 的 reasoner 系列在 chat.completions 返回中带 reasoning_content 字段
# (官方 API 文档说明)。这里抽出来填进 reasoning_text。
def _extract_reasoning(message: Any) -> str | None:
    val = getattr(message, "reasoning_content", None)
    if val:
        return val
    # 有些 provider 把 reasoning 放进 model_dump
    try:
        d = message.model_dump() if hasattr(message, "model_dump") else {}
        if d.get("reasoning_content"):
            return d["reasoning_content"]
    except Exception:
        pass
    return None


class OpenAICompatClient(BaseLLMClient):
    """适用于 OpenAI 和 DeepSeek 等 OpenAI 兼容接口。"""

    def __init__(self,
                 model: str,
                 provider: str = "openai",
                 api_key_env: str = "OPENAI_API_KEY",
                 base_url: str | None = None,
                 system_prompt: str | None = None,
                 completion_style: bool = False,
                 default_headers: dict[str, str] | None = None,
                 **kwargs: Any) -> None:
        super().__init__(model=model, provider=provider, **kwargs)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise FatalClientError(f"missing env {api_key_env}")
        client_kwargs: dict[str, Any] = dict(api_key=api_key, base_url=base_url,
                                              timeout=self.request_timeout_s)
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self._client = OpenAI(**client_kwargs)
        self.system_prompt = system_prompt
        # completion_style=True 走 /v1/completions (gpt-3.5-turbo-instruct, davinci-002),否则走 /v1/chat/completions
        self.completion_style = completion_style

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _generate(self, prompt: str, **params: Any) -> LLMResponse:
        temperature = params.get("temperature", self.temperature)
        max_tokens = params.get("max_tokens", self.max_tokens)
        # reasoning 模型常常忽略/拒绝 temperature=0
        if self.is_reasoning and temperature == 0.0:
            temperature = 1.0

        if self.completion_style:
            return self._generate_completion(prompt, temperature, max_tokens)
        return self._generate_chat(prompt, temperature, max_tokens)

    def _generate_chat(self, prompt: str, temperature: float, max_tokens: int) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=self._build_messages(prompt),
        )
        # o-series (o1/o3/o4) 和 gpt-5 系列只接受 temperature=1 (默认),
        # 显式传非默认值会 400。这里在 model name 里嗅探,简单稳健。
        model_lc = self.model.lower()
        force_default_temp = (model_lc.startswith(("o1", "o3", "o4"))
                              or model_lc.startswith("gpt-5"))
        if not force_default_temp:
            kwargs["temperature"] = temperature
        # o-series 强制用 max_completion_tokens 替代 max_tokens
        if model_lc.startswith(("o1", "o3", "o4")) or model_lc.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except AuthenticationError as e:
            raise FatalClientError(f"auth_error: {e}") from e
        except BadRequestError as e:
            raise FatalClientError(f"bad_request: {e}") from e
        except (RateLimitError, APIError) as e:
            raise ClientError(f"transient_api_error: {e}") from e

        choice = resp.choices[0]
        text = choice.message.content or ""
        reasoning_text = _extract_reasoning(choice.message)

        usage: dict[str, Any] = {}
        if resp.usage is not None:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            # o-series 含 reasoning_tokens 子项
            ctd = getattr(resp.usage, "completion_tokens_details", None)
            if ctd is not None:
                rt = getattr(ctd, "reasoning_tokens", None)
                if rt is not None:
                    usage["reasoning_tokens"] = rt

        return LLMResponse(
            text=text.strip(),
            raw={"finish_reason": choice.finish_reason, "id": resp.id},
            reasoning_text=reasoning_text,
            usage=usage,
        )

    def _generate_completion(self, prompt: str, temperature: float,
                             max_tokens: int) -> LLMResponse:
        """走 legacy /v1/completions(只对 gpt-3.5-turbo-instruct / davinci-002 用)。"""
        try:
            resp = self._client.completions.create(
                model=self.model, prompt=prompt,
                temperature=temperature, max_tokens=max_tokens,
            )
        except AuthenticationError as e:
            raise FatalClientError(f"auth_error: {e}") from e
        except BadRequestError as e:
            raise FatalClientError(f"bad_request: {e}") from e
        except (RateLimitError, APIError) as e:
            raise ClientError(f"transient_api_error: {e}") from e
        ch = resp.choices[0]
        text = (ch.text or "").strip()
        usage: dict[str, Any] = {}
        if resp.usage is not None:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return LLMResponse(text=text,
                           raw={"finish_reason": ch.finish_reason, "id": resp.id},
                           usage=usage)

    # ------- 可选: logprob 协议 -------
    def logprob_of_completion(self, prompt: str, completion: str, **params: Any) -> float:
        """OpenAI chat 接口对 logprobs 的支持有限,且不能针对'强制 completion'打分。
        要做严格的多选 logprob 评测,需要用 legacy completions(text-davinci-003 等),
        或本地 vLLM/HF。这里保留接口但默认抛出,提示评测层退回 prompt 协议。
        """
        raise NotImplementedError(
            "OpenAICompatClient does not implement true completion logprob scoring. "
            "Use 'prompt' protocol for chat-style models."
        )
