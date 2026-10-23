"""Thin, dependency-free clients for vision-language model APIs.

A model is named ``provider:model_id``, e.g. ``openai:gpt-4o-mini``,
``anthropic:claude-haiku-4-5`` or ``gemini:gemini-2.5-flash``. API keys
come from the environment: OPENAI_API_KEY, ANTHROPIC_API_KEY,
GEMINI_API_KEY (or GOOGLE_API_KEY).

``mock:oracle`` and ``mock:random`` need no key and are used for tests.
"""
from __future__ import annotations

import base64
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

# Sampling temperature. Unset (default) = the provider's recommended default.
# Thinking models are prone to endless repetition loops under greedy decoding
# (temperature 0), and providers advise against it, so we use the default.
TEMPERATURE = os.environ.get("NVB_TEMPERATURE")
TIMEOUT = 600  # thinking models can take several minutes on hard items
# Generous budget: some models "think" before answering and the hidden
# thinking counts against this limit. 16k tokens avoids cutting them off.
MAX_TOKENS = int(os.environ.get("NVB_MAX_TOKENS", "16384"))


@dataclass
class Reply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    finish: str = ""          # provider's stop reason (e.g. STOP, MAX_TOKENS)
    thinking_tokens: int = 0  # hidden reasoning tokens, when reported


class APIError(RuntimeError):
    def __init__(self, msg: str, retryable: bool):
        super().__init__(msg)
        self.retryable = retryable


def _b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _post(url, headers, body):
    t0 = time.time()
    try:
        r = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
    except requests.RequestException as e:  # network hiccup
        raise APIError(str(e), retryable=True) from e
    if r.status_code in (408, 409, 429) or r.status_code >= 500:
        raise APIError(f"HTTP {r.status_code}: {r.text[:300]}", retryable=True)
    if r.status_code != 200:
        raise APIError(f"HTTP {r.status_code}: {r.text[:500]}", retryable=False)
    return r.json(), time.time() - t0


class RateLimiter:
    """Spaces out request starts so a process stays under a requests-per-minute
    cap (free API tiers allow e.g. 30 RPM). Set NVB_RPM=0 to disable."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.next_t = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.time()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        time.sleep(max(0.0, t - now))


LIMITER = RateLimiter(float(os.environ.get("NVB_RPM", "0")))


class Model:
    name: str

    def generate(self, system: str, prompt: str, image: str | None = None) -> Reply:
        raise NotImplementedError

    def __call__(self, system, prompt, image=None, retries: int = 10) -> Reply:
        delay = 5.0
        for attempt in range(retries):
            LIMITER.wait()
            try:
                return self.generate(system, prompt, image)
            except APIError as e:
                if not e.retryable or attempt == retries - 1:
                    raise
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")


class OpenAIModel(Model):
    """OpenAI Chat Completions API (works for GPT-4o/4.1/5 families)."""

    def __init__(self, model_id: str):
        self.name, self.model_id = f"openai:{model_id}", model_id
        self.key = os.environ["OPENAI_API_KEY"]
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        # reasoning models (o-series, gpt-5) reject a custom temperature
        self.fixed_temp = not (model_id.startswith("o") or model_id.startswith("gpt-5"))

    def generate(self, system, prompt, image=None):
        content = [{"type": "text", "text": prompt}]
        if image:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_b64(image)}",
                                          "detail": "high"}})
        body = {"model": self.model_id, "max_completion_tokens": MAX_TOKENS,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": content}]}
        if self.fixed_temp and TEMPERATURE is not None:
            body["temperature"] = float(TEMPERATURE)
        js, dt = _post(f"{self.base}/chat/completions",
                       {"Authorization": f"Bearer {self.key}"}, body)
        u = js.get("usage", {})
        ch = js["choices"][0]
        return Reply(ch["message"].get("content") or "", u.get("prompt_tokens", 0),
                     u.get("completion_tokens", 0), dt, ch.get("finish_reason", ""),
                     u.get("completion_tokens_details", {}).get("reasoning_tokens", 0))


class AnthropicModel(Model):
    """Anthropic Messages API."""

    def __init__(self, model_id: str):
        self.name, self.model_id = f"anthropic:{model_id}", model_id
        self.key = os.environ["ANTHROPIC_API_KEY"]

    def generate(self, system, prompt, image=None):
        content = []
        if image:
            content.append({"type": "image", "source": {"type": "base64",
                                                        "media_type": "image/png",
                                                        "data": _b64(image)}})
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model_id, "max_tokens": MAX_TOKENS,
                "system": system, "messages": [{"role": "user", "content": content}]}
        if TEMPERATURE is not None:
            body["temperature"] = float(TEMPERATURE)
        js, dt = _post("https://api.anthropic.com/v1/messages",
                       {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, body)
        text = "".join(b.get("text", "") for b in js.get("content", []) if b["type"] == "text")
        u = js.get("usage", {})
        return Reply(text, u.get("input_tokens", 0), u.get("output_tokens", 0), dt,
                     js.get("stop_reason", ""))


class GeminiModel(Model):
    """Google Gemini generateContent API."""

    def __init__(self, model_id: str):
        self.name, self.model_id = f"gemini:{model_id}", model_id
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]

    def generate(self, system, prompt, image=None):
        parts = [{"text": prompt}]
        if image:
            parts.append({"inline_data": {"mime_type": "image/png", "data": _b64(image)}})
        body = {"system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"maxOutputTokens": MAX_TOKENS}}
        if TEMPERATURE is not None:
            body["generationConfig"]["temperature"] = float(TEMPERATURE)
        # Streaming keeps bytes flowing while the model "thinks" for minutes;
        # a single long silent request gets cut by idle-connection timeouts.
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model_id}:streamGenerateContent?alt=sse")
        text, finish, u, dt = _stream_gemini(url, self.key, body)
        return Reply(text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0), dt,
                     finish, u.get("thoughtsTokenCount", 0))


def _stream_gemini(url, key, body):
    t0 = time.time()
    try:
        r = requests.post(url, headers={"x-goog-api-key": key}, json=body, stream=True,
                          timeout=(30, 240))  # 240 s max silence between chunks
    except requests.RequestException as e:
        raise APIError(str(e), retryable=True) from e
    if r.status_code in (408, 409, 429) or r.status_code >= 500:
        raise APIError(f"HTTP {r.status_code}: {r.text[:300]}", retryable=True)
    if r.status_code != 200:
        raise APIError(f"HTTP {r.status_code}: {r.text[:500]}", retryable=False)
    text, finish, usage = [], "", {}
    try:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            for c in chunk.get("candidates") or []:
                for p in c.get("content", {}).get("parts", []):
                    if not p.get("thought"):
                        text.append(p.get("text", ""))
                finish = c.get("finishReason", finish)
            usage = chunk.get("usageMetadata", usage)
    except (requests.RequestException, ValueError) as e:
        raise APIError(f"stream broken: {e}", retryable=True) from e
    if not finish:  # stream ended without a final chunk: treat as transient
        raise APIError("stream ended early", retryable=True)
    return "".join(text), finish, usage, time.time() - t0


class MockModel(Model):
    """Offline stand-in used to test the pipeline end to end.

    ``mock:random`` guesses; ``mock:oracle`` is handled in evaluate.py,
    which answers with the symbolic oracle instead of calling a model.
    """

    def __init__(self, kind: str):
        self.name, self.kind = f"mock:{kind}", kind

    def generate(self, system, prompt, image=None):
        return Reply(f"ANSWER: {random.choice(['A', 'B', '3', 'C', 'H1', '12'])}")


PROVIDERS = {"openai": OpenAIModel, "anthropic": AnthropicModel, "gemini": GeminiModel,
             "mock": MockModel}


def load_model(spec: str) -> Model:
    provider, _, model_id = spec.partition(":")
    if provider not in PROVIDERS or not model_id:
        raise ValueError(f"model must look like provider:model_id, got {spec!r}")
    return PROVIDERS[provider](model_id)
