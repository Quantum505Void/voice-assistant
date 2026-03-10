"""LLM module - Ollama + OpenAI-compatible + Anthropic streaming client."""
from __future__ import annotations
import json
from typing import Callable
import httpx


class LLMClient:
    """Supports backends:
      - ollama:    POST /api/chat  (Ollama native)
      - openai:    POST /chat/completions  (OpenAI-compatible)
      - anthropic: POST /messages  (Anthropic Messages API)
    """

    def __init__(self, base_url: str, model: str, system_prompt: str,
                 api_key: str = "", backend: str = "ollama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.backend = backend  # "ollama" | "openai" | "anthropic"
        self.history: list[dict] = []
        self.system_prompt = system_prompt

    def _build_messages(self, user_text: str) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-20:])
        messages.append({"role": "user", "content": user_text})
        return messages

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            if self.backend == "anthropic":
                h["x-api-key"] = self.api_key
                h["anthropic-version"] = "2023-06-01"
            else:
                h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat_stream(self, user_text: str, on_token: Callable[[str], None] | None = None) -> str:
        messages = self._build_messages(user_text)
        if self.backend == "anthropic":
            return self._stream_anthropic(messages, user_text, on_token)
        elif self.backend == "openai":
            return self._stream_openai(messages, user_text, on_token)
        else:
            return self._stream_ollama(messages, user_text, on_token)

    def _stream_ollama(self, messages, user_text, on_token) -> str:
        full_response = ""
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True},
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("message", {}).get("content", "")
                    if token:
                        full_response += token
                        if on_token:
                            on_token(token)
                    if data.get("done"):
                        break
        self._append_history(user_text, full_response)
        return full_response

    def _stream_openai(self, messages, user_text, on_token) -> str:
        full_response = ""
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json={"model": self.model, "messages": messages, "stream": True},
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = (data.get("choices", [{}])[0]
                             .get("delta", {}).get("content", ""))
                    if token:
                        full_response += token
                        if on_token:
                            on_token(token)
        self._append_history(user_text, full_response)
        return full_response

    def _stream_anthropic(self, messages, user_text, on_token) -> str:
        """Anthropic Messages API with streaming."""
        # system 消息需单独提取
        system = self.system_prompt
        msgs = [m for m in messages if m["role"] != "system"]
        full_response = ""
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/messages",
                json={
                    "model": self.model,
                    "max_tokens": 4096,
                    "system": system,
                    "messages": msgs,
                    "stream": True,
                },
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "content_block_delta":
                        token = data.get("delta", {}).get("text", "")
                        if token:
                            full_response += token
                            if on_token:
                                on_token(token)
        self._append_history(user_text, full_response)
        return full_response

    def _append_history(self, user_text: str, response: str):
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response})

    def clear_history(self):
        self.history.clear()

    def check_connection(self) -> bool:
        try:
            if self.backend == "openai":
                r = httpx.get(f"{self.base_url}/models",
                              headers=self._headers(), timeout=5.0)
            elif self.backend == "anthropic":
                r = httpx.get(f"{self.base_url}/models",
                              headers=self._headers(), timeout=5.0)
            else:
                r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code < 500
        except Exception:
            return False



class LLMClient:
    """Supports two backends:
      - ollama: POST /api/chat  (Ollama native)
      - openai: POST /chat/completions  (OpenAI-compatible, e.g. Qwen API)
    """

    def __init__(self, base_url: str, model: str, system_prompt: str,
                 api_key: str = "", backend: str = "ollama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.backend = backend  # "ollama" | "openai"
        self.history: list[dict] = []
        self.system_prompt = system_prompt

    def _build_messages(self, user_text: str) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-20:])
        messages.append({"role": "user", "content": user_text})
        return messages

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat_stream(self, user_text: str, on_token: Callable[[str], None] | None = None) -> str:
        messages = self._build_messages(user_text)
        if self.backend == "openai":
            return self._stream_openai(messages, user_text, on_token)
        else:
            return self._stream_ollama(messages, user_text, on_token)

    def _stream_ollama(self, messages, user_text, on_token) -> str:
        full_response = ""
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True},
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("message", {}).get("content", "")
                    if token:
                        full_response += token
                        if on_token:
                            on_token(token)
                    if data.get("done"):
                        break
        self._append_history(user_text, full_response)
        return full_response

    def _stream_openai(self, messages, user_text, on_token) -> str:
        full_response = ""
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json={"model": self.model, "messages": messages, "stream": True},
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = (data.get("choices", [{}])[0]
                             .get("delta", {}).get("content", ""))
                    if token:
                        full_response += token
                        if on_token:
                            on_token(token)
        self._append_history(user_text, full_response)
        return full_response

    def _append_history(self, user_text: str, response: str):
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response})

    def clear_history(self):
        self.history.clear()

    def check_connection(self) -> bool:
        try:
            if self.backend == "openai":
                r = httpx.get(f"{self.base_url}/models",
                              headers=self._headers(), timeout=5.0)
            else:
                r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False
