"""LLM backend adapter — Ollama native vs vLLM/OpenAI-compatible.

Two API shapes need to be supported:

  Ollama native (POST /api/chat):
    payload = {model, messages: [{role, content, images?}], stream, think,
               keep_alive, options:{...}, tools?: [...]}
    stream chunks: {"message": {"content": "..."}, "done": ..., "eval_count": ...}

  vLLM / OpenAI (POST /v1/chat/completions):
    payload = {model, messages: [{role, content}], stream, max_tokens,
               temperature, tools?: [...]}
    image content: [{type: "image_url", image_url: {url: "data:image/jpeg;base64,..."}},
                    {type: "text", text: "..."}]
    stream chunks: {"choices":[{"delta":{"content":"..."}}], "usage":{...}}

This module produces backend-specific payloads from a unified ollama-shape
input plus exposes a normalised stream-line iterator.

Switch via env `LLM_BACKEND=ollama|vllm`.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable

LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.6:35b-a3b")

VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen36-awq")


def chat_url() -> str:
    if LLM_BACKEND == "vllm":
        return f"{VLLM_HOST}/v1/chat/completions"
    return f"{OLLAMA_HOST}/api/chat"


def model_name() -> str:
    return VLLM_MODEL if LLM_BACKEND == "vllm" else OLLAMA_MODEL


def _convert_messages_for_openai(messages: list[dict]) -> list[dict]:
    """Convert Ollama-shape messages (with `images: [b64,...]`) into OpenAI
    multimodal shape with image_url + text content blocks."""
    out: list[dict] = []
    for m in messages:
        if m.get("images"):
            content_blocks: list[dict] = []
            for img_b64 in m["images"]:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
            text = m.get("content") or ""
            if text:
                content_blocks.append({"type": "text", "text": text})
            out.append({"role": m["role"], "content": content_blocks})
        else:
            out.append({k: v for k, v in m.items() if k != "images"})
    return out


def build_payload(
    messages: list[dict],
    *,
    stream: bool = False,
    options: dict | None = None,
    tools: list[dict] | None = None,
    think: bool = False,
    keep_alive: str = "30m",
) -> dict:
    options = options or {}
    if LLM_BACKEND == "vllm":
        payload: dict[str, Any] = {
            "model": VLLM_MODEL,
            "messages": _convert_messages_for_openai(messages),
            "stream": stream,
            "temperature": options.get("temperature", 0.75),
            "top_p": options.get("top_p", 0.92),
            "max_tokens": options.get("num_predict", 500),
        }
        # Disable thinking via chat-template kwargs (qwen3.6 quirk)
        if not think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            payload["tools"] = _convert_tools_for_openai(tools)
            payload["tool_choice"] = "auto"
        return payload

    # Ollama native (default)
    payload = {
        "model": OLLAMA_MODEL,
        "stream": stream,
        "think": think,
        "keep_alive": keep_alive,
        "options": options,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
    return payload


def _convert_tools_for_openai(tools: list[dict]) -> list[dict]:
    """Tool spec shapes already overlap heavily; OpenAI requires
    {type:'function', function:{name,description,parameters}}.
    Ollama may accept the same shape. This passes-through if already in OpenAI
    shape, wraps if in bare-dict shape."""
    out: list[dict] = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            out.append(t)
        else:
            out.append({"type": "function", "function": t})
    return out


def parse_nonstream_message(data: dict) -> dict:
    """Extract assistant message + token usage from a non-stream response.

    Returns: {"content": str, "tool_calls": list, "eval_count": int}"""
    if LLM_BACKEND == "vllm":
        choices = data.get("choices") or []
        if not choices:
            return {"content": "", "tool_calls": [], "eval_count": 0}
        msg = choices[0].get("message", {}) or {}
        usage = data.get("usage", {}) or {}
        return {
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "eval_count": usage.get("completion_tokens", 0),
            "raw_message": msg,
        }
    msg = data.get("message", {}) or {}
    return {
        "content": msg.get("content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "eval_count": data.get("eval_count", 0) or 0,
        "raw_message": msg,
    }


def parse_stream_delta(line: bytes) -> tuple[str, bool]:
    """Parse a stream chunk line, return (content_delta, done_flag).

    For vLLM SSE we strip the leading "data: " prefix and ignore [DONE]."""
    s = line.strip()
    if not s:
        return "", False

    if LLM_BACKEND == "vllm":
        if s.startswith(b"data: "):
            s = s[len(b"data: "):]
        if s == b"[DONE]":
            return "", True
        try:
            obj = json.loads(s.decode("utf-8"))
        except Exception:
            return "", False
        choices = obj.get("choices") or []
        if not choices:
            return "", False
        delta = choices[0].get("delta") or {}
        finish = choices[0].get("finish_reason")
        return (delta.get("content") or ""), bool(finish)

    try:
        obj = json.loads(s.decode("utf-8"))
    except Exception:
        return "", False
    delta = (obj.get("message", {}) or {}).get("content", "") or ""
    return delta, bool(obj.get("done"))


def make_image_message(role: str, b64_jpegs: list[str], text: str) -> dict:
    """Build a single image+text message in **ollama-shape** (so callers stay
    backend-agnostic; we convert downstream)."""
    return {"role": role, "content": text, "images": b64_jpegs}
