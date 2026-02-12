# streaming.py
"""Server-Sent Events (SSE) streaming utilities."""

import json
import time
from typing import Any, Dict, Iterable


def make_openai_response(text: str) -> Dict[str, Any]:
    """Create OpenAI-compatible response format."""
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "moa-router",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def stream_sse(text: str, keepalive_interval: float = 15.0) -> Iterable[str]:
    """
    Stream text as SSE with periodic keepalive pings.
    
    Args:
        text: Text to stream
        keepalive_interval: Seconds between keepalive comments (default 15s)
    """
    if not text:
        text = ""
    
    last_keepalive = time.time()
    
    # Send text in chunks
    for i, char in enumerate(text):
        # Check if we need a keepalive
        now = time.time()
        if now - last_keepalive > keepalive_interval:
            yield ": keepalive\n\n"
            last_keepalive = now
        
        chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "moa-router",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": char},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # Send final chunk
    final_chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "moa-router",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
