"""Stand-in for the Anthropic and OpenAI APIs, for smoke-testing the gateway
without provider keys or network access.

    python scripts/fake_upstream.py 9099

    ANTHROPIC_BASE_URL=http://127.0.0.1:9099 \\
    OPENAI_BASE_URL=http://127.0.0.1:9099/v1 \\
    uvicorn src.api.main:app

Every reply carries a call counter, so a gateway cache hit is visible as a
repeated number. A model name ending in ``-fail`` gets an HTTP 500. Standard
library only; not part of the gateway and never used by it.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

calls = 0


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global calls
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        model = body.get("model", "")
        if model.endswith("-fail"):
            return self._send(500, {"error": {"type": "api_error", "message": "upstream detail"}})
        calls += 1
        text = f"fake reply #{calls} from {model}"
        if self.path.endswith("/messages"):
            return self._send(
                200,
                {
                    "id": f"msg_fake_{calls}",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 5},
                },
            )
        if self.path.endswith("/chat/completions"):
            return self._send(
                200,
                {
                    "id": f"chatcmpl-fake-{calls}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
                },
            )
        return self._send(404, {"error": {"type": "not_found_error", "message": self.path}})

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9099
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
