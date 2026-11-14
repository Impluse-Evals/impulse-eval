# SPDX-License-Identifier: AGPL-3.0-only
"""A standard-library echo agent behind the HTTP adapter's three endpoints, for local testing only.

    python examples/http/server.py            # listens on http://localhost:8080
    python examples/http/server.py 9000       # another port

Endpoints (POST, JSON bodies; README.md in this folder has the exact JSON):

- ``/start``: answers ``{"system_prompt": "..."}`` (used only when config.yaml sets none).
- ``/turn``: echoes ``user_text``; the opener (``user_text`` null) is a greeting; when the text
  contains the whole word "book" the answer also carries one dummy ``book_slot`` tool call.
- ``/close``: answers ``{}``.

It keeps no state, needs no key, and is not meant to be exposed beyond your own machine.
"""

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SYSTEM_PROMPT = (
    "You are the booking assistant of a demo clinic. Greet the caller and answer briefly. "
    "When the caller asks to book an appointment, call the book_slot tool with the date, "
    "the patient's name, and the phone number, and confirm the booking only if the tool succeeded."
)

_BOOK_WORD = re.compile(r"\bbook\b", re.IGNORECASE)


def turn_answer(body):
    """The answer to one ``/turn`` request body."""
    user_text = body.get("user_text")
    if not user_text:
        return {"text": "Hello, this is the demo clinic. How can I help you today?"}
    answer = {"text": f"You said: {user_text}", "tool_calls": []}
    if _BOOK_WORD.search(user_text):
        answer["text"] += " I have booked a slot for you."
        answer["tool_calls"].append(
            {
                "name": "book_slot",
                "arguments": {"date": "2026-01-01", "patient_name": "Test User", "phone": "0000000000"},
                "result": {"success": True},
                "success": True,
            }
        )
    return answer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):   # the method name http.server dispatches POST requests to
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except ValueError:
            self._send(400, {"error": "the body is not JSON"})
            return
        if not isinstance(body, dict):
            self._send(400, {"error": "the body is not a JSON object"})
        elif self.path == "/start":
            self._send(200, {"system_prompt": SYSTEM_PROMPT})
        elif self.path == "/turn":
            self._send(200, turn_answer(body))
        elif self.path == "/close":
            self._send(200, {})
        else:
            self._send(404, {"error": f"no endpoint {self.path}"})

    def _send(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = ThreadingHTTPServer(("localhost", port), Handler)
    print(f"Echo agent listening on http://localhost:{port}/turn (start: /start, close: /close)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
