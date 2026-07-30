#!/usr/bin/env python3
"""Codex (ChatGPT subscription) -> OpenAI chat/completions shim.

Why this exists
---------------
A Codex/ChatGPT subscription can drive the GPT-5.x models, but ONLY through
`https://chatgpt.com/backend-api/codex/responses`. Verified live (2026-07-30):

    chatgpt.com/backend-api/codex/responses   200  <- the only lane that works
    api.openai.com/v1/responses               401  missing scope api.responses.write
    api.openai.com/v1/chat/completions        429  "billing_not_active" (no PAYG balance)

That lane speaks the **Responses API** and *requires* `stream: true` (it 400s with
"Stream must be set to true" otherwise). LiteLLM speaks chat/completions. This proxy
translates between the two so `cod-*` aliases behave like any other OpenAI backend.

Same role as claude_oauth_proxy.py, but that one is a pass-through; this one has to
rewrite the request and re-assemble the streamed response.

Auth
----
Reads `~/.codex/auth.json` (written by the Codex CLI) **fresh on every request**, so a
token the CLI refreshes in the background is picked up with no restart. We deliberately
do NOT implement the OAuth refresh grant: the CLI already owns that, and duplicating it
risks racing the CLI for the same refresh token. If the token has expired, requests fail
with a clear message telling you to run `codex` once to re-authenticate.

Usage:  python3 providers/codex_oauth_proxy.py      (port 4042, override CODEX_OAUTH_PORT)
"""
import http.server
import json
import os
import ssl
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PORT = int(os.environ.get("CODEX_OAUTH_PORT", "4042"))
CREDS_PATH = Path(os.environ.get("CODEX_CREDS_PATH", Path.home() / ".codex" / "auth.json"))
UPSTREAM = "https://chatgpt.com/backend-api/codex/responses"
SESSION_ID = uuid.uuid4().hex
_SSL_CTX = ssl.create_default_context()

# Rejected by the Codex lane with "not supported when using Codex with a ChatGPT account".
# Kept here only so the error we return names the working set instead of echoing upstream noise.
SUPPORTED = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.4")


def _load_creds():
    """Return the parsed auth.json, or None. Re-read per request on purpose (see docstring)."""
    try:
        return json.loads(CREDS_PATH.read_text())
    except Exception:
        return None


def _token():
    """(access_token, account_id, error). error is a human-readable string when unusable."""
    d = _load_creds()
    if not d:
        return None, None, f"no credentials at {CREDS_PATH} — run `codex` and sign in"
    tok = (d.get("tokens") or {}).get("access_token")
    acct = (d.get("tokens") or {}).get("account_id") or ""
    if not tok:
        return None, None, "auth.json has no access_token — run `codex` and sign in"
    # Best-effort expiry check so the failure names the cause instead of surfacing a bare 401.
    try:
        import base64
        payload = tok.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        if exp and exp < time.time():
            return None, None, ("Codex token expired — run `codex` once to refresh it "
                                "(this proxy intentionally does not refresh it itself)")
    except Exception:
        pass          # opaque/!JWT token: let upstream be the judge
    return tok, acct, None


def _to_responses(body: dict) -> dict:
    """chat/completions request -> Responses API request."""
    out_input, instructions = [], []
    for m in body.get("messages", []):
        role, content = m.get("role"), m.get("content")
        if role == "system":
            # Responses API carries system separately; concatenate if several were sent.
            instructions.append(content if isinstance(content, str) else json.dumps(content))
            continue
        if role == "tool":
            out_input.append({"type": "function_call_output",
                              "call_id": m.get("tool_call_id", ""),
                              "output": content if isinstance(content, str) else json.dumps(content)})
            continue
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                out_input.append({"type": "function_call", "call_id": tc.get("id", ""),
                                  "name": fn.get("name", ""), "arguments": fn.get("arguments", "")})
            if not content:
                continue
        # "input_text" for user turns, "output_text" for assistant turns — the API rejects a mismatch.
        ctype = "output_text" if role == "assistant" else "input_text"
        if isinstance(content, list):     # already multimodal-shaped; pass parts we understand
            parts = [{"type": ctype, "text": p.get("text", "")}
                     for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text")]
        else:
            parts = [{"type": ctype, "text": content or ""}]
        out_input.append({"role": role, "content": parts})

    req = {"model": body.get("model", "gpt-5.6-sol"), "input": out_input,
           "stream": True,          # non-negotiable: the lane 400s without it
           "store": False}
    if instructions:
        req["instructions"] = "\n\n".join(instructions)
    # NOTE: this lane rejects `max_output_tokens` outright ("Unsupported parameter"), and it is a
    # subscription so there is no per-token cost to cap anyway. max_tokens from the caller is
    # therefore DROPPED, not translated — length is governed by the model/subscription.
    if body.get("temperature") is not None:
        req["temperature"] = body["temperature"]
    # Responses API flattens the function schema one level vs chat/completions.
    if body.get("tools"):
        tools = []
        for t in body["tools"]:
            fn = t.get("function") or {}
            if fn:
                tools.append({"type": "function", "name": fn.get("name"),
                              "description": fn.get("description", ""),
                              "parameters": fn.get("parameters", {})})
        if tools:
            req["tools"] = tools
    if body.get("reasoning_effort"):
        req["reasoning"] = {"effort": body["reasoning_effort"]}
    return req


def _consume(resp, model: str) -> dict:
    """Read the upstream SSE stream -> one chat/completions response object."""
    text, reasoning, tool_calls, usage = "", "", [], {}
    for raw in resp:
        line = raw.decode("utf8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            ev = json.loads(chunk)
        except Exception:
            continue
        t = ev.get("type", "")
        if t == "response.output_text.delta":
            text += ev.get("delta", "")
        elif t == "response.reasoning_summary_text.delta":
            reasoning += ev.get("delta", "")
        elif t == "response.output_item.done":
            item = ev.get("item") or {}
            if item.get("type") == "function_call":
                tool_calls.append({"id": item.get("call_id") or item.get("id", ""),
                                   "type": "function",
                                   "function": {"name": item.get("name", ""),
                                                "arguments": item.get("arguments", "")}})
        elif t == "response.completed":
            u = (ev.get("response") or {}).get("usage") or {}
            usage = {"prompt_tokens": u.get("input_tokens", 0),
                     "completion_tokens": u.get("output_tokens", 0),
                     "total_tokens": (u.get("input_tokens", 0) + u.get("output_tokens", 0))}

    msg = {"role": "assistant", "content": text or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


class CodexProxyHandler(http.server.BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/health/liveliness"):
            tok, _, err = _token()
            self._send(200 if tok else 503,
                       {"status": "ok" if tok else "error", "creds": bool(tok),
                        "session_id": SESSION_ID, **({"reason": err} if err else {})})
        elif self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send(200, {"object": "list",
                             "data": [{"id": m, "object": "model", "owned_by": "openai"} for m in SUPPORTED]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": {"message": "only /v1/chat/completions is served"}})
        tok, acct, err = _token()
        if err:
            return self._send(401, {"error": {"message": err, "type": "invalid_request_error"}})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except Exception as e:
            return self._send(400, {"error": {"message": f"bad JSON: {e}"}})

        model = body.get("model", "gpt-5.6-sol")
        req = urllib.request.Request(
            UPSTREAM, data=json.dumps(_to_responses(body)).encode(),
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json",
                     "chatgpt-account-id": acct, "OpenAI-Beta": "responses=experimental",
                     "Accept": "text/event-stream"})
        try:
            with urllib.request.urlopen(req, timeout=600, context=_SSL_CTX) as r:
                out = _consume(r, model)
        except urllib.error.HTTPError as e:
            detail = e.read()[:400].decode("utf8", "ignore")
            if e.code == 400 and "not supported" in detail:
                detail += f" | this subscription serves: {', '.join(SUPPORTED)}"
            return self._send(e.code, {"error": {"message": detail, "type": "upstream_error"}})
        except Exception as e:
            return self._send(502, {"error": {"message": f"upstream: {e}", "type": "upstream_error"}})
        self._send(200, out)

    def log_message(self, fmt, *args):        # keep the caller's stdout clean
        pass


def main():
    tok, _, err = _token()
    print(f"   Codex OAuth proxy on :{PORT} — " +
          (json.dumps({"status": "ok", "creds": True, "session_id": SESSION_ID}) if tok
           else json.dumps({"status": "error", "creds": False, "reason": err})))
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), CodexProxyHandler).serve_forever()


if __name__ == "__main__":
    main()
