#!/usr/bin/env python3
"""Claude Code OAuth proxy — stdlib-only, host-side, port 4041.

Receives Anthropic-format requests from LiteLLM (via host.docker.internal),
injects Claude Max OAuth headers, proxies to api.anthropic.com.

Flow:
  LiteLLM container  →  POST http://host.docker.internal:4041/v1/messages
  This proxy         →  reads ~/.claude/.credentials.json for OAuth token
                     →  POST https://api.anthropic.com/v1/messages
                          Authorization: Bearer <token>
                          anthropic-beta: claude-code-20250219,oauth-2025-04-20,...
                          user-agent: claude-cli/2.1.185 (external, sdk-cli)
                     →  stream response back to LiteLLM

Prereq: logged-in claude CLI so ~/.claude/.credentials.json exists.
  Run:  /opt/homebrew/bin/claude   →  /login   (or claude /login)

Stop: kill $(cat /tmp/claude-oauth-proxy.pid)
"""
import gzip
import http.server
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

PORT = int(os.environ.get("CLAUDE_OAUTH_PORT", "4041"))
CREDS_PATH = Path(os.environ.get("CLAUDE_CREDS_PATH",
                                  Path.home() / ".claude" / ".credentials.json"))
KEYCHAIN_SERVICE = "Claude Code-credentials"
KEYCHAIN_ACCOUNT = os.environ.get("USER", "")
UPSTREAM = "https://api.anthropic.com"
CHUNK = 8192
SESSION_ID = uuid.uuid4().hex

# OAuth token refresh (Claude Code's well-known public client)
OAUTH_TOKEN_URL = "https://api.anthropic.com/v1/oauth/token"
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
REFRESH_MARGIN_MS = 60_000  # refresh if within 60s of expiry (or already expired)
_refresh_lock = threading.Lock()

# Headers required for Claude Max OAuth
BETA = (
    "claude-code-20250219,"
    "oauth-2025-04-20,"
    "interleaved-thinking-2025-05-14,"
    "prompt-caching-scope-2026-01-05,"
    "context-management-2025-06-27"
)
INJECT_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."

# Headers LiteLLM sends that we must drop before forwarding
_DROP_REQ = {"x-api-key", "authorization", "anthropic-version", "anthropic-beta",
             "user-agent", "x-claude-code-session-id", "host", "connection",
             "transfer-encoding", "content-length"}  # content-length recalculated after system injection

_SSL_CTX = ssl.create_default_context()


def _load_creds():
    """Return (source, data): source is 'file' or 'keychain', data is the parsed
    credential JSON (whole blob, so refresh can write it back intact). File takes
    precedence, matching the prior _read_token() priority."""
    if CREDS_PATH.exists():
        try:
            data = json.loads(CREDS_PATH.read_text(encoding="utf-8"))
            if (data.get("claudeAiOauth") or {}).get("accessToken"):
                return "file", data
        except Exception:
            pass

    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            if (data.get("claudeAiOauth") or {}).get("accessToken"):
                return "keychain", data
    except Exception:
        pass

    raise FileNotFoundError(
        "Claude Code OAuth token not found.\n"
        f"  Tried: {CREDS_PATH}"
        + (f"  and  Keychain '{KEYCHAIN_SERVICE}'\n" if sys.platform == "darwin" else "\n")
        + "  Fix:   run `claude` then /login"
    )


def _save_creds(source: str, data: dict) -> None:
    if source == "file":
        CREDS_PATH.write_text(json.dumps(data), encoding="utf-8")
    else:
        subprocess.run(
            ["security", "add-generic-password", "-U", "-s", KEYCHAIN_SERVICE,
             "-a", KEYCHAIN_ACCOUNT, "-w", json.dumps(data)],
            capture_output=True, text=True, timeout=5, check=True,
        )


def _refresh(data: dict) -> dict:
    """POST refresh_token grant to Anthropic's OAuth endpoint. Returns the updated
    claudeAiOauth dict (new accessToken/refreshToken/expiresAt, rest untouched).
    Never logs token values — callers must not print `data` or the return value."""
    oa = data["claudeAiOauth"]
    body = json.dumps({
        "grant_type": "refresh_token",
        "refresh_token": oa["refreshToken"],
        "client_id": OAUTH_CLIENT_ID,
    }).encode()
    req = urllib.request.Request(
        OAUTH_TOKEN_URL, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            # Cloudflare 403s this endpoint without a browser/CLI-like UA.
            "User-Agent": "claude-cli/2.1.185 (external, sdk-cli)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=15) as resp:
        tok = json.loads(resp.read())
    oa["accessToken"] = tok["access_token"]
    oa["refreshToken"] = tok["refresh_token"]
    oa["expiresAt"] = int(time.time() * 1000) + int(tok["expires_in"]) * 1000
    return oa


def _read_token() -> str:
    """Read OAuth accessToken, refreshing first if expired or near expiry (<60s).
    Credentials file takes priority, macOS Keychain (Claude Code 2.x default) as
    fallback — refresh writes back to whichever source the token came from."""
    source, data = _load_creds()
    oa = data["claudeAiOauth"]

    if oa.get("expiresAt", 0) - time.time() * 1000 < REFRESH_MARGIN_MS:
        with _refresh_lock:
            # Re-check under the lock: another thread may have already refreshed
            # while we were waiting, so don't burn the (single-use) refresh token twice.
            source, data = _load_creds()
            oa = data["claudeAiOauth"]
            if oa.get("expiresAt", 0) - time.time() * 1000 < REFRESH_MARGIN_MS:
                oa = _refresh(data)
                data["claudeAiOauth"] = oa
                _save_creds(source, data)
                exp_str = time.strftime("%H:%M", time.localtime(oa["expiresAt"] / 1000))
                print(f"[oauth] refreshed, new expiry {exp_str}", flush=True)

    return oa["accessToken"]


def _inject_system(body_bytes: bytes) -> bytes:
    """Prepend Claude Code system prompt if not already present."""
    try:
        payload = json.loads(body_bytes)
        existing = payload.get("system") or ""
        if isinstance(existing, str):
            payload["system"] = (f"{INJECT_SYSTEM}\n\n{existing}").strip() \
                if existing else INJECT_SYSTEM
        # list-form system (multi-block): prepend as first text block
        elif isinstance(existing, list):
            payload["system"] = [{"type": "text", "text": INJECT_SYSTEM}] + existing
        else:
            payload["system"] = INJECT_SYSTEM
        return json.dumps(payload).encode("utf-8")
    except Exception:
        return body_bytes  # malformed body — forward as-is


class OAuthProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            creds_ok = _creds_available()
            body = json.dumps({
                "status": "ok",
                "creds": creds_ok,
                "session_id": SESSION_ID,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        # Auth
        try:
            token = _read_token()
        except Exception as exc:
            msg = f"OAuth error: {exc}".encode()
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        # Inject Claude Code system prompt
        body = _inject_system(body)

        # Build upstream request headers
        fwd_headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": BETA,
            "user-agent": "claude-cli/2.1.185 (external, sdk-cli)",
            "X-Claude-Code-Session-Id": SESSION_ID,
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        # Forward any non-auth client headers (e.g. custom trace IDs)
        for key, val in self.headers.items():
            if key.lower() not in _DROP_REQ:
                fwd_headers[key] = val

        url = f"{UPSTREAM}{self.path}"
        req = urllib.request.Request(url, data=body, headers=fwd_headers, method="POST")

        try:
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=600) as resp:
                self.send_response(resp.status)
                for key, val in resp.headers.items():
                    k = key.lower()
                    if k in ("transfer-encoding", "connection", "keep-alive"):
                        continue
                    self.send_header(key, val)
                self.end_headers()
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except urllib.error.HTTPError as exc:
            err_body = exc.read()
            # Upstream honours the client's accept-encoding, so an error body may arrive
            # gzipped. We relabel it application/json and drop content-encoding, so it must
            # be decompressed here or the caller sees raw gzip bytes instead of the message.
            if exc.headers.get("content-encoding", "").lower() == "gzip":
                try:
                    err_body = gzip.decompress(err_body)
                except Exception:
                    pass
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)

        except Exception as exc:
            msg = f'{{"error": "{exc}"}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def log_message(self, fmt, *args):  # silence default stdout spam
        pass


def _creds_available() -> bool:
    if CREDS_PATH.exists():
        return True
    r = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
                       capture_output=True, timeout=5)
    return r.returncode == 0


def main():
    creds_ok = _creds_available()
    print(f"Claude OAuth proxy  0.0.0.0:{PORT}  "
          f"creds={'OK ✓' if creds_ok else 'MISSING ✗'}", flush=True)
    if not creds_ok:
        print(f"  ⚠  no credentials: file {CREDS_PATH}  or  Keychain '{KEYCHAIN_SERVICE}'",
              flush=True)
        print("  Log in: /opt/homebrew/bin/claude  then type /login", flush=True)
    print(f"  session_id={SESSION_ID}", flush=True)

    # Bind to 0.0.0.0 so host.docker.internal (OrbStack) can reach it from the container
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), OAuthProxyHandler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
