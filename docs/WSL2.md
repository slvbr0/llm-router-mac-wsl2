# Running the router on WSL2

The tree is the same one macOS runs; the OAuth shims already pick their bind address per
platform, so there are no per-host edits. What follows is the Linux/WSL2 operational detail
that is either different from macOS or cost real debugging time here.

Read the two compose notes in the [README](../README.md#install) first — `docker start` not
re-reading compose, and `litellm` being the service while `litellm-proxy` is only the
container name. Everything below is on top of those.

---

## Reaching the router

Bound to `127.0.0.1:4040` inside the distro. WSL2 forwards localhost, so a Windows-side
process reaches the same router at `http://localhost:4040` with no extra configuration.
Nothing needs to be published to `0.0.0.0` for that to work.

Clients (opencode, Claude Code, Codex) run **inside** the distro. From another machine, SSH
into the Windows host, `wsl -d <distro>`, and start the client there — the router, the
compression proxy and the memory service are all on the distro's loopback.

## Why the Codex shim binds differently here

On macOS, Docker Desktop and OrbStack proxy `host.docker.internal` to the host's loopback, so
a proxy bound to `127.0.0.1` is reachable from the router container. On Linux/WSL2,
`host.docker.internal` resolves to the **docker bridge gateway** instead, and a loopback-bound
socket is invisible from inside a container — every `cod-*` call fails with
`OpenAIException - Connection error` while `ant-*` keeps working, because the Claude shim
already binds wider.

`scripts/start_codex_proxy.sh` handles this: when a `docker0` interface exists it reads that
interface's address and exports `CODEX_OAUTH_HOST`. The bridge is deliberately narrower than
`0.0.0.0` — containers and the host reach it, the LAN does not. Neither shim performs inbound
authentication, so anything that can reach one spends the subscription behind it. Do not widen
the bind past the bridge.

Check which address is actually bound:

```bash
ss -tlnp | grep -E ':(4041|4042)'
```

## Keeping the shims alive across a restart

`scripts/start.sh` launches both shims with `nohup`, which does not survive a WSL restart —
the `ant-*` and `cod-*` lanes come back dead and every request falls through to another lane.
On a distro with systemd, user units fix that permanently. Enable lingering once
(`loginctl enable-linger $USER`) so they start without a login session.

```ini
# ~/.config/systemd/user/codex-oauth-proxy.service
[Unit]
Description=Codex OAuth proxy for llm-router (:4042)
After=network.target docker.service

[Service]
Type=simple
WorkingDirectory=%h/projects/llm-router-mac-wsl2
ExecStart=/bin/sh -c 'CODEX_OAUTH_HOST=$(ip -4 -o addr show docker0 2>/dev/null | awk "{print \\$4}" | cut -d/ -f1); export CODEX_OAUTH_HOST="${CODEX_OAUTH_HOST:-127.0.0.1}"; exec /usr/bin/python3 providers/codex_oauth_proxy.py'
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

`Restart=always`, not `on-failure`: an external `SIGTERM` looks like a clean exit to systemd,
so `on-failure` leaves the lane silently dead after a stray `pkill`. `systemctl --user stop`
still stops it — systemd distinguishes that from the process dying underneath it. The Claude
shim takes the same unit without the bind detection.

## Two traps that look like the router being broken

### Right project name, wrong directory

Compose identifies a stack by **project name**, not by where you ran it. Start it from a
different checkout while the project name matches and you get the original named volumes —
database, OAuth tokens, all intact — with config, `priority_router.py`, `availability.yaml`
and `.env` read from the *other* directory. The stack comes up healthy and serves a stale
model list, and because `.env` differs, every client fails with
`Invalid proxy server token`.

```bash
docker inspect litellm-proxy \
  --format 'project={{index .Config.Labels "com.docker.compose.project"}}
dir={{index .Config.Labels "com.docker.compose.project.working_dir"}}'
```

If `dir` is not the checkout you expect, `docker compose down` and `up -d` from the right one.
Keeping only one directory whose name could plausibly be "the router" removes the trap
entirely — an archived copy should be renamed, not left beside the live one.

### Postgres keeps the password it was first initialised with

`POSTGRES_PASSWORD` is read only when the data directory is created. Point a different `.env`
at an existing volume and LiteLLM boot-loops on
`P1000: Authentication failed against database server at 'db'`, which reads like a config typo
but is a mismatch against what the volume already stores.

Testing it is its own trap: `psql` inside the db container over the local socket, or with
`-h 127.0.0.1`, succeeds regardless of the password. Only a TCP connection over the compose
network actually exercises authentication:

```bash
docker exec -e PGPASSWORD="$PW" litellm-db psql -h db -U litellm -d litellm -c 'select 1'
```

Align the volume to the `.env` rather than editing the `.env` — one `ALTER USER litellm WITH
PASSWORD '…'` as the working password, and the audit history survives. Deleting the volume
also works and throws away every routing record.

## Health probes and native reasoners

`scripts/nim_health.sh` imports `NATIVE_REASONERS` from `priority_router` so the two cannot
drift, and falls back to an empty list if that import fails. An empty list is silent: every
reasoner is then judged against the non-reasoner latency ceiling and benched for thinking,
which is the failure the reasoner ceiling exists to prevent. Worth re-checking after anything
that moves files around:

```bash
python3 -c 'import priority_router as p; print(sorted(p.NATIVE_REASONERS))'
```

Empty output means the health audit is about to mark your reasoners unhealthy.

A model flagged `ok:false` is usually just down — NIM in particular returns 503 and 529 under
load and recovers on the next tick. Replaying the probe payload against one alias separates a
dead lane from a misjudged one:

```bash
curl -s http://localhost:4040/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"nim-step","messages":[{"role":"user","content":"Reply with the single word: OK"}],
       "max_tokens":16,"num_retries":0,"fallbacks":[],"metadata":{"health_probe":true}}'
```

`finish_reason=length` with `content:""` and a non-empty `reasoning_content` is a healthy
reasoner spending the cap on thought, not a failure.

## Reading routing decisions

The `[alias · think:level · tier]` banner is prepended to the response **content**, so it
disappears whenever a reasoner spends the whole budget on `reasoning_content` — a test that
parses the banner then reports nothing and looks like a routing failure. The audit trail is
the reliable source:

```bash
sh scripts/show_routing.sh 20      # time · routed alias · actual model · tokens
sh scripts/show_routing.sh watch   # live
```
