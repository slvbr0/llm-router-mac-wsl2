"""Minimal stdio MCP server exposing the fusion engine as one tool: fuse."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion.fusion import fuse  # noqa: E402

TOOL = {
    "name": "fuse",
    "description": ("Mixture-of-Agents fusion for [NOVEL] prompts: fans one task to a "
                    "provider-diverse model panel via the local router and synthesizes the best "
                    "answer, returning a cost receipt. mode='research' uses Perplexity council "
                    "(web-grounded; requires confirm_research=true after quota warning)."),
    "inputSchema": {"type": "object", "properties": {
        "prompt": {"type": "string", "description": "The task (with or without [NOVEL] tags)"},
        "mode": {"type": "string", "enum": ["fuse", "research"]},
        "depth": {"type": "string", "enum": ["easy", "hard"]},
        "confirm_research": {"type": "boolean"}},
        "required": ["prompt"]},
}


def reply(id_, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": id_, "result": result}) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid, method, params = msg.get("id"), msg.get("method", ""), msg.get("params", {}) or {}
        if method == "initialize":
            reply(mid, {"protocolVersion": params.get("protocolVersion", "2025-03-26"),
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fusion", "version": "2.0"}})
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            reply(mid, {"tools": [TOOL]})
        elif method == "tools/call" and params.get("name") == "fuse":
            a = params.get("arguments", {}) or {}
            try:
                res = fuse(a["prompt"], mode=a.get("mode"), depth=a.get("depth"),
                           confirm_research=bool(a.get("confirm_research")))
                r = res["receipt"]
                summary = (f"[fusion receipt] mode={r.get('mode')} diff={r.get('difficulty','-')} "
                           f"tokens={r.get('total_tokens','-')} est_cost={r.get('est_cost',{})} "
                           f"wall={r.get('wall_ms','-')}ms degraded={r.get('degraded', False)}")
                reply(mid, {"content": [{"type": "text", "text": res["answer"] + "\n\n" + summary}]})
            except Exception as e:
                reply(mid, {"content": [{"type": "text", "text": f"fusion error: {e}"}],
                            "isError": True})
        elif mid is not None:
            reply(mid, {})


if __name__ == "__main__":
    main()
