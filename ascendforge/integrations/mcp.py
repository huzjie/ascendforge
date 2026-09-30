"""A Model Context Protocol (MCP) server exposing ascendforge kernels.

Implements the MCP JSON-RPC protocol over stdio, so any MCP client can invoke
ascendforge operators (gemm / mla / moe / bench) as tools.
"""
import json
import sys

from ..backends import make_backend


_TOOLS = [
    {
        "name": "ascendforge_bench",
        "description": "Run ascendforge kernel benchmarks (gemm/mla/moe/dedup/comm).",
        "inputSchema": {"type": "object", "properties": {"which": {"type": "string"}}},
    },
    {
        "name": "ascendforge_train",
        "description": "Train the deterministic mock backend for N steps.",
        "inputSchema": {"type": "object", "properties": {"steps": {"type": "integer"}}},
    },
    {
        "name": "ascendforge_doctor",
        "description": "Run a health check and list backends/kernels.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class AscendForgeMCPServer:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.backend = make_backend("mock", cfg=cfg)
        self.tools = _TOOLS

    def handle(self, method, params):
        if method == "initialize":
            return {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ascendforge", "version": "1.0.0"}}
        if method == "tools/list":
            return {"tools": self.tools}
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {})
            return {"content": [{"type": "text", "text": json.dumps(self._call(name, args))}]}
        if method == "ping":
            return {}
        return {"error": f"unknown method {method}"}

    def _call(self, name, args):
        if name == "ascendforge_doctor":
            from ..backends import list_backends
            return {"status": "ok", "backends": list_backends()}
        if name == "ascendforge_train":
            steps = int(args.get("steps", 100))
            from ..train.trainer import Trainer
            t = Trainer(self.backend, self.cfg)
            h = t.train(steps=steps)
            return {"final_skill": h["skill"][-1]}
        if name == "ascendforge_bench":
            which = args.get("which", "gemm")
            from ..bench.kernels import bench_gemm, bench_mla, bench_moe, bench_dedup, bench_comm
            fn = {"gemm": bench_gemm, "mla": bench_mla, "moe": bench_moe,
                  "dedup": bench_dedup, "comm": bench_comm}.get(which, bench_gemm)
            return fn()
        return {"error": f"unknown tool {name}"}

    def serve_stdio(self):
        """Run the MCP server over stdio (JSON-RPC, newline-delimited)."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = {"jsonrpc": "2.0", "id": req.get("id"),
                        "result": self.handle(req.get("method"), req.get("params", {}))}
            except Exception as e:  # noqa: BLE001
                resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


def serve_mcp(cfg=None):
    AscendForgeMCPServer(cfg).serve_stdio()
