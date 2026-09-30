"""Parse a mini tile-DSL source string into a TileProgram.

Grammar (subset):
    program <name>(<shape>) {
        tile <name> : <dtype>[M, K] @ <mem>
        loop <var> in range(<start>, <stop>[, <step>]) { ... }
        <kind>(in=<tile>, out=<tile>, ...)
    }
"""
import re

from .ir import Op, OpKind, Loop, Tile, TileProgram


def parse_source(src: str) -> TileProgram:
    name_m = re.search(r"program\s+(\w+)\s*\(([^)]*)\)\s*\{", src)
    name = name_m.group(1) if name_m else "kernel"
    body = []
    tiles = []
    tile_pat = re.compile(r"tile\s+(\w+)\s*:\s*(\w+)\s*\[([^\]]+)\]\s*(?:@\s*(\w+))?")
    loop_pat = re.compile(r"loop\s+(\w+)\s+in\s+range\(([^)]+)\)\s*\{")
    op_pat = re.compile(r"(\w+)\(([^)]*)\)")
    for line in src.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("program") or line == "}":
            continue
        tm = tile_pat.match(line)
        if tm:
            dims = tuple(int(x.strip()) for x in tm.group(3).split(","))
            tiles.append(Tile(tm.group(1), dims, tm.group(2), tm.group(4) or "L1"))
            continue
        lm = loop_pat.match(line)
        if lm:
            parts = [p.strip() for p in lm.group(2).split(",")]
            start = int(parts[0]) if len(parts) > 0 else 0
            stop = int(parts[1]) if len(parts) > 1 else 1
            step = int(parts[2]) if len(parts) > 2 else 1
            body.append(Loop(lm.group(1), start, stop, step))
            continue
        om = op_pat.match(line)
        if om:
            kind = _kind_of(om.group(1))
            attrs = {}
            ins, outs = [], []
            for kv in om.group(2).split(","):
                kv = kv.strip()
                if not kv or "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                v = v.strip()
                if k == "in":
                    ins.append(v)
                elif k == "out":
                    outs.append(v)
                else:
                    attrs[k] = v
            body.append(Op(kind, ins, outs, attrs))
    return TileProgram(name, tiles, body)


def _kind_of(word):
    mapping = {
        "load": OpKind.LOAD, "store": OpKind.STORE, "gemm": OpKind.GEMM,
        "add": OpKind.ELEMENTWISE, "mul": OpKind.ELEMENTWISE, "relu": OpKind.ELEMENTWISE,
        "softmax": OpKind.REDUCE, "layernorm": OpKind.REDUCE, "rmsnorm": OpKind.REDUCE,
        "attention": OpKind.ATTENTION, "fused": OpKind.FUSED,
    }
    return mapping.get(word, OpKind.ELEMENTWISE)
