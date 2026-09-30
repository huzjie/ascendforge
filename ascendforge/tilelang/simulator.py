"""Deterministic CPU simulator for tile programs (no real NPU needed)."""
from ..core.tensor import matmul, add, mul, relu, gelu, softmax, layernorm, rmsnorm
from .ir import OpKind


def _exec_load(ctx, ins, outs, attrs):
    for o, i in zip(outs, ins):
        ctx[o] = ctx[i]
    return ctx


def _exec_store(ctx, ins, outs, attrs):
    for o, i in zip(outs, ins):
        ctx[o] = ctx[i]
    return ctx


def _exec_gemm(ctx, ins, outs, attrs):
    a, b = ins[0], ins[1]
    ctx[outs[0]] = matmul(ctx[a], ctx[b])
    return ctx


def _exec_elementwise(ctx, ins, outs, attrs):
    op = attrs.get("op", "add")
    a, b = ctx[ins[0]], ctx[ins[1]] if len(ins) > 1 else None
    if op == "mul":
        ctx[outs[0]] = mul(a, b)
    elif op == "relu":
        ctx[outs[0]] = relu(a)
    elif op == "gelu":
        ctx[outs[0]] = gelu(a)
    else:
        ctx[outs[0]] = add(a, b)
    return ctx


def _exec_reduce(ctx, ins, outs, attrs):
    op = attrs.get("op", "softmax")
    a = ctx[ins[0]]
    if op == "layernorm":
        ctx[outs[0]] = layernorm(a)
    elif op == "rmsnorm":
        ctx[outs[0]] = rmsnorm(a)
    else:
        ctx[outs[0]] = softmax(a)
    return ctx


def _exec_attention(ctx, ins, outs, attrs):
    # deterministic attention: softmax(Q @ K^T) @ V
    q, k, v = ctx[ins[0]], ctx[ins[1]], ctx[ins[2]]
    kt = _transpose(k)
    scores = matmul(q, kt)
    probs = softmax(scores)
    ctx[outs[0]] = matmul(probs, v)
    return ctx


def _exec_fused(ctx, ins, outs, attrs):
    # apply ops in sequence
    val = ctx[ins[0]]
    for op in attrs.get("ops", []):
        if op == "relu":
            val = relu(val)
        elif op == "gelu":
            val = gelu(val)
        elif op == "softmax":
            val = softmax(val)
        elif op == "layernorm":
            val = layernorm(val)
        elif op == "rmsnorm":
            val = rmsnorm(val)
    ctx[outs[0]] = val
    return ctx


def _transpose(t):
    m, n = t.shape
    data = [0.0] * (n * m)
    for i in range(m):
        for j in range(n):
            data[j * m + i] = t.data[i * n + j]
    from ..core.tensor import Tensor
    return Tensor(data, (n, m), t.dtype, t.device)


_DISPATCH = {
    "load": _exec_load, "store": _exec_store, "gemm": _exec_gemm,
    "elementwise": _exec_elementwise, "reduce": _exec_reduce,
    "attention": _exec_attention, "fused": _exec_fused,
}


class Simulator:
    def __init__(self):
        self.ctx = {}

    def feed(self, name, tensor):
        self.ctx[name] = tensor

    def run(self, program):
        from .ir import Op, Loop
        ctx = self.ctx
        for node in program.body:
            ctx = self._exec_node(node, ctx)
        self.ctx = ctx
        return ctx

    def _exec_node(self, node, ctx):
        from .ir import Op, Loop
        if isinstance(node, Loop):
            for _ in range(node.start, node.stop, node.step):
                for child in node.body:
                    ctx = self._exec_node(child, ctx)
            return ctx
        if isinstance(node, Op):
            fn = _DISPATCH.get(node.kind.value)
            if fn is None:
                return ctx
            return fn(ctx, node.inputs, node.outputs, node.attrs)
        return ctx

    def get(self, name):
        return self.ctx[name]
