"""Generate Ascend C pseudo-code (and a CPU fallback) from a TileProgram."""
from .ir import Op, OpKind, Loop


_OP_ASCEND = {
    OpKind.LOAD: "DataCopy({outs}, {ins}, GM->L1)",
    OpKind.STORE: "DataCopy({outs}, {ins}, L1->GM)",
    OpKind.GEMM: "Matmul({outs}, {ins})",
    OpKind.ELEMENTWISE: "Elemwise({outs}, {ins})",
    OpKind.REDUCE: "Reduce({outs}, {ins})",
    OpKind.ATTENTION: "Attention({outs}, {ins})",
    OpKind.FUSED: "Fused({outs}, {ins})",
}


def generate_ascend_c(program, indent="    "):
    lines = []
    lines.append(f"extern \"C\" __global__ __aicore__ void {program.name}() {{")
    lines.append(f"{indent}// target: {program.target}, pipeline_stages={program.pipeline_stages}")
    for t in program.tiles:
        lines.append(f"{indent}LocalTensor<{t.dtype}> {t.name};  // {t.shape} @ {t.memory}")
    lines.append("")
    for node in program.body:
        if isinstance(node, Op):
            tmpl = _OP_ASCEND.get(node.kind, "Op({outs}, {ins})")
            args = {"ins": ", ".join(node.inputs), "outs": ", ".join(node.outputs)}
            lines.append(indent + tmpl.format(**args) + ";")
        elif isinstance(node, Loop):
            lines.append(f"{indent}for (int {node.var} = {node.start}; {node.var} < {node.stop}; {node.var} += {node.step}) {{")
            for child in node.body:
                if isinstance(child, Op):
                    tmpl = _OP_ASCEND.get(child.kind, "Op({outs}, {ins})")
                    args = {"ins": ", ".join(child.inputs), "outs": ", ".join(child.outputs)}
                    lines.append(indent + indent + tmpl.format(**args) + ";")
            lines.append(indent + "}")
    lines.append("}")
    return "\n".join(lines)


def generate_cpu(program, indent="    "):
    """Plain-Python transliteration for the simulator (executed by Simulator)."""
    lines = [f"def {program.name}(ctx):"]
    for t in program.tiles:
        lines.append(f"{indent}{t.name} = ctx['{t.name}']")
    for node in program.body:
        if isinstance(node, Op):
            lines.append(f"{indent}ctx = _exec_{node.kind.value}(ctx, {node.inputs!r}, {node.outputs!r}, {node.attrs!r})")
        elif isinstance(node, Loop):
            lines.append(f"{indent}for {node.var} in range({node.start}, {node.stop}, {node.step}):")
            for child in node.body:
                if isinstance(child, Op):
                    lines.append(f"{indent}{indent}ctx = _exec_{child.kind.value}(ctx, {child.inputs!r}, {child.outputs!r}, {child.attrs!r})")
    lines.append(f"{indent}return ctx")
    return "\n".join(lines)
