"""Tile scheduling passes: split, fuse, pipeline, tiling."""
from .ir import Op, Loop, OpKind, TileProgram


def tile_shape(shape, tile_size):
    return tuple(tile_size if s > tile_size else s for s in shape)


def split_loop(loop, factor):
    """Split a loop [0, stop) into an outer and inner loop by `factor`."""
    if loop.stop - loop.start <= factor:
        return [loop]
    outer = Loop(loop.var + "_outer", loop.start, loop.stop, factor)
    inner = Loop(loop.var + "_inner", 0, factor, 1, list(loop.body))
    outer.body = [inner]
    return [outer]


def pipeline(program: TileProgram, stages: int = None):
    """Naive software-pipelining annotation (metadata only)."""
    stages = stages or program.pipeline_stages
    program.pipeline_stages = stages
    # tag the first LOAD ops as prefetch candidates
    for node in program.body:
        if isinstance(node, Op) and node.kind == OpKind.LOAD:
            node.attrs["prefetch"] = True
    return program


def fuse_elementwise(program: TileProgram):
    """Fuse adjacent elementwise ops into FUSED ops."""
    fused_body = []
    pending = None
    for node in program.body:
        if isinstance(node, Op) and node.kind in (OpKind.ELEMENTWISE, OpKind.REDUCE):
            if pending is None:
                pending = Op(OpKind.FUSED, list(node.inputs), list(node.outputs),
                             {"ops": [node.kind.value]})
            else:
                pending.attrs["ops"].append(node.kind.value)
                pending.outputs = list(node.outputs)
        else:
            if pending is not None:
                fused_body.append(pending)
                pending = None
            fused_body.append(node)
    if pending is not None:
        fused_body.append(pending)
    program.body = fused_body
    return program


def schedule(program: TileProgram):
    pipeline(program)
    fuse_elementwise(program)
    return program
