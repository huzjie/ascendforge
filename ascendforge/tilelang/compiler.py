"""Compile a tile program (from IR or source) and return runnable artifacts."""
from .ir import TileProgram
from .parser import parse_source
from .scheduler import schedule
from .codegen import generate_ascend_c, generate_cpu


def compile_program(program: TileProgram, optimize: bool = True):
    if optimize:
        program = schedule(program)
    asc = generate_ascend_c(program)
    cpu = generate_cpu(program)
    return {"program": program, "ascend_c": asc, "cpu": cpu}


def compile_source(src: str, optimize: bool = True):
    program = parse_source(src)
    return compile_program(program, optimize)


def compile_and_run(src: str, tensors: dict, optimize: bool = True):
    """Convenience: compile source and run it on the simulator."""
    from .simulator import Simulator
    program = parse_source(src)
    if optimize:
        program = schedule(program)
    sim = Simulator()
    for name, t in tensors.items():
        sim.feed(name, t)
    out = sim.run(program)
    return out, program
