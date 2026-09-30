"""Command-line interface: doctor / train / bench / serve / info."""
import argparse
import json


def _doctor(cfg):
    from .backends import list_backends
    from .core.registry import list_kernels
    from .ops.deepgemm import gemm
    from .core.tensor import randn
    # smoke: run a tiny gemm
    c = gemm(randn((8, 8), "doctor_a"), randn((8, 8), "doctor_b"))
    return {
        "status": "ok",
        "version": __import__("ascendforge.version", fromlist=["__version__"]).__version__,
        "backends": list_backends(),
        "kernels": list_kernels(),
        "gemm_smoke_shape": c.shape,
    }


def _train(cfg, steps):
    from .backends import make_backend
    from .train.trainer import Trainer
    backend = make_backend(cfg.backend, cfg=cfg)
    trainer = Trainer(backend, cfg)
    history = trainer.train(steps=steps)
    return {"backend": cfg.backend, "final_skill": history["skill"][-1],
            "final_loss": history["loss"][-1]}


def _bench(cfg):
    from .bench.kernels import bench_gemm, bench_mla, bench_moe, bench_dedup, bench_comm
    return [bench_gemm(), bench_mla(), bench_moe(), bench_dedup(), bench_comm()]


def _serve(cfg):
    from .serving.http_server import serve
    serve(cfg, backend=cfg.backend, host=cfg.serving.host, port=cfg.serving.port)


def _info(cfg):
    from .config import asdict  # noqa: F401  (kept for clarity)
    from dataclasses import asdict as _asdict
    return {"config": _asdict(cfg)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ascendforge", description="DeepSeek Ascend operator infrastructure")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("doctor")
    p_train = sub.add_parser("train")
    p_train.add_argument("--steps", type=int, default=500)
    sub.add_parser("bench")
    sub.add_parser("serve")
    sub.add_parser("info")

    args = parser.parse_args(argv)
    from .config import load_config
    cfg = load_config(args.config)

    if args.cmd == "doctor":
        out = _doctor(cfg)
    elif args.cmd == "train":
        out = _train(cfg, args.steps)
    elif args.cmd == "bench":
        out = _bench(cfg)
    elif args.cmd == "serve":
        _serve(cfg)
        return
    elif args.cmd == "info":
        out = _info(cfg)
    else:
        out = _doctor(cfg)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
