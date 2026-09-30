"""Configuration loading with a zero-dependency YAML fallback."""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TileLangConfig:
    tile_size: int = 128
    dtype: str = "fp16"
    unroll: int = 4
    pipeline_stages: int = 2
    target: str = "ascend-c"


@dataclass
class GemmConfig:
    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    grouped: bool = True
    group_size: int = 8


@dataclass
class MlaConfig:
    hidden_size: int = 512
    num_heads: int = 8
    kv_lora_rank: int = 64
    qk_rope_dim: int = 32
    sliding_window: int = 256
    sparse: bool = True


@dataclass
class MoeConfig:
    num_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 1024
    aux_loss_coef: float = 0.01
    capacity_factor: float = 1.25


@dataclass
class CommConfig:
    num_ranks: int = 8
    topology: str = "hierarchical"
    all_to_all: str = "dispatch_combine"
    expert_parallel: bool = True


@dataclass
class TrainConfig:
    lr: float = 3e-4
    warmup_steps: int = 100
    max_steps: int = 2000
    batch_size: int = 8
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    scheduler: str = "cosine"


@dataclass
class DataConfig:
    root: str = "data/corpus"
    num_docs: int = 1000
    dedup: bool = True
    quality_threshold: float = 0.5
    curriculum: bool = True
    synth_seed: int = 42


@dataclass
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    tilelang: TileLangConfig = field(default_factory=TileLangConfig)
    gemm: GemmConfig = field(default_factory=GemmConfig)
    mla: MlaConfig = field(default_factory=MlaConfig)
    moe: MoeConfig = field(default_factory=MoeConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    backend: str = "mock"
    output_dir: str = "outputs"
    seed: int = 42


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _defaults() -> dict:
    return {
        "tilelang": asdict(TileLangConfig()),
        "gemm": asdict(GemmConfig()),
        "mla": asdict(MlaConfig()),
        "moe": asdict(MoeConfig()),
        "comm": asdict(CommConfig()),
        "train": asdict(TrainConfig()),
        "data": asdict(DataConfig()),
        "serving": asdict(ServingConfig()),
        "backend": "mock",
        "output_dir": "outputs",
        "seed": 42,
    }


def _build(cfg: dict) -> Config:
    merged = _merge(_defaults(), cfg)
    return Config(
        tilelang=TileLangConfig(**merged["tilelang"]),
        gemm=GemmConfig(**merged["gemm"]),
        mla=MlaConfig(**merged["mla"]),
        moe=MoeConfig(**merged["moe"]),
        comm=CommConfig(**merged["comm"]),
        train=TrainConfig(**merged["train"]),
        data=DataConfig(**merged["data"]),
        serving=ServingConfig(**merged["serving"]),
        backend=merged.get("backend", "mock"),
        output_dir=merged.get("output_dir", "outputs"),
        seed=merged.get("seed", 42),
    )


def load_config(path: Optional[str] = None) -> Config:
    if not path or not os.path.exists(path):
        return Config()
    data = None
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except ImportError:
            from .utils.yamlish import parse
            data = parse(path)
    if data is None:
        data = {}
    return _build(data)
