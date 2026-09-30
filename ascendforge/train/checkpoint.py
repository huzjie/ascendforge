"""Checkpoint save/load (JSON)."""
import json
import os


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return path


def load_checkpoint(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
