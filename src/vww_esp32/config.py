from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a notebook, script, or test directory."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "base.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not find configs/base.yaml above the current directory")


def load_config(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    root = find_project_root(Path(path).parent if path else None)
    config_path = Path(path) if path else root / "configs" / "base.yaml"
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, root


def resolve_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    paths = {key: root / value for key, value in config["paths"].items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for child in ("checkpoints", "models", "logs", "figures", "reports"):
        (paths["artifacts"] / child).mkdir(parents=True, exist_ok=True)
    return paths


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except (ImportError, RuntimeError):
        pass
