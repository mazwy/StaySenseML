"""Shared helpers for the DVC stage scripts. Reads params.yaml and resolves
paths relative to the project root, so a stage can be run from anywhere."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict:
    with PARAMS_PATH.open() as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    """Turn a params-yaml path into an absolute path under the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
