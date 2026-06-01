"""配置加载工具。
- load_models / load_experiment: 读 yaml
- get_active_models: 过滤 enabled
- task_sample_size: 按当前 mode (smoke/full) 取样本量字段
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path else CONFIG_DIR / "models.yaml"
    data = load_yaml(path)
    return data.get("models", [])


def load_experiment(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else CONFIG_DIR / "experiment.yaml"
    return load_yaml(path)


def get_active_models(models: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    models = models if models is not None else load_models()
    return [m for m in models if m.get("enabled", False)]


def task_sample_size(exp_cfg: dict[str, Any], task_name: str, field: str) -> Any:
    mode = exp_cfg.get("mode", "smoke")
    task_cfg = exp_cfg.get("tasks", {}).get(task_name, {})
    section = task_cfg.get(mode, {})
    return section.get(field)


def project_root() -> Path:
    return PROJECT_ROOT
