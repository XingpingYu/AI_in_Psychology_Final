"""脚手架自检:不打 API,只确认模块可 import、config 可加载。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_imports():
    import src.utils.config as cfg
    import src.utils.env as env  # noqa: F401
    import src.utils.io as io_mod  # noqa: F401
    import src.utils.logging_setup as lg  # noqa: F401
    import src.clients as clients
    models = cfg.load_models()
    exp = cfg.load_experiment()
    assert isinstance(models, list) and len(models) > 0
    assert "tasks" in exp
    assert "openai" in clients.list_supported_providers()


if __name__ == "__main__":
    test_imports()
    print("smoke ok")
