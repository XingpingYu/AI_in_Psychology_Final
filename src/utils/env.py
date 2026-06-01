""".env 加载封装。务必不打印任何 key 内容。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .config import project_root


def load_env(verbose: bool = True) -> dict[str, bool]:
    """加载项目根目录下的 .env,返回各 key 是否存在(只报存在性,不报内容)。"""
    dotenv_path = project_root() / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)
    checks = {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "DEEPSEEK_API_KEY": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
        "TOGETHER_API_KEY": bool(os.environ.get("TOGETHER_API_KEY")),
    }
    if verbose:
        for k, v in checks.items():
            status = "set" if v else "MISSING"
            print(f"  env {k:22s} : {status}")
    return checks
