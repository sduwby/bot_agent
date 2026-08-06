"""本地环境变量加载测试。"""

from __future__ import annotations

import os
from pathlib import Path

from env_config import load_local_env


def test_load_local_env_without_overriding_shell(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_MODEL=from-file\nLLM_API_KEY='secret-value'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_MODEL", "from-shell")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    load_local_env(env_file)

    assert os.environ["LLM_MODEL"] == "from-shell"
    assert os.environ["LLM_API_KEY"] == "secret-value"
