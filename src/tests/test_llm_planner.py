"""LLM 后端配置边界测试。"""

from __future__ import annotations

import openai

from llm_planner import OpenAIBackend, _SYSTEM_PROMPT_TEMPLATE


def test_openai_backend_passes_finite_timeout_and_retries(monkeypatch) -> None:
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(openai, "OpenAI", fake_client)
    backend = OpenAIBackend(
        api_base="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        timeout_seconds=12.5,
        max_retries=1,
    )

    backend._ensure_client()

    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 1


def test_openai_backend_invalid_env_uses_safe_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "invalid")
    monkeypatch.setenv("LLM_MAX_RETRIES", "-3")

    backend = OpenAIBackend(api_key="test-key")

    assert backend.timeout_seconds == 60.0
    assert backend.max_retries == 1


def test_system_prompt_exposes_executor_search_contract() -> None:
    assert "看不到目标物体，也要直接调用 pick" in _SYSTEM_PROMPT_TEMPLATE
    assert "看不到目标容器，也要直接调用 place" in _SYSTEM_PROMPT_TEMPLATE
    assert "不要用重复 scan 或手工 move_to" in _SYSTEM_PROMPT_TEMPLATE
    assert "不得创造 ahead" in _SYSTEM_PROMPT_TEMPLATE


# ---- 多模态图像输入（S1） ----

def test_image_to_data_url_from_ndarray() -> None:
    import numpy as np

    from llm_planner import image_to_data_url

    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    url = image_to_data_url(img, max_size=64)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > 40


def test_image_to_data_url_passthrough() -> None:
    from llm_planner import image_to_data_url

    url = "data:image/jpeg;base64,AAAA"
    assert image_to_data_url(url) == url


def test_image_to_data_url_invalid_input_raises() -> None:
    import pytest

    from llm_planner import image_to_data_url

    with pytest.raises((TypeError, ValueError)):
        image_to_data_url("not an image path")


def test_build_multimodal_content_no_images_is_text() -> None:
    from llm_planner import build_multimodal_content

    content = build_multimodal_content("你好", None)
    assert isinstance(content, str) and content == "你好"


def test_build_multimodal_content_with_images() -> None:
    from llm_planner import build_multimodal_content

    content = build_multimodal_content("看画面", ["data:image/jpeg;base64,A", "data:image/jpeg;base64,B"])
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "看画面"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,A"}}
    assert len(content) == 3
