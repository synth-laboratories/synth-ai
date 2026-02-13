from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_task_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "demos" / "mtg_artist_style" / "mtg_image_container.py"
    if not module_path.exists():
        pytest.skip(f"Missing demo module at {module_path}")
    spec = importlib.util.spec_from_file_location("mtg_image_container", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_extract_image_data_url_from_chat_parts():
    module = _load_task_module()
    extract = module._extract_image_data_url
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "here you go"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                    ]
                }
            }
        ]
    }
    assert extract(response) == "data:image/png;base64,aGVsbG8="


def test_extract_image_data_url_from_chat_parts_without_type():
    module = _load_task_module()
    extract = module._extract_image_data_url
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                    ]
                }
            }
        ]
    }
    assert extract(response) == "data:image/png;base64,aGVsbG8="


def test_extract_image_data_url_from_chat_string():
    module = _load_task_module()
    extract = module._extract_image_data_url
    response = {
        "choices": [
            {
                "message": {
                    "content": "data:image/png;base64,aGVsbG8=",
                }
            }
        ]
    }
    assert extract(response) == "data:image/png;base64,aGVsbG8="


def test_extract_image_data_url_from_images_data_b64():
    module = _load_task_module()
    extract = module._extract_image_data_url
    response = {"data": [{"b64_json": "aGVsbG8="}]}
    assert extract(response) == "data:image/png;base64,aGVsbG8="


def test_extract_image_data_url_missing():
    module = _load_task_module()
    extract = module._extract_image_data_url
    response = {"choices": [{"message": {"content": "no image"}}]}
    assert extract(response) is None
