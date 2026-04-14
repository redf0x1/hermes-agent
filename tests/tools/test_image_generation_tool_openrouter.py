import base64
import json
import types

import yaml


PNG_BYTES = b"\x89PNG\r\n\x1a\n fake png data"


class _FakeOpenRouterResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def _fake_openrouter_client(payload, captured: dict):
    def create(**kwargs):
        captured["create_kwargs"] = kwargs
        return _FakeOpenRouterResponse(payload)

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        )
    )


def test_openrouter_provider_includes_input_images_as_message_blocks(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "image_generation": {
                    "provider": "openrouter",
                    "model": "google/gemini-2.5-flash-image",
                }
            }
        ),
        encoding="utf-8",
    )

    source_a = tmp_path / "item-a.png"
    source_b = tmp_path / "item-b.jpg"
    source_a.write_bytes(PNG_BYTES)
    source_b.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"fake jpg data")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "image-cache")

    from tools import image_generation_tool

    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(PNG_BYTES).decode("ascii")
                            },
                        }
                    ],
                }
            }
        ]
    }
    captured = {}
    monkeypatch.setattr(
        image_generation_tool,
        "_get_openrouter_client",
        lambda *_args, **_kwargs: _fake_openrouter_client(payload, captured),
        raising=False,
    )

    result = json.loads(
        image_generation_tool.image_generate_tool(
            "combine these product shots into one clean catalog composition",
            input_images=[str(source_a), str(source_b)],
        )
    )

    assert result["success"] is True
    assert captured["create_kwargs"]["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "combine these product shots into one clean catalog composition"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"fake jpg data").decode("ascii")}},
            ],
        }
    ]


def test_openrouter_provider_rejects_missing_input_image(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "image_generation": {
                    "provider": "openrouter",
                    "model": "google/gemini-2.5-flash-image",
                }
            }
        ),
        encoding="utf-8",
    )

    missing = tmp_path / "missing.png"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("FAL_KEY", raising=False)

    from tools import image_generation_tool

    result = json.loads(
        image_generation_tool.image_generate_tool(
            "combine these product shots",
            input_images=[str(missing)],
        )
    )

    assert result["success"] is False
    assert result["error_type"] == "ValueError"
    assert "Input image does not exist" in result["error"]


def test_openrouter_provider_materializes_data_url_to_local_file(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "image_generation": {
                    "provider": "openrouter",
                    "model": "google/gemini-2.5-flash-image",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "image-cache")

    from tools import image_generation_tool

    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Generated image",
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(PNG_BYTES).decode("ascii")
                            },
                        }
                    ],
                }
            }
        ]
    }
    captured = {}
    monkeypatch.setattr(
        image_generation_tool,
        "_get_openrouter_client",
        lambda *_args, **_kwargs: _fake_openrouter_client(payload, captured),
        raising=False,
    )

    result = json.loads(image_generation_tool.image_generate_tool("make a sunset"))

    assert result["success"] is True
    assert result["image"].startswith(str(tmp_path))
    assert result["image"].endswith(".png")
    assert (tmp_path / "image-cache").exists()
    assert captured["create_kwargs"]["model"] == "google/gemini-2.5-flash-image"
    assert captured["create_kwargs"]["modalities"] == ["image", "text"]
    assert captured["create_kwargs"]["messages"] == [
        {"role": "user", "content": "make a sunset"}
    ]


def test_openrouter_provider_sends_image_config_aspect_ratio(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "image_generation": {
                    "provider": "openrouter",
                    "model": "google/gemini-2.5-flash-image",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "image-cache")

    from tools import image_generation_tool

    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(PNG_BYTES).decode("ascii")
                            },
                        }
                    ],
                }
            }
        ]
    }
    captured = {}
    monkeypatch.setattr(
        image_generation_tool,
        "_get_openrouter_client",
        lambda *_args, **_kwargs: _fake_openrouter_client(payload, captured),
        raising=False,
    )

    result = json.loads(image_generation_tool.image_generate_tool("make a logo", aspect_ratio="square"))

    assert result["success"] is True
    assert captured["create_kwargs"]["extra_body"] == {"image_config": {"aspect_ratio": "1:1"}}


def test_openrouter_provider_rejects_unsupported_aspect_ratio_model(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "image_generation": {
                    "provider": "openrouter",
                    "model": "openai/gpt-5-image-mini",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("FAL_KEY", raising=False)

    from tools import image_generation_tool

    result = json.loads(image_generation_tool.image_generate_tool("make a logo", aspect_ratio="square"))

    assert result["success"] is False
    assert result["error_type"] == "ValueError"
    assert "does not support Hermes aspect_ratio control" in result["error"]


def test_openrouter_provider_allows_default_landscape_for_model_without_aspect_ratio_control(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "image_generation": {
                    "provider": "openrouter",
                    "model": "openai/gpt-5-image-mini",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "image-cache")

    from tools import image_generation_tool

    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(PNG_BYTES).decode("ascii")
                            },
                        }
                    ],
                }
            }
        ]
    }
    captured = {}
    monkeypatch.setattr(
        image_generation_tool,
        "_get_openrouter_client",
        lambda *_args, **_kwargs: _fake_openrouter_client(payload, captured),
        raising=False,
    )

    result = json.loads(image_generation_tool.image_generate_tool("make a logo"))

    assert result["success"] is True
    assert "extra_body" not in captured["create_kwargs"]


def test_check_image_generation_requirements_accepts_openrouter_backend(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"image_generation": {"provider": "openrouter"}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("FAL_KEY", raising=False)

    from tools import image_generation_tool

    assert image_generation_tool.check_image_generation_requirements() is True
