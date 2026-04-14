from types import SimpleNamespace

from hermes_cli.status import show_status


def test_show_status_includes_tavily_key(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-1...cdef")

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Tavily" in output
    assert "tvly...cdef" in output


def test_show_status_includes_image_generation_backend(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setattr(
        "hermes_cli.status.load_config",
        lambda: {
            "model": {"default": "anthropic/claude-sonnet-4", "provider": "anthropic"},
            "image_generation": {
                "provider": "openrouter",
                "model": "google/gemini-3.1-flash-image-preview",
            },
        },
        raising=False,
    )

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Image Gen" in output
    assert "openrouter (google/gemini-3.1-flash-image-preview)" in output


def test_show_status_marks_image_generation_unready_without_active_backend_key(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.status.load_config",
        lambda: {
            "model": {"default": "anthropic/claude-sonnet-4", "provider": "anthropic"},
            "image_generation": {
                "provider": "fal",
                "model": "fal-ai/flux-2-pro",
            },
        },
        raising=False,
    )
    monkeypatch.setattr("hermes_cli.status.check_fal_api_key", lambda: False, raising=False)

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Image Gen" in output
    assert "✗ fal (fal-ai/flux-2-pro)" in output


def test_show_status_normalizes_image_generation_provider_case(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setattr(
        "hermes_cli.status.load_config",
        lambda: {
            "model": {"default": "anthropic/claude-sonnet-4", "provider": "anthropic"},
            "image_generation": {
                "provider": "OpenRouter",
                "model": "google/gemini-3.1-flash-image-preview",
            },
        },
        raising=False,
    )

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "✓ openrouter (google/gemini-3.1-flash-image-preview)" in output


def test_show_status_fal_backend_uses_runtime_readiness(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.status.load_config",
        lambda: {
            "model": {"default": "anthropic/claude-sonnet-4", "provider": "anthropic"},
            "image_generation": {
                "provider": "fal",
                "model": "fal-ai/flux-2-pro",
            },
        },
        raising=False,
    )
    monkeypatch.setattr("hermes_cli.status.check_fal_api_key", lambda: True, raising=False)

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "✓ fal (fal-ai/flux-2-pro)" in output


def test_show_status_termux_gateway_section_skips_systemctl(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    def _unexpected_systemctl(*args, **kwargs):
        raise AssertionError("systemctl should not be called in the Termux status view")

    monkeypatch.setattr(status_mod.subprocess, "run", _unexpected_systemctl)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Manager:      Termux / manual process" in output
    assert "Start with:   hermes gateway" in output
    assert "systemd (user)" not in output
