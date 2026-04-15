"""Tests for Codex auth — tokens stored in Hermes auth store (~/.hermes/auth.json)."""

import json
import time
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    PROVIDER_REGISTRY,
    _read_codex_tokens,
    _save_codex_tokens,
    _write_codex_cli_tokens,
    _import_codex_cli_tokens,
    get_codex_auth_status,
    get_provider_auth_state,
    resolve_codex_runtime_credentials,
    resolve_provider,
)
from hermes_cli.model_switch import list_authenticated_providers


VALID_ACCESS_TOKEN = "plain-token-abcdefghijklmnopqrstuvwxyz0123456789"
VALID_REFRESH_TOKEN = "refresh-token-abcdefghijklmnopqrstuvwxyz0123456789"


def _setup_hermes_auth(hermes_home: Path, *, access_token: str = VALID_ACCESS_TOKEN, refresh_token: str = VALID_REFRESH_TOKEN):
    """Write Codex tokens into the Hermes auth store."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                "last_refresh": "2026-02-26T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
    }
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps(auth_store, indent=2))
    return auth_file


def _jwt_with_exp(exp_epoch: int) -> str:
    payload = {"exp": exp_epoch}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode("utf-8")
    return f"h.{encoded}.s"


def test_read_codex_tokens_success(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == VALID_ACCESS_TOKEN
    assert data["tokens"]["refresh_token"] == VALID_REFRESH_TOKEN


def test_read_codex_tokens_rejects_placeholder_tokens(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="access-new", refresh_token="refresh-new")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        _read_codex_tokens()
    assert exc.value.code == "codex_auth_invalid_access_token"


def test_read_codex_tokens_missing(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Empty auth store
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        _read_codex_tokens()
    assert exc.value.code == "codex_auth_missing"


def test_resolve_codex_runtime_credentials_missing_access_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == "codex_auth_missing_access_token"
    assert exc.value.relogin_required is True


def test_resolve_codex_runtime_credentials_refreshes_expiring_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    expiring_token = _jwt_with_exp(int(time.time()) - 10)
    _setup_hermes_auth(hermes_home, access_token=expiring_token, refresh_token="refresh-old-abcdefghijklmnopqrstuvwxyz0123456789")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    called = {"count": 0}

    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "access-new", "refresh_token": "refresh-new"}

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials()

    assert called["count"] == 1
    assert resolved["api_key"] == "access-new"


def test_resolve_codex_runtime_credentials_force_refresh(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="access-current-abcdefghijklmnopqrstuvwxyz0123456789", refresh_token="refresh-old-abcdefghijklmnopqrstuvwxyz0123456789")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    called = {"count": 0}

    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "access-forced", "refresh_token": "refresh-new"}

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials(force_refresh=True, refresh_if_expiring=False)

    assert called["count"] == 1
    assert resolved["api_key"] == "access-forced"


def test_resolve_provider_explicit_codex_does_not_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert resolve_provider("openai-codex") == "openai-codex"


def test_save_codex_tokens_roundtrip(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    access_token = "hermes-at-abcdefghijklmnopqrstuvwxyz0123456789"
    refresh_token = "hermes-rt-abcdefghijklmnopqrstuvwxyz0123456789"
    _save_codex_tokens({"access_token": access_token, "refresh_token": refresh_token})
    data = _read_codex_tokens()

    assert data["tokens"]["access_token"] == access_token
    assert data["tokens"]["refresh_token"] == refresh_token


def test_import_codex_cli_tokens(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-cli"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {
            "access_token": "cli-access-token-abcdefghijklmnopqrstuvwxyz0123456789",
            "refresh_token": "cli-refresh-token-abcdefghijklmnopqrstuvwxyz0123456789",
        },
    }))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    tokens = _import_codex_cli_tokens()
    assert tokens is not None
    assert tokens["access_token"] == "cli-access-token-abcdefghijklmnopqrstuvwxyz0123456789"
    assert tokens["refresh_token"] == "cli-refresh-token-abcdefghijklmnopqrstuvwxyz0123456789"


def test_import_codex_cli_tokens_ignores_placeholder_tokens(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-cli"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {"access_token": "access-new", "refresh_token": "refresh-new"},
    }))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert _import_codex_cli_tokens() is None


def test_import_codex_cli_tokens_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nonexistent"))
    assert _import_codex_cli_tokens() is None


def test_codex_tokens_not_written_to_shared_file(tmp_path, monkeypatch):
    """Verify _save_codex_tokens writes only to Hermes auth store, not ~/.codex/."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    hermes_home.mkdir(parents=True, exist_ok=True)
    codex_home.mkdir(parents=True, exist_ok=True)

    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    _save_codex_tokens({"access_token": "hermes-at-abcdefghijklmnopqrstuvwxyz0123456789", "refresh_token": "hermes-rt-abcdefghijklmnopqrstuvwxyz0123456789"})

    # ~/.codex/auth.json should NOT exist — _save_codex_tokens only touches Hermes store
    assert not (codex_home / "auth.json").exists()

    # Hermes auth store should have the tokens
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == "hermes-at-abcdefghijklmnopqrstuvwxyz0123456789"


def test_write_codex_cli_tokens_creates_file(tmp_path, monkeypatch):
    """_write_codex_cli_tokens creates ~/.codex/auth.json with refreshed tokens."""
    codex_home = tmp_path / "codex-cli"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    _write_codex_cli_tokens("new-access", "new-refresh", last_refresh="2026-04-12T00:00:00Z")

    auth_path = codex_home / "auth.json"
    assert auth_path.exists()
    data = json.loads(auth_path.read_text())
    assert data["tokens"]["access_token"] == "new-access"
    assert data["tokens"]["refresh_token"] == "new-refresh"
    assert data["last_refresh"] == "2026-04-12T00:00:00Z"
    # Verify file permissions are restricted
    assert (auth_path.stat().st_mode & 0o777) == 0o600


def test_write_codex_cli_tokens_preserves_existing(tmp_path, monkeypatch):
    """_write_codex_cli_tokens preserves extra fields in existing auth.json."""
    codex_home = tmp_path / "codex-cli"
    codex_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    existing = {
        "tokens": {
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "extra_field": "preserved",
        },
        "last_refresh": "2026-01-01T00:00:00Z",
        "custom_key": "keep_me",
    }
    (codex_home / "auth.json").write_text(json.dumps(existing))

    _write_codex_cli_tokens("updated-access", "updated-refresh")

    data = json.loads((codex_home / "auth.json").read_text())
    assert data["tokens"]["access_token"] == "updated-access"
    assert data["tokens"]["refresh_token"] == "updated-refresh"
    assert data["tokens"]["extra_field"] == "preserved"
    assert data["custom_key"] == "keep_me"
    # last_refresh not updated since we didn't pass it
    assert data["last_refresh"] == "2026-01-01T00:00:00Z"


def test_write_codex_cli_tokens_handles_missing_dir(tmp_path, monkeypatch):
    """_write_codex_cli_tokens creates parent directories if missing."""
    codex_home = tmp_path / "does" / "not" / "exist"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    _write_codex_cli_tokens("at", "rt")

    assert (codex_home / "auth.json").exists()
    data = json.loads((codex_home / "auth.json").read_text())
    assert data["tokens"]["access_token"] == "at"


def test_refresh_codex_auth_tokens_writes_back_to_cli(tmp_path, monkeypatch):
    """After refreshing, _refresh_codex_auth_tokens writes back to ~/.codex/auth.json."""
    from hermes_cli.auth import _refresh_codex_auth_tokens

    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    hermes_home.mkdir(parents=True, exist_ok=True)
    codex_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    # Write initial CLI tokens
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {"access_token": "old-at", "refresh_token": "old-rt"},
    }))

    # Mock the pure refresh to return new tokens
    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", lambda *a, **kw: {
        "access_token": "refreshed-at",
        "refresh_token": "refreshed-rt",
        "last_refresh": "2026-04-12T01:00:00Z",
    })

    _refresh_codex_auth_tokens(
        {"access_token": "old-at", "refresh_token": "old-rt"},
        timeout_seconds=10,
    )

    # Verify CLI file was updated
    cli_data = json.loads((codex_home / "auth.json").read_text())
    assert cli_data["tokens"]["access_token"] == "refreshed-at"
    assert cli_data["tokens"]["refresh_token"] == "refreshed-rt"


def test_resolve_returns_hermes_auth_store_source(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    creds = resolve_codex_runtime_credentials()
    assert creds["source"] == "hermes-auth-store"
    assert creds["provider"] == "openai-codex"
    assert creds["base_url"] == DEFAULT_CODEX_BASE_URL


def test_get_codex_auth_status_ignores_unusable_pool_entry(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(
        hermes_home,
        access_token="access-new",
        refresh_token="refresh-old-abcdefghijklmnopqrstuvwxyz0123456789",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    invalid_entry = SimpleNamespace(
        runtime_api_key="access-new",
        access_token="access-new",
        last_refresh="2026-04-15T00:00:00Z",
        label="stale-singleton",
    )

    class _FakePool:
        def has_credentials(self):
            return True

        def select(self):
            return invalid_entry

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: _FakePool())

    status = get_codex_auth_status()
    assert status["logged_in"] is False
    assert "unusable access_token" in status["error"]


def test_list_authenticated_providers_skips_invalid_codex_status(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.auth.get_codex_auth_status", lambda: {"logged_in": False})
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})

    providers = list_authenticated_providers(current_provider="openrouter")

    assert all(provider["slug"] != "openai-codex" for provider in providers)


def test_get_codex_auth_status_is_passive_for_legacy_auth_store(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    expired_token = _jwt_with_exp(int(time.time()) - 10)
    _setup_hermes_auth(
        hermes_home,
        access_token=expired_token,
        refresh_token="refresh-old-abcdefghijklmnopqrstuvwxyz0123456789",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: None)

    refresh_calls = []

    def _fake_refresh(tokens, timeout_seconds):
        refresh_calls.append((tokens, timeout_seconds))
        return {
            "access_token": "refreshed-access-token-abcdefghijklmnopqrstuvwxyz0123456789",
            "refresh_token": "refreshed-refresh-token-abcdefghijklmnopqrstuvwxyz0123456789",
            "last_refresh": "2026-04-15T00:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    status = get_codex_auth_status()

    assert refresh_calls == []
    assert status["logged_in"] is True
    assert status["source"] == "hermes-auth-store"
    assert status["auth_mode"] == "chatgpt"
    assert status["last_refresh"] == "2026-02-26T00:00:00Z"
    assert status["api_key"] == expired_token


def test_resolve_provider_auto_is_passive_for_legacy_codex_auth_store(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    expired_token = _jwt_with_exp(int(time.time()) - 10)
    _setup_hermes_auth(
        hermes_home,
        access_token=expired_token,
        refresh_token="refresh-old-abcdefghijklmnopqrstuvwxyz0123456789",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: None)

    refresh_calls = []

    def _fake_refresh(tokens, timeout_seconds):
        refresh_calls.append((tokens, timeout_seconds))
        return {
            "access_token": "refreshed-access-token-abcdefghijklmnopqrstuvwxyz0123456789",
            "refresh_token": "refreshed-refresh-token-abcdefghijklmnopqrstuvwxyz0123456789",
            "last_refresh": "2026-04-15T00:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    provider = resolve_provider("auto")

    assert refresh_calls == []
    assert provider == "openai-codex"
