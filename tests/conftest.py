
import pytest


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "gemini-test-text")
    monkeypatch.setenv("GEMINI_IMAGE_MODEL", "gemini-test-image")
