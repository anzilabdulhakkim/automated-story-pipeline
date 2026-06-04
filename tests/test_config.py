
import pytest

from config import ModelConfig, _required_env


def test_required_env_missing(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    with pytest.raises(EnvironmentError):
        _required_env("MISSING_VAR")

def test_model_cost_estimation():
    model = ModelConfig("test", 1000, 0.50, 3.00, 10)
    # 2000 input (2k * 0.5/1k = 1.0) + 1000 output (1k * 3.0/1k = 3.0) = 4.0
    cost = model.estimate_cost(2000, 1000)
    assert cost == 4.0
