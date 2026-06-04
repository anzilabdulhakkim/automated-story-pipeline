import pytest

from rate_limiter import RateLimiter


@pytest.fixture
def limiter(tmp_path):
    state_file = tmp_path / "test_rate_limits.json"
    return RateLimiter(default_budget=1000, state_file=str(state_file))

def test_token_budget_tracking(limiter):
    session = limiter.record_usage("test_session", 100)
    assert session.tokens_used == 100
    assert session.tokens_remaining == 900

def test_soft_limit_downgrade(limiter):
    # Default soft limit is 0.80. Budget is 1000, so 800 tokens.
    limiter.record_usage("test_session_2", 800)
    assert limiter.should_downgrade("test_session_2") is True
