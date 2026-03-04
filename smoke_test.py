"""
smoke_test.py — Phase 1 import + logic verification.
Run: python smoke_test.py
"""
import io
import os
import sys

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []

def ok(label):
    print(f"  {PASS}  {label}")

def fail(label, exc):
    print(f"  {FAIL}  {label}: {exc}")
    errors.append((label, exc))

print("\n[1] config.py")
try:
    from config import CONFIG, FLASH_MODEL, PipelineConfig
    assert isinstance(FLASH_MODEL.name, str) and FLASH_MODEL.name.strip()
    assert CONFIG.max_concurrent_requests == 3
    assert CONFIG.get_max_tokens("story_generation") == 8192
    assert CONFIG.get_max_tokens("classification")   == 32
    cost = FLASH_MODEL.estimate_cost(1000, 500)
    assert cost > 0
    ok(f"TEXT={FLASH_MODEL.name}")
    ok(f"cost_estimate(1k in, 500 out) = ${cost:.7f}")
    ok(f"max_tokens(story_generation)  = {CONFIG.get_max_tokens('story_generation')}")
except Exception as e:
    fail("config", e)

print("\n[2] logger.py")
try:
    from logger import APICallRecord, PipelineLogger
    # Use a temp log path so we don't pollute the real log
    test_logger = PipelineLogger(log_path="logs/_smoke_test.jsonl")
    record = APICallRecord(
        timestamp=APICallRecord.now_iso(),
        task_type="smoke_test",
        model_used="gemini-2.0-flash",
        input_tokens=200,
        output_tokens=100,
        latency_ms=342.1,
        cost_usd_estimate=0.0000453,
        cache_hit=False,
        story_id="smoke-001",
        router_decision="task type smoke_test → Flash",
    )
    test_logger.log(record)
    records = test_logger.read_all()
    assert len(records) >= 1
    assert records[-1]["task_type"] == "smoke_test"
    assert records[-1]["total_tokens"] == 300
    ok(f"Wrote + read JSONL record | total_tokens={records[-1]['total_tokens']}")
    os.remove("logs/_smoke_test.jsonl")
except Exception as e:
    fail("logger", e)

print("\n[3] cache.py")
try:
    from cache import PipelineCache
    test_cache = PipelineCache(cache_dir=".cache/_smoke", ttl_seconds=60)
    test_cache.set("flash", "system", "user_prompt_A", '{"title":"Test"}')
    hit = test_cache.get("flash", "system", "user_prompt_A")
    miss = test_cache.get("flash", "system", "user_prompt_B")
    assert hit  == '{"title":"Test"}', f"Expected hit, got: {hit}"
    assert miss is None,               f"Expected miss, got: {miss}"
    test_cache.invalidate("flash", "system", "user_prompt_A")
    after_invalidate = test_cache.get("flash", "system", "user_prompt_A")
    assert after_invalidate is None, "Invalidate failed"
    ok(f"backend={test_cache.backend}  hit=PASS  miss=PASS  invalidate=PASS")
except Exception as e:
    fail("cache", e)

print("\n[4] rate_limiter.py")
try:
    from rate_limiter import BudgetExhaustedError, RateLimiter, SessionBudget
    lim = RateLimiter(default_budget=1000, state_file="logs/_smoke_rate.json")

    # Fresh session — should NOT be at soft limit
    assert not lim.should_downgrade("sess-A"), "Fresh session should not downgrade"

    # Record 85% usage → should trigger soft limit
    lim.record_usage("sess-A", 850)
    assert lim.should_downgrade("sess-A"),     "850/1000 should trigger soft limit"
    assert not lim.is_exhausted("sess-A"),     "850/1000 should not be exhausted"

    # Record remainder → exhausted
    lim.record_usage("sess-A", 150)
    assert lim.is_exhausted("sess-A"),         "1000/1000 should be exhausted"

    report = lim.status_report()
    entry = next(r for r in report if r["session_id"] == "sess-A")
    ok(f"session-A: {entry['tokens_used']}/{entry['token_budget']}  exhausted={entry['exhausted']}")

    lim.reset_session("sess-A")
    assert not lim.is_exhausted("sess-A"), "After reset should not be exhausted"
    ok("reset: PASS")

    if os.path.exists("logs/_smoke_rate.json"):
        os.remove("logs/_smoke_rate.json")
except Exception as e:
    fail("rate_limiter", e)

print("\n[5] router.py")
try:
    from router import ModelTier, RequestRouter
    rt = RequestRouter()

    cases = [
        ("write a bedtime story for Olivia", "story_generation", ModelTier.FLASH),
        ("generate image of a princess",     "story_generation", ModelTier.IMAGEN),
        ("analyze and compare these research papers in depth", "story_generation", ModelTier.FLASH),
        ("summarize this text",              "summarization",    ModelTier.FLASH),
        ("classify this intent",             "classification",   ModelTier.FLASH),
    ]
    for prompt, task, expected in cases:
        decision = rt.route(user_text=prompt, task_type=task, session_id="smoke-router")
        assert decision.tier == expected, \
            f"Prompt '{prompt[:40]}': expected {expected}, got {decision.tier}"
        ok(f"[{decision.tier.value:6}] conf={decision.confidence:.0%}  '{prompt[:40]}'")

    # Force override test (IMAGEN)
    forced = rt.route(
        user_text="simple prompt",
        task_type="story_generation",
        session_id="smoke-router",
        force_tier=ModelTier.IMAGEN,
    )
    assert forced.tier == ModelTier.IMAGEN
    ok("force_tier=IMAGEN override: PASS")

except Exception as e:
    fail("router", e)

# [6] gemini_client (import + dataclass checks — no live API call)
print("\n[6] gemini_client.py (import + types check)")
try:
    from gemini_client import (
        GeminiClient,
        GenerationRequest,
        GenerationResponse,
        get_client,
    )

    req = GenerationRequest(
        prompt="Tell me a story",
        task_type="story_generation",
        session_id="smoke-client",
        story_id="smoke-001",
        force_json=True,
        max_output_tokens=512,
    )
    assert req.prompt            == "Tell me a story"
    assert req.force_json        == True
    assert req.max_output_tokens == 512
    assert req.history           == []
    ok("GenerationRequest dataclass: PASS")

    assert callable(get_client)
    ok("get_client() lazy factory: importable (live call skipped)")

    resp = GenerationResponse(
        text='{"title":"Test"}',
        model_used="gemini-2.0-flash",
        input_tokens=100,
        output_tokens=50,
        latency_ms=200.0,
        cost_usd_estimate=0.00001,
        cache_hit=False,
    )
    assert resp.total_tokens == 150
    ok(f"GenerationResponse.total_tokens = {resp.total_tokens}: PASS")

except Exception as e:
    fail("gemini_client", e)


print("="*55)
if errors:
    print(f"  {len(errors)} FAILURE(S):")
    for label, exc in errors:
        print(f"    {FAIL} {label}: {exc}")
    sys.exit(1)
else:
    print("  ALL 6 MODULES PASSED [OK]")
    print("="*55)
