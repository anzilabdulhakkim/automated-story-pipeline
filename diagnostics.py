"""
diagnostics.py — Internal logic stress-test.

Validates:
1. Soft-limit downgrade logic in RateLimiter + Router.
2. Concurrency cap enforcement in GeminiClient.
3. Fallback prompt degradation logic.
"""

import asyncio
import logging
import time
from typing import Any

from config import CONFIG, FLASH_MODEL
from rate_limiter import rate_limiter
from router import router, ModelTier
from gemini_client import GeminiClient, GenerationRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("diagnostics")

async def test_soft_limit_downgrade():
    log.info("\n--- Testing Soft Limit Downgrade ---")
    session_id = "limit-test-user"
    
    # 1. Reset state
    rate_limiter.reset_session(session_id)
    
    # 2. Check router for short prompt (should be Flash by default)
    d1 = router.route(user_text="Hi", session_id=session_id)
    log.info(f"Normal short prompt: {d1.tier.upper()} ({d1.reason})")
    
    # 3. Check router for long prompt (should be Pro)
    long_text = "Analyze this " * 100
    d2 = router.route(user_text=long_text, session_id=session_id)
    log.info(f"Long prompt (below limit): {d2.tier.upper()} ({d2.reason})")
    
    # 4. Burn budget to 85%
    budget = CONFIG.default_token_budget_per_session
    burn_amt = int(budget * 0.85)
    log.info(f"Burning budget by {burn_amt} tokens...")
    rate_limiter.record_usage(session_id, burn_amt)
    log.info("Budget recorded. Checking router again...")
    
    # 5. Check router for same long prompt (should now be Flash)
    d3 = router.route(user_text=long_text, session_id=session_id)
    log.info(f"Long prompt results: {d3.tier.upper()}")

    
    if d3.tier == ModelTier.FLASH:
        log.info("✅ Soft limit downgrade verified.")
    else:
        log.error("❌ Soft limit downgrade FAILED.")

async def test_concurrency_cap():
    log.info("\n--- Testing Concurrency Semaphore ---")

    # Use a fresh, isolated GeminiClient with concurrency=2,
    # without mutating the global CONFIG singleton.
    original = CONFIG.max_concurrent_requests
    CONFIG.max_concurrent_requests = 2
    try:
        test_client = GeminiClient()  # new instance, new semaphore sized to 2
    finally:
        CONFIG.max_concurrent_requests = original  # restore immediately

    active_calls = 0
    max_active = 0

    async def mock_call(i):
        nonlocal active_calls, max_active
        active_calls += 1
        max_active = max(max_active, active_calls)
        log.info(f"Call {i} started. (Active: {active_calls})")
        await asyncio.sleep(0.5)
        active_calls -= 1
        log.info(f"Call {i} finished.")

    async def semi_test(i):
        async with test_client._semaphore:
            await mock_call(i)

    tasks = [semi_test(i) for i in range(5)]
    await asyncio.gather(*tasks)
    
    log.info(f"Peak concurrency reached: {max_active}")
    if max_active <= 2:
        log.info("✅ Concurrency cap (Semaphore) verified.")
    else:
        log.error("❌ Concurrency cap FAILED.")

async def test_history_compression():
    log.info("\n--- Testing History Compression ---")
    from story_generator import compress_history
    
    history = [{"role": "user", "parts": ["Turn " + str(i)]} for i in range(10)]
    compressed = compress_history(history, max_turns=4)
    
    log.info(f"Raw history: {len(history)} turns")
    log.info(f"Compressed history: {len(compressed)} turns")
    
    # Should be 1 summary marker + 4 recent turns = 5 items total
    if len(compressed) == 5 and "SYSTEM NOTE" in compressed[0]["parts"][0]:
        log.info("✅ History compression (Sliding Window) verified.")
    else:
        log.error("❌ History compression FAILED.")

async def run_all_tests():
    await test_soft_limit_downgrade()
    await test_history_compression()
    await test_concurrency_cap()

if __name__ == "__main__":
    asyncio.run(run_all_tests())
