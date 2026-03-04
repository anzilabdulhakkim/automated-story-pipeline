"""
diagnostics.py — Internal logic stress-test.

Validates:
1. Concurrency cap enforcement in GeminiClient.
2. History compression logic.
"""

import asyncio
import logging

from config import CONFIG
from gemini_client import GeminiClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("diagnostics")

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
    await test_history_compression()
    await test_concurrency_cap()

if __name__ == "__main__":
    asyncio.run(run_all_tests())
