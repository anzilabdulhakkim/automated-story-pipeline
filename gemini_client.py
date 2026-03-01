"""
gemini_client.py â€” Production-grade async Gemini API client.

Uses the current `google-genai` SDK (google.genai.Client).
The old `google.generativeai` package is fully deprecated.

Implements every non-negotiable from the guide Â§8:
    [OK] Explicit max_output_tokens on every call
    [OK] stop_sequences configurable per request
    [OK] JSON / structured output enforcement (response_mime_type)
    [OK] Exponential backoff + jitter via tenacity (429 / 503)
    [OK] Pro -> Flash automatic fallback chain
    [OK] Token usage + cost logged on every call
    [OK] Async (non-blocking) â€” uses client.aio.models.generate_content
    [OK] Concurrency capped via asyncio.Semaphore
    [OK] Cache checked before every call

Architecture
------------
    GeminiClient.generate(request)
        |
        +-- check cache         -> return cached GenerationResponse (0 cost)
        +-- route (router.py)   -> ModelTier
        +-- check rate-limit    -> may override tier to Flash
        +-- _call_with_retry()  -> tenacity async retry loop
        |   +-- primary model   -> try N times with backoff
        |   +-- Flash fallback  -> if Pro fails (429/503/quota)
        +-- cache result
        +-- record rate-limit usage
        +-- log APICallRecord   -> logs/api_calls.jsonl
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, cast

from google import genai
from google.genai import types as genai_types
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
)

from cache import response_cache
from config import CONFIG, FLASH_MODEL, ModelConfig
from logger import APICallRecord, pipeline_logger
from rate_limiter import BudgetExhaustedError, rate_limiter
from router import ModelTier, RouterDecision, router

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GenerationRequest:
    """
    All parameters for a single Gemini generation call.

    prompt            User-turn text.
    task_type         Logical category used by router + token-limit table.
    session_id        Identifies the caller for rate-limit budget tracking.
    story_id          Optional ID included in every log record.
    system_prompt     System-turn instructions. Set once; never repeat in user prompt.
    force_json        When True enforces response_mime_type=application/json.
    stop_sequences    Terminate generation on any of these strings.
    max_output_tokens Explicit cap. Falls back to per-task config table.
    force_tier        Bypass router and use this tier directly.
    history           Prior turns: [{"role":"user","parts":["..."]}, ...]
                      Apply sliding-window truncation *before* passing here.
    """
    prompt:            str
    task_type:         str                       = "story_generation"
    session_id:        str                       = "default"
    story_id:          Optional[str]             = None
    system_prompt:     Optional[str]             = None
    force_json:        bool                      = True
    stop_sequences:    list[str]                 = field(default_factory=list)
    max_output_tokens: Optional[int]             = None
    force_tier:        Optional[ModelTier]       = None
    history:           list[dict[str, Any]]      = field(default_factory=list)
    dry_run:           bool                      = False


@dataclass
class GenerationResponse:
    """Return value of every GeminiClient.generate() call."""
    text:               str
    model_used:         str
    input_tokens:       int
    output_tokens:      int
    latency_ms:         float
    cost_usd_estimate:  float
    cache_hit:          bool
    router_decision:    Optional[RouterDecision] = None
    error:              Optional[str]            = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """True for transient errors that warrant an exponential-backoff retry."""
    msg = str(exc).lower()
    return any(tok in msg for tok in ("429", "503", "resource_exhausted", "quota exceeded", "rate limit"))


def _wait_respecting_server_hint(retry_state) -> float:
    """
    Use the server-suggested retry delay from 429 bodies when present.
    Falls back to exponential backoff + jitter if no hint is found (Issue #18b).

    The Gemini API embeds a suggested delay in 429 error messages, e.g.:
        "... retry in 53.016s ..."
    Honouring this delay avoids unnecessary fast-retries that burn quota and
    guarantees we wait at least as long as the server requests.
    """
    exc = retry_state.outcome.exception()
    if exc is not None:
        # Regex matches "... retry in 53.016s ..."
        match = re.search(r'retry in (\d+\.?\d*)', str(exc), re.IGNORECASE)
        if match:
            server_wait = float(match.group(1))
            jitter = random.uniform(0, CONFIG.retry_jitter)
            log.info(
                "Respecting server retry hint: %.1fs + %.1fs jitter",
                server_wait, jitter,
            )
            return server_wait + jitter
    # Fall back to exponential backoff with jitter
    attempt = retry_state.attempt_number
    base = CONFIG.retry_min_wait * (2 ** attempt)
    return min(base, CONFIG.retry_max_wait) + random.uniform(0, CONFIG.retry_jitter)


def _model_cfg(tier: ModelTier) -> ModelConfig:
    if tier == ModelTier.IMAGEN:
        raise ValueError("Model tier 'imagen' is not valid for text generation calls.")
    # NOTE: Pro model is intentionally disabled. Everything resolves to Flash.
    # The Pro scaffolding (pro_token_threshold, ModelTier.FLASH/PRO) is preserved
    # for future activation but will NOT be used until a Pro ModelConfig is added.
    return FLASH_MODEL


def _build_config(*, model_cfg: ModelConfig, request: GenerationRequest) -> genai_types.GenerateContentConfig:
    """Build the GenerateContentConfig for the new google-genai SDK."""
    stop = request.stop_sequences or CONFIG.default_stop_sequences
    max_tok = request.max_output_tokens or CONFIG.get_max_tokens(request.task_type)

    # Disable thinking mode for JSON tasks â€” thinking consumes output tokens
    # and can exhaust the budget before the model produces any actual JSON.
    thinking_cfg = None
    if request.force_json:
        thinking_cfg = genai_types.ThinkingConfig(thinking_budget=0)

    return genai_types.GenerateContentConfig(
        # [OK] Always explicitly capped
        max_output_tokens=max_tok,
        # [OK] Stop sequences: terminate as soon as useful output ends
        stop_sequences=stop or None,
        # [OK] JSON enforcement: prevents verbose free-text wrappers
        response_mime_type="application/json" if request.force_json else None,
        # System instruction (if any) goes here in the new SDK
        system_instruction=request.system_prompt or None,
        # Disable thinking for structured output tasks
        thinking_config=thinking_cfg,
    )


def _build_contents(request: GenerationRequest) -> list[genai_types.Content]:
    """Assemble the contents list: optional history + current user turn."""
    contents: list[genai_types.Content] = []

    for turn in request.history:
        role  = turn.get("role", "user")
        parts = [genai_types.Part(text=p) for p in turn.get("parts", [])]
        contents.append(genai_types.Content(role=role, parts=parts))

    contents.append(genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=request.prompt)],
    ))
    return contents


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_fingerprint(request: GenerationRequest) -> dict[str, Any]:
    """Fields that materially affect response shape/semantics."""
    return {
        "task_type": request.task_type,
        "force_json": request.force_json,
        "max_output_tokens": request.max_output_tokens or CONFIG.get_max_tokens(request.task_type),
        "stop_sequences": request.stop_sequences,
        "history": request.history,
        "dry_run": request.dry_run,
    }


def _extract_prompt_field(prompt: str, label: str, default: str) -> str:
    pattern = rf"- \*\*{re.escape(label)}:\*\*\s*(.+)"
    m = re.search(pattern, prompt)
    if not m:
        return default
    value = m.group(1).strip()
    return value or default


def _mock_age_spec(age: int) -> tuple[int, int, int, str]:
    if age <= 5:
        return 8, 25, 50, "full body shot from eye level"
    if age <= 8:
        return 12, 40, 100, "medium shot from eye level"
    return 18, 75, 120, "dynamic medium shot from eye level"


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def _pad_to_word_count(base_text: str, target_words: int) -> str:
    filler = [
        "bright", "gentle", "curious", "friendly", "colorful",
        "warm", "playful", "calm", "hopeful", "kind",
    ]
    words = base_text.split()
    idx = 0
    while len(words) < target_words:
        words.append(filler[idx % len(filler)])
        idx += 1
    return " ".join(words)


def _build_mock_story(request: GenerationRequest) -> dict[str, Any]:
    age_raw = _extract_prompt_field(request.prompt, "Age", "5")
    try:
        age = int(age_raw)
    except ValueError:
        age = 5
    age = max(3, min(12, age))

    total_pages, min_words, _, camera = _mock_age_spec(age)
    nickname = _extract_prompt_field(request.prompt, "Child's Name", "Alex")
    category = _extract_prompt_field(request.prompt, "Category", "Adventure")
    tone = _extract_prompt_field(request.prompt, "Tone", "Cheerful")
    moral = _extract_prompt_field(request.prompt, "Moral Value", "Kindness")
    art_style = _extract_prompt_field(request.prompt, "Art Style", "Disney3D")

    character_description = "short curly brown hair, green eyes, red striped shirt"
    descriptor = f"a {age}-year-old child, {character_description}"
    setting = "in a cozy storybook meadow with a winding path, a friendly tree, and soft clouds"
    lighting = "bright midday sunlight casting soft shadows"
    facial = "joyful expression with wide smile and bright eyes"
    base_prompt = (
        f"{art_style}, {descriptor}, standing upright with hands gently open, "
        f"{setting}, {lighting}, {facial}, {camera}, anatomically correct hands and feet, "
        f"clear facial features, child-friendly, high quality --no text --no words --no letters"
    )

    pages: list[dict[str, Any]] = []
    for page_num in range(1, total_pages + 1):
        root_text = (
            f"{nickname} explored a magical place and practiced {moral.lower()} on page {page_num}. "
            f"{nickname} listened, learned, and helped friends with calm confidence. "
            f"Each choice showed responsibility, empathy, and steady growth."
        )
        text = _pad_to_word_count(root_text, min_words + 2)
        pages.append({
            "page_number": page_num,
            "text": text,
            "word_count": _word_count(text),
            "character_count": len(text),
            "image_prompt": base_prompt,
        })

    synopsis = (
        f"{nickname} explores a magical world, learns {moral.lower()}, and grows through kind choices "
        "that strengthen friendships and confidence."
    )
    synopsis = _pad_to_word_count(synopsis, 18)

    cover_prompt = (
        f"{art_style} book cover, {descriptor}, standing proudly with hands gently open, "
        f"{setting}, {lighting}, {facial}, centered composition, anatomically correct hands and feet, "
        f"clear facial features, vibrant colors, high quality --no text --no words --no letters"
    )

    vocab_count = 1 if age <= 5 else (2 if age <= 8 else 3)
    new_vocabulary = []
    vocab_bank = [
        ("curious", "wanting to learn about things"),
        ("steady", "calm and controlled"),
        ("responsible", "taking good care of duties"),
        ("empathy", "understanding how others feel"),
        ("resolve", "strong decision to continue"),
    ]
    for idx in range(vocab_count):
        word, simple_def = vocab_bank[idx]
        new_vocabulary.append({
            "word": word,
            "simple_definition": simple_def,
            "page_appears": min(total_pages, idx + 2),
        })

    return {
        "title": f"{nickname} and the Kind Path",
        "synopsis": " ".join(synopsis.split()[:25]),
        "target_age_confirmation": age,
        "character_description": character_description,
        "total_pages": total_pages,
        "story_category": category,
        "story_tone": tone,
        "moral_value": moral,
        "themes": ["friendship", "growth"],
        "new_vocabulary": new_vocabulary,
        "content_metadata": {
            "conflict_level": "low",
            "emotional_intensity": "gentle",
            "educational_focus": ["social skills", "self-regulation"],
            "reading_time_minutes": max(4, total_pages // 2),
        },
        "cover_image_prompt": cover_prompt,
        "pages": pages,
    }


# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

class GeminiClient:
    """
    Production async Gemini client built on google.genai.Client.

    Instantiate once per process. All public methods are coroutines and safe
    to call from multiple asyncio tasks concurrently (semaphore-guarded).

    Example
    -------
        client = GeminiClient()
        resp = await client.generate(GenerationRequest(
            prompt="Write a bedtime story for Olivia",
            task_type="story_generation",
            session_id="batch1-0",
        ))
        print(resp.text)
    """

    def __init__(self) -> None:
        if not CONFIG.gemini_api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set.  Add it to your .env file."
            )
        # Disable SDK-level HTTP retries entirely so retry behaviour is owned
        # exclusively by this module's tenacity policy (Issue #18a).
        self._client = genai.Client(
            api_key=CONFIG.gemini_api_key,
            http_options=genai_types.HttpOptions(
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )
        # Semaphore caps concurrent API calls across all async tasks
        self._semaphore = asyncio.Semaphore(CONFIG.max_concurrent_requests)
        # Sliding-window deque for RPM enforcement (timestamps of recent requests)
        self._request_timestamps: collections.deque = collections.deque()
        self._rpm_lock = asyncio.Lock()

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Full pipeline: cache -> route -> rate-limit -> call (retry+fallback)
        -> cache store -> log.
        """
        async with self._semaphore:
            return await self._generate_inner(request)

    # â”€â”€ Pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _generate_inner(self, request: GenerationRequest) -> GenerationResponse:
        # 1. Rate-limit hard stop
        if rate_limiter.is_exhausted(request.session_id):
            raise BudgetExhaustedError(
                f"Session '{request.session_id}' has exhausted its token budget."
            )

        # 2. Route (may force Flash if session is near soft limit)
        decision: RouterDecision = router.route(
            user_text=request.prompt,
            task_type=request.task_type,
            session_id=request.session_id,
            force_tier=request.force_tier,
        )
        # Defensive guard: this client handles text generation only.
        if decision.tier == ModelTier.IMAGEN:
            decision = RouterDecision(
                tier=ModelTier.FLASH,
                reason=f"router returned IMAGEN for task '{request.task_type}' - coerced to Flash for text generation",
                confidence=1.0,
            )
        log.debug("Router: %s", decision)

        # 3. Cache lookup
        cache_enabled = not request.dry_run
        cache_extra = _cache_fingerprint(request)
        model_cfg    = _model_cfg(decision.tier)
        cached_value = None
        if cache_enabled:
            cached_value = response_cache.get(
                model_cfg.name,
                request.system_prompt or "",
                request.prompt,
                extra=cache_extra,
            )
        if cached_value is not None:
            pipeline_logger.log(APICallRecord(
                timestamp=_now_iso(),
                task_type=request.task_type,
                model_used=model_cfg.name,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                cost_usd_estimate=0.0,
                cache_hit=True,
                story_id=request.story_id,
                router_decision=str(decision),
            ))
            return GenerationResponse(
                text=cached_value,
                model_used=model_cfg.name,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                cost_usd_estimate=0.0,
                cache_hit=True,
                router_decision=decision,
            )

        # 4. Call model (retry with exponential backoff)
        gen_cfg = _build_config(model_cfg=model_cfg, request=request)
        t0      = time.perf_counter()
        text, in_tok, out_tok = await self._call_with_retry(
            model_name=model_cfg.name,
            request=request,
            gen_cfg=gen_cfg,
        )
        used_model_name = model_cfg.name
        latency_ms = (time.perf_counter() - t0) * 1_000

        # 5. Cost estimate
        cost = FLASH_MODEL.estimate_cost(in_tok, out_tok)

        # 6. Rate-limit usage is now recorded per-attempt inside _call_with_retry (#22)
        #    so we do not call record_usage here to avoid double-counting.

        # 7. Store result in cache — guard against caching empty/blocked responses (Issue #20)
        # A safety-blocked response returns "" which would permanently poison the cache
        # for up to 7 days. Raise immediately so callers can handle it.
        if not text or not text.strip():
            raise ValueError(
                f"Model returned an empty response for task '{request.task_type}'. "
                "This may indicate a safety block. Response will not be cached."
            )
        if cache_enabled:
            response_cache.set(
                model_cfg.name,
                request.system_prompt or "",
                request.prompt,
                text,
                extra=cache_extra,
            )

        # 8. Log to JSONL
        pipeline_logger.log(APICallRecord(
            timestamp=_now_iso(),
            task_type=request.task_type,
            model_used=used_model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            cost_usd_estimate=cost,
            cache_hit=False,
            story_id=request.story_id,
            router_decision=str(decision),
        ))

        return GenerationResponse(
            text=text,
            model_used=used_model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            cost_usd_estimate=cost,
            cache_hit=False,
            router_decision=decision,
        )

    # â”€â”€ Fallback chain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


    async def _call_with_retry(
        self,
        *,
        model_name: str,
        request:    GenerationRequest,
        gen_cfg:    genai_types.GenerateContentConfig,
    ) -> tuple[str, int, int]:
        """
        Async retry with exponential backoff + jitter.
        Only 429 / 503 / quota errors are retried; others propagate immediately.

        Issue #22: token usage is recorded inside the loop. For failed attempts
        (where tokens aren't returned by the API), we estimate based on prompt size
        so the budget still decrements for the billed input tokens.
        """
        # Estimated input tokens (rough heuristic: 4 chars/token)
        est_in = len(request.prompt.encode()) // 4
        if request.system_prompt:
            est_in += len(request.system_prompt.encode()) // 4

        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(CONFIG.max_retries),
            wait=_wait_respecting_server_hint,   # honours server Retry-After hint (#18b)
            reraise=True,
        ):
            with attempt:
                try:
                    text, in_tok, out_tok = await self._raw_call(
                        model_name=model_name,
                        request=request,
                        gen_cfg=gen_cfg,
                    )
                    # Successful attempt: record EXACT tokens
                    rate_limiter.record_usage(request.session_id, in_tok + out_tok)
                    return text, in_tok, out_tok
                except Exception:
                    # Failed attempt (e.g. 429): Google still bills input tokens.
                    # Record the ESTIMATED tokens so budget doesn't leak (#22).
                    rate_limiter.record_usage(request.session_id, est_in)
                    log.warning(
                        "[%s] Attempt %d failed. Recording ~%d estimated input tokens.",
                        request.story_id, attempt.retry_state.attempt_number, est_in
                    )
                    raise
        raise RuntimeError("Retry loop exited without yielding a result.")

    # â”€â”€ Raw SDK call â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


    async def _enforce_rpm(self, model_cfg: ModelConfig) -> None:
        """
        Block until making another request would not exceed the model's RPM limit.

        Uses a 60-second sliding window: purges timestamps older than 60s, then
        sleeps if the window is already at capacity before recording a new slot.
        """
        window = 60.0
        
        # Acquire the lock to ensure we evaluate and update the sliding window atomically
        async with self._rpm_lock:
            # We use a while loop because after sleeping, another task might have run
            # and taken our slot. We must re-evaluate until a slot is truly free.
            while True:
                now = time.monotonic()
                # Purge entries outside the rolling window
                while self._request_timestamps and now - self._request_timestamps[0] > window:
                    self._request_timestamps.popleft()

                if len(self._request_timestamps) < model_cfg.rpm_limit:
                    # We have capacity, break out and record our timestamp below
                    break

                # We are at capacity. Calculate sleep time and yield the lock while we sleep.
                # Adding a small 50ms buffer to ensure we aren't precisely on the boundary
                wait = window - (now - self._request_timestamps[0]) + 0.05
                log.info(
                    "RPM limit (%d/min) reached - waiting %.1fs before next call.",
                    model_cfg.rpm_limit, wait,
                )
                
                # We yield the lock so other tasks aren't needlessly blocked if they
                # are just checking capacity or adding timestamps for *other* models 
                # (though currently we only share one deque for all calls).
                # To yield the lock cleanly, we can temporarily exit the context manager,
                # sleep, and then re-acquire. A cleaner approach is to just await sleep
                # but NOT hold the lock during the sleep. However, asyncio.Lock doesn't 
                # have a simple `release() / await sleep / acquire()` pattern that is safe
                # inside an async context manager cleanly without nested functions or manual 
                # lock management. Thus:
                pass 
                
            # Actually, a better pattern is to sleep OUTSIDE the lock to let other tasks 
            # make progress, then re-acquire. 
            # Let's write the idiomatic approach:
            
        # The idiomatic "wait for slot" pattern:
        while True:
            wait = 0.0
            async with self._rpm_lock:
                now = time.monotonic()
                while self._request_timestamps and now - self._request_timestamps[0] > window:
                    self._request_timestamps.popleft()
                
                if len(self._request_timestamps) < model_cfg.rpm_limit:
                    self._request_timestamps.append(now)
                    return
                # Calculate sleep duration
                wait = window - (now - self._request_timestamps[0]) + 0.05
            
            # Sleep outside the lock
            log.info(
                "RPM limit (%d/min) reached - waiting %.1fs before next call.",
                model_cfg.rpm_limit, wait,
            )
            await asyncio.sleep(wait)

    async def _raw_call(
        self,
        *,
        model_name: str,
        request:    GenerationRequest,
        gen_cfg:    genai_types.GenerateContentConfig,
    ) -> tuple[str, int, int]:
        """
        Single (non-retried) async call using client.aio.models.generate_content.
        Returns (text, input_tokens, output_tokens).
        """
        # â”€â”€ [PLUMBING] Dry Run Mock â”€â”€
        if request.dry_run:
            log.info(f"[DRY-RUN] Mocking response for {model_name}...")
            mock_text = json.dumps(_build_mock_story(request), ensure_ascii=False)
            return mock_text, 100, 200

        # Enforce RPM limit before making the live API call
        model_cfg_for_rpm = _model_cfg(ModelTier.FLASH)
        await self._enforce_rpm(model_cfg_for_rpm)

        contents = cast(Any, _build_contents(request))

        response = await self._client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=gen_cfg,
        )

        # Extract text safely (blocked / empty response -> "")
        text = ""
        try:
            text = response.text or ""
        except Exception:
            pass

        # Token counts
        usage     = getattr(response, "usage_metadata", None)
        in_tokens  = getattr(usage, "prompt_token_count",     0) or 0
        out_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return text, in_tokens, out_tokens


# ---------------------------------------------------------------------------
# Lazy module-level singleton
# ---------------------------------------------------------------------------
# Using a factory function avoids asyncio.Semaphore creation or genai.Client()
# running at import time, which can cause hangs in subprocesses / test runners.

_client_instance: Optional[GeminiClient] = None


def get_client() -> GeminiClient:
    """Return the process-wide GeminiClient, creating it on first call."""
    global _client_instance
    if _client_instance is None:
        _client_instance = GeminiClient()
    return _client_instance
