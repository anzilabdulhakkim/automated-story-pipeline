"""
config.py — Central pipeline configuration.

All tunables live here. Import CONFIG from this module; never hardcode
model names, token limits, or costs anywhere else in the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _required_env(var_name: str) -> str:
    """Return non-empty env var or raise a clear configuration error."""
    value = (os.getenv(var_name, "") or "").strip()
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var_name}")
    return value


def _env_model_name(var_name: str) -> str:
    """Read model name from env and normalize optional 'models/' prefix."""
    return _required_env(var_name).removeprefix("models/")


# ---------------------------------------------------------------------------
# Model Specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Immutable spec for a single model variant."""
    name: str
    max_output_tokens: int
    # USD per 1 000 tokens (used only for cost estimates — not billing)
    cost_per_1k_input: float
    cost_per_1k_output: float
    # Requests-per-minute quota (free tier defaults; update for paid tier)
    rpm_limit: int

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return estimated USD cost for a single call."""
        return (
            (input_tokens  / 1_000) * self.cost_per_1k_input
            + (output_tokens / 1_000) * self.cost_per_1k_output
        )


# Gemini 3 Flash Preview — workhorse model (fast, multimodal)
# Pricing (paid tier, as of 2026): $0.50/1M input, $3.00/1M output
# Free tier: $0/1M (capped, not accurate for cost tracking)
FLASH_MODEL = ModelConfig(
    name=_env_model_name("GEMINI_TEXT_MODEL"),
    max_output_tokens=8_192,
    cost_per_1k_input=0.000_500,   # $0.50 per 1M input tokens (paid tier)
    cost_per_1k_output=0.003_000,  # $3.00 per 1M output tokens (paid tier)
    rpm_limit=int(os.getenv("GEMINI_TEXT_RPM_LIMIT", "2000"))
)

# Gemini 1.5 Pro — used only for complex / long-context tasks
# Pricing: $1.25 / 1M input, $5.00 / 1M output (≤128K ctx)


# ---------------------------------------------------------------------------
# Pipeline-wide Settings
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # ── Model registry ──────────────────────────────────────────────────────
    flash_model: ModelConfig = field(default_factory=lambda: FLASH_MODEL)
    imagen_model: str        = field(default_factory=lambda: _env_model_name("GEMINI_IMAGE_MODEL"))

    # ── API keys (loaded from .env) ──────────────────────────────────────────
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # ── Retry / backoff ──────────────────────────────────────────────────────
    max_retries:     int   = 3
    retry_min_wait:  float = 1.0   # seconds
    retry_max_wait:  float = 60.0  # seconds (caps exponential growth)
    retry_jitter:    float = 2.0   # seconds of random jitter added on top

    # ── Response cache ───────────────────────────────────────────────────────
    cache_dir:         str = ".cache"
    cache_ttl_seconds: int = 7 * 24 * 3_600  # 7 days — story content is deterministic, no staleness

    # ── Rate-limit budgets ───────────────────────────────────────────────────
    # Tokens per session before soft-limit kicks in (→ forces Flash)
    # Raised to 500k for paid tier: 100 stories × ~5k tokens = ~500k peak
    default_token_budget_per_session: int = 500_000
    # Fraction of budget that triggers soft-limit Flash downgrade
    soft_limit_fraction: float = 0.80

    # ── Concurrency ──────────────────────────────────────────────────────────
    # Max simultaneous Gemini calls in the async batch runner
    max_concurrent_requests: int = 3

    # ── Conversation history ─────────────────────────────────────────────────
    # Raw turns to keep before summarising older ones into a context block
    max_history_turns: int = 5

    # ── Logging ──────────────────────────────────────────────────────────────
    log_dir:             str = "logs"
    configs_dir:         str = "configs"
    analytics_dir:       str = "analytics"
    output_dir:          str = "documents"
    story_images_dir:     str = "story_images"
    api_call_log_file:   str = "logs/api_calls.jsonl"
    rate_limit_state_file: str = "logs/rate_limits.json"
    
    # ── Imagen Quotas ────────────────────────────────────────────────────────
    # Defaults to Tier 1 limits (10 RPM, 70 RPD). Override via .env if needed.
    imagen_rpm_limit: int = int(os.getenv("IMAGEN_RPM_LIMIT", "10"))
    imagen_rpd_limit: int = int(os.getenv("IMAGEN_RPD_LIMIT", "70"))

    # ── Router thresholds ────────────────────────────────────────────────────
    # Prompts whose estimated token count exceeds this are routed to Pro
    pro_token_threshold: int = 500

    # ── Output control ───────────────────────────────────────────────────────
    # Stop sequences are NOT used for JSON-enforced tasks (response_mime_type
    # handles termination). Kept empty to avoid accidental truncation.
    default_stop_sequences: list = field(default_factory=list)

    # ── Task-specific max_output_tokens overrides ────────────────────────────
    max_tokens_by_task: dict = field(default_factory=lambda: {
        "story_generation":   8_192,
        "summarization":        512,
        "classification":        32,
        "intent_detection":      64,
        "image_prompt_build":   256,
    })

    def get_max_tokens(self, task_type: str) -> int:
        return self.max_tokens_by_task.get(task_type, self.flash_model.max_output_tokens)


# Singleton — import this everywhere
CONFIG = PipelineConfig()
