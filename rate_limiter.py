"""
rate_limiter.py — Per-session token budget tracking.

Strategy (from the guide §5)
    - Track tokens used per session/user in real time
    - At SOFT_LIMIT (80% of budget) → force Flash downgrade before hitting
      the hard quota
    - Persist budget state to disk so it survives process restarts within
      the same billing window

Usage
-----
    from rate_limiter import rate_limiter

    if rate_limiter.should_downgrade(session_id):
        tier = ModelTier.FLASH          # cheap model

    ... after call ...
    rate_limiter.record_usage(session_id, input_tokens + output_tokens)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from config import CONFIG

log = logging.getLogger(__name__)

# Budget auto-resets after this window (prevents cross-day accumulation)
BUDGET_WINDOW_SECONDS: float = 86_400  # 24 hours


# ---------------------------------------------------------------------------
# Session Budget Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SessionBudget:
    session_id:   str
    token_budget: int
    tokens_used:  int   = 0
    image_requests_today: int = 0
    image_quota_exceeded: bool = False
    last_reset:   float = field(default_factory=time.time)

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    @property
    def usage_fraction(self) -> float:
        if self.token_budget <= 0:
            return 1.0
        return self.tokens_used / self.token_budget

    @property
    def is_exhausted(self) -> bool:
        return self.tokens_remaining == 0

    @property
    def at_soft_limit(self) -> bool:
        return self.usage_fraction >= CONFIG.soft_limit_fraction

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionBudget":
        # Handle legacy files missing image fields
        if "image_requests_today" not in data:
            data["image_requests_today"] = 0
        if "image_quota_exceeded" not in data:
            data["image_quota_exceeded"] = False
        return cls(**data)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Tracks token and image usage per session and enforces budgets:

    [Text]
    soft limit (80%)  → route to Flash instead of Pro
    hard limit (100%) → raise BudgetExhaustedError

    [Images]
    Daily limit (RPD) → prevents further calls once reached
    Circuit Breaker   → trips if a "Quota Exceeded" error is received

    State is persisted to ``logs/rate_limits.json``.
    """

    def __init__(
        self,
        default_budget: int = CONFIG.default_token_budget_per_session,
        state_file:     str = CONFIG.rate_limit_state_file,
    ) -> None:
        self.default_budget = default_budget
        self.state_file     = state_file
        self._sessions: dict[str, SessionBudget] = {}
        self._lock = threading.RLock()

        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        self._load_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_or_create(self, session_id: str) -> SessionBudget:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionBudget(
                    session_id=session_id,
                    token_budget=self.default_budget,
                )
            else:
                # Auto-reset if the 24-hour budget window has expired.
                budget = self._sessions[session_id]
                if time.time() - budget.last_reset > BUDGET_WINDOW_SECONDS:
                    log.info(
                        "Session '%s' budget window expired — resetting.",
                        session_id
                    )
                    self._sessions[session_id] = SessionBudget(
                        session_id=session_id,
                        token_budget=self.default_budget,
                    )
                    self._save_state()
            return self._sessions[session_id]

    # ── Text Budget ───────────────────────────────────────────────────────────

    def should_downgrade(self, session_id: str) -> bool:
        """Return True when session has used ≥80% of its token budget."""
        return self.get_or_create(session_id).at_soft_limit

    def is_exhausted(self, session_id: str) -> bool:
        return self.get_or_create(session_id).is_exhausted

    def record_usage(self, session_id: str, tokens: int) -> SessionBudget:
        """Record token consumption for a session."""
        with self._lock:
            session = self.get_or_create(session_id)
            session.tokens_used += tokens
            self._save_state()
            return session

    # ── Image Quota & Circuit Breaker ────────────────────────────────────────

    def check_imagen_quota(self, session_id: str) -> bool:
        """
        Returns True if image generation is allowed.
        False if daily RPD limit is reached or circuit breaker is tripped.
        """
        with self._lock:
            session = self.get_or_create(session_id)
            if session.image_quota_exceeded:
                return False
            if session.image_requests_today >= CONFIG.imagen_rpd_limit:
                log.warning("Daily Imagen RPD limit (%d) reached.", CONFIG.imagen_rpd_limit)
                return False
            return True

    def record_image_request(self, session_id: str) -> None:
        """Increment the daily image request counter."""
        with self._lock:
            session = self.get_or_create(session_id)
            session.image_requests_today += 1
            self._save_state()

    def trip_image_circuit_breaker(self, session_id: str, reason: str) -> None:
        """Stop all further image requests for this session."""
        with self._lock:
            session = self.get_or_create(session_id)
            if not session.image_quota_exceeded:
                session.image_quota_exceeded = True
                log.error("IMAGE CIRCUIT BREAKER TRIPPED for session %s: %s", session_id, reason)
                self._save_state()

    # ── Management ────────────────────────────────────────────────────────────

    def reset_session(self, session_id: str, new_budget: Optional[int] = None) -> None:
        """Reset a session's usage counter."""
        with self._lock:
            budget = new_budget or self.default_budget
            self._sessions[session_id] = SessionBudget(
                session_id=session_id,
                token_budget=budget,
            )
            self._save_state()

    def status_report(self) -> list[dict]:
        """Return a list of all session budget statuses."""
        with self._lock:
            return [
                {
                    "session_id":       s.session_id,
                    "tokens_used":      s.tokens_used,
                    "token_budget":     s.token_budget,
                    "usage_pct":        f"{s.usage_fraction * 100:.1f}%",
                    "images_today":     s.image_requests_today,
                    "images_rpd_limit": CONFIG.imagen_rpd_limit,
                    "circuit_breaker":  s.image_quota_exceeded,
                    "exhausted":        s.is_exhausted,
                }
                for s in self._sessions.values()
            ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for sid, sdata in data.items():
                self._sessions[sid] = SessionBudget.from_dict(sdata)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass  # corrupt state → start fresh

    def _save_state(self) -> None:
        try:
            payload = {sid: s.to_dict() for sid, s in self._sessions.items()}
            with open(self.state_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except OSError:
            pass  # non-fatal


class BudgetExhaustedError(Exception):
    """Raised when a session has consumed 100% of its token budget."""


# Module-level singleton
rate_limiter = RateLimiter()
