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
        return cls(**data)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Tracks token usage per session and enforces a two-level budget:

    soft limit (80%)  → route to Flash instead of Pro
    hard limit (100%) → raise BudgetExhaustedError

    State is persisted to ``logs/rate_limits.json`` so it is maintained
    across batch restarts within the same billing window.
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
                # Prevents a run on day N from consuming day N+1's quota.
                budget = self._sessions[session_id]
                if time.time() - budget.last_reset > BUDGET_WINDOW_SECONDS:
                    log.info(
                        "Session '%s' budget window expired — resetting to %d tokens.",
                        session_id, self.default_budget,
                    )
                    self._sessions[session_id] = SessionBudget(
                        session_id=session_id,
                        token_budget=self.default_budget,
                    )
                    self._save_state()
            return self._sessions[session_id]

    def should_downgrade(self, session_id: str) -> bool:
        """Return True when session has used ≥80% of its token budget."""
        return self.get_or_create(session_id).at_soft_limit

    def is_exhausted(self, session_id: str) -> bool:
        return self.get_or_create(session_id).is_exhausted

    def record_usage(self, session_id: str, tokens: int) -> SessionBudget:
        """
        Record token consumption for a session.
        Returns the updated SessionBudget so callers can check limits.
        """
        with self._lock:
            session = self.get_or_create(session_id)
            session.tokens_used += tokens
            self._save_state()
            return session

    def reset_session(self, session_id: str, new_budget: Optional[int] = None) -> None:
        """Reset a session's usage counter (e.g. at billing window rollover)."""
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
                    "tokens_remaining": s.tokens_remaining,
                    "usage_pct":        f"{s.usage_fraction * 100:.1f}%",
                    "at_soft_limit":    s.at_soft_limit,
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
