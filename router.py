"""
router.py — Pre-router for classifying request complexity.

Uses rule-based heuristics to route requests to appropriate model tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rate_limiter import rate_limiter


class ModelTier(str, Enum):
    FLASH  = "flash"
    IMAGEN = "imagen"



@dataclass
class RouterDecision:
    tier:       ModelTier
    reason:     str
    confidence: float   # 0.0 â€“ 1.0

    def __str__(self) -> str:
        return f"[{self.tier.upper()}] {self.reason} (conf={self.confidence:.0%})"



# Strong specific verbs for image generation
_IMAGE_VERBS: tuple[str, ...] = (
    "generate", "create", "draw", "illustrate", "render",
    "visualize", "sketch", "paint", "produce",
)

# Nouns that must follow or precede the verb for an "intent" match
_IMAGE_NOUNS: tuple[str, ...] = (
    "image", "picture", "artwork", "illustration", "scene",
    "background", "cover", "graphic", "art",
)

# Exclusion list to prevent narrative triggers
_IMAGE_EXCLUSIONS: tuple[str, ...] = (
    "story about", "tale of", "imagine a", "talk about",
)


_FLASH_NATIVE_TASKS: frozenset[str] = frozenset({
    "story_generation",
    "classification",
    "summarization",
    "intent_detection",
    "image_prompt_build",
    "simple_qa",
})


class RequestRouter:
    """
    Stateless rule-based pre-router.

    All routing decisions are logged to RouterDecision objects so the
    caller can attach `.reason` to the API call log record.
    """

    def route(
        self,
        *,
        user_text:  str,
        task_type:  str                  = "story_generation",
        session_id: str                  = "default",
        force_tier: Optional[ModelTier]  = None,
    ) -> RouterDecision:
        """
        Classify the request and return a RouterDecision.

        Parameters
        ----------
        user_text
            Raw user prompt (not the system prompt).
        task_type
            Logical task category (e.g. "story_generation").
        session_id
            Used to check rate-limit budget state.
        force_tier
            Bypass rules and return this tier directly (for tests / override).
        """
        if force_tier is not None:
            return RouterDecision(
                tier=force_tier,
                reason="caller-forced tier override",
                confidence=1.0,
            )

        text_lower = user_text.lower()

        # Check for image intent based on verb/noun matches avoiding exclusions
        is_image_intent = False

        # Verb + Noun combination (e.g. "generate image", "draw picture")
        has_verb = any(v in text_lower for v in _IMAGE_VERBS)
        has_noun = any(n in text_lower for n in _IMAGE_NOUNS)
        has_exclusion = any(ex in text_lower for ex in _IMAGE_EXCLUSIONS)

        if (has_verb and has_noun) and not has_exclusion:
            is_image_intent = True

        # Direct signals that are unambiguous
        if "generate_image" in text_lower or "call_imagen" in text_lower:
            is_image_intent = True

        if is_image_intent:
            return RouterDecision(
                tier=ModelTier.IMAGEN,
                reason="strict image-generation intent detected (verb+noun match)",
                confidence=0.95,
            )

        if rate_limiter.should_downgrade(session_id):
            return RouterDecision(
                tier=ModelTier.FLASH,
                reason=f"session '{session_id}' at ≥80% token budget — downgraded to Flash",
                confidence=1.0,
            )

        if task_type in _FLASH_NATIVE_TASKS:
            return RouterDecision(
                tier=ModelTier.FLASH,
                reason=f"task type '{task_type}' is Flash-native",
                confidence=0.95,
            )

        return RouterDecision(
            tier=ModelTier.FLASH,
            reason="default Flash routing (no Pro/Imagen signals detected)",
            confidence=0.60,
        )


# Module-level singleton
router = RequestRouter()
