"""
router.py â€” Pre-router: classify request complexity before hitting a model.

Decision tree (cheapest path first)
    1. Image-generation intent â†’ Imagen
    2. Session at soft token-budget limit â†’ Flash (cost guardrail)
    3. Forced tier override â†’ use it directly
    4. Pro complexity signals in prompt â†’ Pro
    5. Estimated token count exceeds threshold â†’ Pro
    6. Task type is a Flash-native task â†’ Flash
    7. Default â†’ Flash

Rule-based routing is ~zero latency and ~zero cost.  For high-volume
production use you can replace or augment the rules with a tiny classifier
model (running locally or via a single cheap embeddings call).

Usage
-----
    from router import router, ModelTier

    decision = router.route(
        user_text  = "Write a bedtime story for Olivia",
        task_type  = "story_generation",
        session_id = "batch1-story-0",
    )
    # decision.tier    â†’ ModelTier.FLASH
    # decision.reason  â†’ "task type suitable for Flash"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import CONFIG
from rate_limiter import rate_limiter


# ---------------------------------------------------------------------------
# Model Tier Enum
# ---------------------------------------------------------------------------

class ModelTier(str, Enum):
    FLASH  = "flash"
    IMAGEN = "imagen"


# ---------------------------------------------------------------------------
# Router Decision Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RouterDecision:
    tier:       ModelTier
    reason:     str
    confidence: float   # 0.0 â€“ 1.0

    def __str__(self) -> str:
        return f"[{self.tier.upper()}] {self.reason} (conf={self.confidence:.0%})"


# ---------------------------------------------------------------------------
# Signal Lists
# ---------------------------------------------------------------------------

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

# Any of these phrases suggest complex reasoning that warrants Pro.
_PRO_SIGNALS: tuple[str, ...] = (
    "analyze", "analyse", "compare and contrast", "synthesize",
    "evaluate", "critique", "multi-step", "in-depth analysis",
    "detailed analysis", "explain why", "long document",
    "provide a thorough", "comprehensive overview",
    "chain of thought",
)

# Task types natively suited to Flash (fast, simple, short outputs).
_FLASH_NATIVE_TASKS: frozenset[str] = frozenset({
    "story_generation",
    "classification",
    "summarization",
    "intent_detection",
    "image_prompt_build",
    "simple_qa",
})


# ---------------------------------------------------------------------------
# Token Estimator
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token â‰ˆ 4 characters (BPE heuristic)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

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
        # â”€â”€ 0. Forced override â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if force_tier is not None:
            return RouterDecision(
                tier=force_tier,
                reason="caller-forced tier override",
                confidence=1.0,
            )

        text_lower = user_text.lower()

        # â”€â”€ 1. Image intent (Strict Check) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # To prevent accidental routing when "image" or "draw" is in a story,
        # we check for Verb + Noun combinations and apply an exclusion list.
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

        # â”€â”€ 2. Session soft-limit guardrail â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if rate_limiter.should_downgrade(session_id):
            return RouterDecision(
                tier=ModelTier.FLASH,
                reason=f"session '{session_id}' at â‰¥80% token budget â€” downgraded to Flash",
                confidence=1.0,
            )

        # â”€â”€ 3. Complexity signals in text â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Complex prompts are still served by the configured text model.
        triggered = [sig for sig in _PRO_SIGNALS if sig in text_lower]
        if triggered:
            return RouterDecision(
                tier=ModelTier.FLASH,
                reason=f"complexity signal(s) detected â€” using text model: {triggered[:3]}",
                confidence=0.85,
            )

        # â”€â”€ 4. Prompt length heuristic â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Long prompts stay on the configured text model.
        est_tokens = _estimate_tokens(user_text)
        if est_tokens > CONFIG.pro_token_threshold:
            return RouterDecision(
                tier=ModelTier.FLASH,
                reason=f"prompt â‰ˆ{est_tokens} tokens > threshold ({CONFIG.pro_token_threshold}) â€” using text model",
                confidence=0.75,
            )

        # â”€â”€ 5. Flash-native task types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if task_type in _FLASH_NATIVE_TASKS:
            return RouterDecision(
                tier=ModelTier.FLASH,
                reason=f"task type '{task_type}' is Flash-native",
                confidence=0.95,
            )

        # â”€â”€ 6. Default â†’ Flash (cheapest) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        return RouterDecision(
            tier=ModelTier.FLASH,
            reason="default Flash routing (no Pro/Imagen signals detected)",
            confidence=0.60,
        )


# Module-level singleton
router = RequestRouter()
