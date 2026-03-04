"""
image_generator.py — Imagen integration for batch cover and scene image generation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Optional

from google import genai
from google.genai import types as genai_types

from config import CONFIG
from router import ModelTier, router

log = logging.getLogger(__name__)

# Template prevents arbitrary prompt injection
IMAGE_PROMPT_TEMPLATE = (
    "A highly detailed, professional {art_style} illustration for a children's book. "
    "Scene: {scene_description}. "
    "Tone: {tone}. "
    "No text, no watermarks, vibrant colors."
)

_image_generation_disabled_reason: Optional[str] = None


def _normalize_error_text(exc: Exception) -> str:
    return " ".join(str(exc).split()).lower()


def _is_rate_limit_error(error_text: str) -> bool:
    return (
        "429" in error_text
        or "resource_exhausted" in error_text
        or "rate limit" in error_text
    )


def _is_non_retryable_error(error_text: str) -> bool:
    if _is_rate_limit_error(error_text):
        return False

    if "only accessible to billed users" in error_text:
        return True
    if "permission_denied" in error_text or "forbidden" in error_text:
        return True
    if "invalid_argument" in error_text:
        return True
    if "enum value is not supported" in error_text:
        return True
    if "is not supported in gemini api" in error_text:
        return True
    if "person_generation parameter is not supported" in error_text:
        return True

    status_codes = re.findall(r"\b\d{3}\b", error_text)
    for code_str in status_codes:
        code = int(code_str)
        if 400 <= code < 500 and code != 429:
            return True
    return False


def _disable_image_generation(reason: str) -> None:
    global _image_generation_disabled_reason
    if _image_generation_disabled_reason is None:
        _image_generation_disabled_reason = reason
        log.warning("Image generation disabled for this run: %s", reason)


def _sanitize_image_prompt(prompt: str, *, is_cover: bool) -> str:
    """
    Reduce common text/number artifacts that image models may add to covers.
    """
    prompt = re.sub(r"\s--no\s(text|words|letters)\b", "", prompt, flags=re.IGNORECASE).strip()
    if is_cover:
        prompt = re.sub(r"\bbook cover\b", "storybook cover illustration", prompt, flags=re.IGNORECASE)
        prompt = f"{prompt}, centered composition, clean background space near top"
    no_text_clause = (
        "no visible text, no letters, no words, no numbers, no symbols, "
        "no logos, no watermark, no signature"
    )
    return f"{prompt}, {no_text_clause}"



_imagen_client: "Optional[genai.Client]" = None
_image_semaphore: "Optional[asyncio.Semaphore]" = None


def _get_imagen_client() -> "genai.Client":
    """Lazy-init a single shared Gemini client for image generation."""
    global _imagen_client
    if _imagen_client is None:
        if not CONFIG.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        _imagen_client = genai.Client(
            api_key=CONFIG.gemini_api_key,
            http_options=genai_types.HttpOptions(
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )
    return _imagen_client


async def generate_image(
    scene_description: str,
    config: dict[str, Any],
    session_id: str,
    image_dir: Optional[str] = None,
    filename: str = "cover.jpg",
) -> Optional[str]:
    """
    Generate an image for a specific scene if the intent passes the router.
    Returns absolute file path on success, else None.
    """
    global _image_semaphore
    if _image_semaphore is None:
        _image_semaphore = asyncio.Semaphore(CONFIG.max_concurrent_requests)

    if _image_generation_disabled_reason:
        log.info(
            "Skipping image generation for %s (disabled: %s).",
            filename,
            _image_generation_disabled_reason,
        )
        return None

    if image_dir is None:
        image_dir = CONFIG.story_images_dir

    decision = router.route(
        user_text=f"generate image of {scene_description}",
        task_type="image_generation",
        session_id=session_id,
    )
    if decision.tier != ModelTier.IMAGEN:
        log.info("Image skipped by router logic (%s)", decision.reason)
        return None

    # 1. Build prompt from template
    art_style = config.get("art_style", "Disney3D")
    tone = config.get("story_tone", "Cheerful")
    prompt = IMAGE_PROMPT_TEMPLATE.format(
        art_style=art_style,
        scene_description=scene_description,
        tone=tone
    )

    # 2. Cleanup and anti-text hardening
    is_cover = os.path.basename(filename).lower().startswith("cover")
    prompt = _sanitize_image_prompt(prompt, is_cover=is_cover)

    os.makedirs(image_dir, exist_ok=True)
    cache_path = os.path.abspath(os.path.join(image_dir, filename))
    if os.path.exists(cache_path):
        log.info("Image cache hit for %s - returning existing file.", filename)
        return cache_path

    if not CONFIG.gemini_api_key:
        log.warning("Image skipped: GEMINI_API_KEY missing.")
        return None
    client = _get_imagen_client()  # module-level singleton — no new session per call
    max_retries = 3

    for attempt in range(max_retries):
        try:
            image_cfg = genai_types.GenerateContentConfig(
                image_config=genai_types.ImageConfig(
                    aspect_ratio="1:1"  # Only aspect ratio is supported for Gemini image models
                )
            )

            # Gemini models require generate_content, not generate_images
            async with _image_semaphore:
                response = await client.aio.models.generate_content(
                    model=CONFIG.imagen_model,
                    contents=prompt,
                    config=image_cfg,
                )

            if not response.candidates or not response.candidates[0].content.parts:
                log.error("Gemini Image API returned empty content (attempt %s/%s).", attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                continue

            # Extract image bytes from Gemini response by finding the first part with inline_data
            img_bytes = None
            for part in response.candidates[0].content.parts:
                if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                    img_bytes = part.inline_data.data
                    break

            if not img_bytes:
                log.error("Could not extract image bytes from Gemini response (attempt %s/%s).", attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                continue

            with open(cache_path, "wb") as f:
                f.write(img_bytes)
            log.info("Image generated and saved to %s", cache_path)
            return cache_path

        except Exception as e:
            error_text = _normalize_error_text(e)
            if _is_rate_limit_error(error_text):
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
                log.warning(
                    "Image API rate-limited. Waiting %ss before retry %s/%s.",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait)
                continue

            if _is_non_retryable_error(error_text):
                _disable_image_generation(str(e))
                log.warning("Image API non-retryable error. Disabling further image calls for this run.")
                return None

            if attempt < max_retries - 1:
                wait = (attempt + 1) * 3
                log.warning(
                    "Image API error (attempt %s/%s): %s. Retrying in %ss.",
                    attempt + 1,
                    max_retries,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)
            else:
                log.error("Image API failed after %s attempts: %s", max_retries, e)
                return None

    return None
