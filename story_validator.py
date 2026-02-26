"""
story_validator.py - deterministic runtime validation for generated stories.

The model prompt includes strong constraints, but production systems must
enforce critical contracts in code before downstream steps (docs/images).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


SAFETY_SUFFIX = "--no text --no words --no letters"


@dataclass(frozen=True)
class AgeSpec:
    min_age: int
    max_age: int
    total_pages: int
    min_words_per_page: int
    max_words_per_page: int


AGE_SPECS: tuple[AgeSpec, ...] = (
    AgeSpec(3, 5, 8, 25, 50),
    AgeSpec(6, 8, 12, 40, 100),
    AgeSpec(9, 12, 18, 75, 120),
)


def normalize_age(value: Any) -> int:
    """Clamp age to supported app range [3, 12]."""
    try:
        age = int(value)
    except (TypeError, ValueError):
        age = 5
    return max(3, min(12, age))


def _spec_for_age(age: int) -> AgeSpec:
    for spec in AGE_SPECS:
        if spec.min_age <= age <= spec.max_age:
            return spec
    return AGE_SPECS[0]


def _count_words(text: str) -> int:
    # Tokenizer-independent approximation for business-rule validation.
    return len(re.findall(r"[A-Za-z0-9']+", text))


def normalize_story_metadata(story_data: dict[str, Any]) -> None:
    """
    Normalize non-semantic numeric metadata in-place.

    This prevents false failures when models return slightly inconsistent
    `word_count`/`character_count` fields while page text is otherwise valid.
    """
    pages = story_data.get("pages")
    if not isinstance(pages, list):
        return

    for page in pages:
        if not isinstance(page, dict):
            continue
        text = page.get("text", "")
        if not isinstance(text, str):
            text = ""
        page["word_count"] = _count_words(text)
        page["character_count"] = len(text)


def _trim_to_word_limit(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]).strip()


def _padding_sentence_for_age(age: int) -> str:
    if age <= 5:
        return "They took turns, used gentle words, and smiled together."
    if age <= 8:
        return "They listened carefully, solved the problem fairly, and felt proud of their teamwork."
    return (
        "They considered different viewpoints, accepted responsibility for their choices, "
        "and worked together toward a thoughtful, hopeful resolution."
    )


def enforce_image_prompts(story_data: dict[str, Any], input_config: dict[str, Any]) -> None:
    """
    Deterministically fix common image prompt issues (safety suffix, character descriptor).
    """
    pages = story_data.get("pages")
    if not isinstance(pages, list):
        return

    target_age = normalize_age(input_config.get("target_age", 5))
    char_desc = str(story_data.get("character_description", "")).strip()
    required_descriptor = f"a {target_age}-year-old child, {char_desc}".strip()

    # 1. Fix Cover Prompt
    cover_prompt = story_data.get("cover_image_prompt", "")
    if isinstance(cover_prompt, str) and cover_prompt.strip():
        if SAFETY_SUFFIX not in cover_prompt:
            cover_prompt = f"{cover_prompt.strip()} {SAFETY_SUFFIX}"
        if char_desc and required_descriptor not in cover_prompt:
            cover_prompt = f"{required_descriptor}, {cover_prompt.strip()}"
        story_data["cover_image_prompt"] = cover_prompt

    # 2. Fix Page Prompts
    for page in pages:
        if not isinstance(page, dict):
            continue
        prompt = page.get("image_prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        
        if SAFETY_SUFFIX not in prompt:
            prompt = f"{prompt.strip()} {SAFETY_SUFFIX}"
        if char_desc and required_descriptor not in prompt:
            prompt = f"{required_descriptor}, {prompt.strip()}"
        page["image_prompt"] = prompt


def enforce_word_bounds(story_data: dict[str, Any], input_config: dict[str, Any]) -> None:
    """
    Deterministically enforce per-page word bounds in-place.

    This avoids wasting API calls when the model violates length constraints
    despite otherwise valid structure/content.
    """
    enforce_image_prompts(story_data, input_config) # Fix prompts too

    pages = story_data.get("pages")
    if not isinstance(pages, list):
        return

    age = normalize_age(input_config.get("target_age", 5))
    spec = _spec_for_age(age)
    pad_sentence = _padding_sentence_for_age(age)

    for page in pages:
        if not isinstance(page, dict):
            continue
        text = page.get("text", "")
        if not isinstance(text, str):
            text = ""
        text = text.strip()
        if not text:
            text = pad_sentence

        while _count_words(text) < spec.min_words_per_page:
            text = f"{text} {pad_sentence}".strip()

        if _count_words(text) > spec.max_words_per_page:
            text = _trim_to_word_limit(text, spec.max_words_per_page)

        page["text"] = text

    normalize_story_metadata(story_data)


def validate_story_output(story_data: dict[str, Any], input_config: dict[str, Any]) -> list[str]:
    """
    Validate runtime-critical contracts for story JSON.

    Returns:
        list[str]: Human-readable validation errors. Empty means valid.
    """
    errors: list[str] = []

    if not isinstance(story_data, dict):
        return ["Output must be a JSON object."]

    target_age = normalize_age(input_config.get("target_age", 5))
    spec = _spec_for_age(target_age)

    target_age_confirmation = normalize_age(story_data.get("target_age_confirmation", target_age))
    if target_age_confirmation != target_age:
        errors.append(
            f"target_age_confirmation mismatch: expected {target_age}, got {story_data.get('target_age_confirmation')!r}"
        )

    total_pages = story_data.get("total_pages")
    if total_pages != spec.total_pages:
        errors.append(f"total_pages invalid for age {target_age}: expected {spec.total_pages}, got {total_pages!r}")

    pages = story_data.get("pages")
    if not isinstance(pages, list):
        errors.append("pages must be an array.")
        pages = []

    if len(pages) != spec.total_pages:
        errors.append(f"pages length mismatch: expected {spec.total_pages}, got {len(pages)}")

    character_description = story_data.get("character_description")
    if not isinstance(character_description, str) or not character_description.strip():
        errors.append("character_description missing or empty.")
        character_description = ""

    required_descriptor = f"a {target_age}-year-old child, {character_description}".strip()

    for expected_page_num, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            errors.append(f"pages[{expected_page_num}] must be an object.")
            continue

        page_number = page.get("page_number")
        if page_number != expected_page_num:
            errors.append(f"page_number sequence mismatch at index {expected_page_num}: got {page_number!r}")

        text = page.get("text", "")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"page {expected_page_num}: text missing or empty.")
            text = ""

        actual_word_count = _count_words(text)
        actual_char_count = len(text)

        if not (spec.min_words_per_page <= actual_word_count <= spec.max_words_per_page):
            errors.append(
                f"page {expected_page_num}: word_count out of range "
                f"({actual_word_count}, expected {spec.min_words_per_page}-{spec.max_words_per_page})"
            )

        image_prompt = page.get("image_prompt", "")
        if not isinstance(image_prompt, str) or not image_prompt.strip():
            errors.append(f"page {expected_page_num}: image_prompt missing or empty.")
        else:
            if SAFETY_SUFFIX not in image_prompt:
                errors.append(f"page {expected_page_num}: image_prompt missing safety suffix.")
            if character_description and required_descriptor not in image_prompt:
                errors.append(f"page {expected_page_num}: image_prompt missing exact character descriptor.")

    cover_image_prompt = story_data.get("cover_image_prompt", "")
    if not isinstance(cover_image_prompt, str) or not cover_image_prompt.strip():
        errors.append("cover_image_prompt missing or empty.")
    else:
        if SAFETY_SUFFIX not in cover_image_prompt:
            errors.append("cover_image_prompt missing safety suffix.")
        if character_description and required_descriptor not in cover_image_prompt:
            errors.append("cover_image_prompt missing exact character descriptor.")

    nickname = str(input_config.get("user_nickname", "")).strip()
    if nickname and pages:
        combined_text = " ".join(
            p.get("text", "")
            for p in pages
            if isinstance(p, dict)
        )
        mentions = len(re.findall(rf"\b{re.escape(nickname)}\b", combined_text))
        if mentions < 3:
            errors.append(f"user_nickname mention count too low: expected >=3, got {mentions}")

    return errors
