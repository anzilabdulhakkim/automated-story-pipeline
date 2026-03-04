"""
doc_builder.py - Word document assembly and image generation.

Builds a .docx from a generated story JSON. If image generation is unavailable
for the current project/account, the document still builds with text content.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from google import genai
from google.genai import types as genai_types

from config import CONFIG

log = logging.getLogger(__name__)


_image_client: Optional[genai.Client] = None
_image_generation_disabled_reason: Optional[str] = None


def _get_image_client() -> genai.Client:
    """Lazy-init Gemini client for image generation."""
    global _image_client
    if _image_client is None:
        if not CONFIG.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        _image_client = genai.Client(
            api_key=CONFIG.gemini_api_key,
            http_options=genai_types.HttpOptions(
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )
    return _image_client


# Delay between image generation requests (rate-limit guard).
# Defaults to 4s. Override via IMAGE_DELAY_SECONDS env var.
IMAGE_DELAY_SECONDS = int(os.getenv("IMAGE_DELAY_SECONDS", "4"))


def _normalize_error_text(exc: Exception) -> str:
    return " ".join(str(exc).split()).lower()


def _is_rate_limit_error(error_text: str) -> bool:
    return (
        "429" in error_text
        or "resource_exhausted" in error_text
        or "rate limit" in error_text
        or "quota exceeded" in error_text
    )


def _is_non_retryable_error(error_text: str) -> bool:
    """
    Treat 4xx (except 429) as non-retryable.
    This includes billing/access restrictions and invalid arguments.
    """
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
    Tighten prompt safety to reduce random text/glyph artifacts in generated images.
    """
    # Remove Midjourney-style suffixes
    prompt = re.sub(r"\s--no\s(text|words|letters)\b", "", prompt, flags=re.IGNORECASE).strip()

    if is_cover:
        # "book cover" strongly biases models to invent title typography.
        prompt = re.sub(r"\bbook cover\b", "storybook cover illustration", prompt, flags=re.IGNORECASE)
        prompt = f"{prompt}, centered composition, clean background space near top"

    no_text_clause = (
        "no visible text, no letters, no words, no numbers, no symbols, "
        "no logos, no watermark, no signature"
    )
    return f"{prompt}, {no_text_clause}"


from rate_limiter import rate_limiter  # noqa: E402


def download_page_image(
    prompt: str,
    filename: str,
    image_dir: str = "story_images",
    session_id: str = "default"
) -> Optional[str]:
    """
    Generate a single image via Imagen and save it to image_dir/.

    Returns:
        Absolute file path on success, otherwise None.
    """
    if _image_generation_disabled_reason:
        log.info(
            "Skipping image generation for %s (disabled: %s).",
            filename,
            _image_generation_disabled_reason,
        )
        return None

    # Proactive Quota Check
    if not rate_limiter.check_imagen_quota(session_id):
        _disable_image_generation("Daily RPD quota reached or circuit breaker tripped.")
        return None

    os.makedirs(image_dir, exist_ok=True)
    filepath = os.path.abspath(os.path.join(image_dir, filename))
    if os.path.exists(filepath):
        log.info("Image cache hit for %s - reusing local file.", filename)
        return filepath

    log.info("Generating image: %s", filename)
    print(f"  [IMG] Generating image for: {filename}...")

    # Rate-limit delay
    time.sleep(IMAGE_DELAY_SECONDS)

    is_cover = os.path.basename(filename).lower().startswith("cover")
    prompt = _sanitize_image_prompt(prompt, is_cover=is_cover)

    client = _get_image_client()
    max_retries = 3

    for attempt in range(max_retries):
        # Final safety check before actual API call
        if not rate_limiter.check_imagen_quota(session_id):
            return None

        # Record attempt (dashboard tracks all attempts)
        rate_limiter.record_image_request(session_id)

        try:
            image_cfg = genai_types.GenerateContentConfig(
                image_config=genai_types.ImageConfig(
                    aspect_ratio="1:1"
                )
            )

            # Gemini models require generate_content, not generate_images
            response = client.models.generate_content(
                model=CONFIG.imagen_model,
                contents=prompt,
                config=image_cfg,
            )

            if response.candidates and response.candidates[0].content.parts:
                img_data = None
                for part in response.candidates[0].content.parts:
                    if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                        img_data = part.inline_data.data
                        break

                if img_data:
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    print(f"  [OK] Image saved: {filepath}")
                    return filepath

            print(
                f"  [WARN] No image data in response for {filename}. "
                f"Attempt {attempt + 1}/{max_retries}."
            )
            if attempt < max_retries - 1:
                time.sleep(5)

        except Exception as e:
            error_text = _normalize_error_text(e)

            # Quota Exceeded? Trip the circuit breaker.
            if "quota exceeded" in error_text or "resource_exhausted" in error_text:
                rate_limiter.trip_image_circuit_breaker(session_id, f"API Error: {e}")
                _disable_image_generation(f"Quota exceeded: {e}")
                return None

            if _is_rate_limit_error(error_text):
                # Respect server hint if possible or use exponential backoff
                match = re.search(r'retry in (\d+\.?\d*)', str(e), re.IGNORECASE)
                if match:
                    wait = float(match.group(1)) + 2.0
                else:
                    wait = 30 * (2 ** attempt)  # 30s, 60s, 120s

                print(
                    f"  [WARN] Rate limited (429). Waiting {wait:.1f}s before retry "
                    f"{attempt + 1}/{max_retries}..."
                )
                time.sleep(wait)
                continue

            if _is_non_retryable_error(error_text):
                reason = str(e)
                _disable_image_generation(reason)
                print("  [WARN] Non-retryable image API error.")
                print(f"  [WARN] Disabling further image calls for this run: {reason}")
                return None

            wait = (attempt + 1) * 5
            print(
                f"  [WARN] Error for {filename}: {e}. "
                f"Attempt {attempt + 1}/{max_retries}."
            )
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)

    print(f"  [FAIL] Failed to generate image for {filename} after {max_retries} attempts.")
    return None


def create_word_document(
    story_data: dict,
    output_dir: str = ".",
    image_dir: str = "story_images",
    session_id: str = "default"
) -> str:
    """
    Build a formatted Word document from a story JSON.

    Layout: cover page, then one spread per page (left image, right text).
    Returns:
        Path to saved .docx.
    """
    print("[DOC] Assembling the Word document...")
    doc = Document()

    image_report = {
        "expected_images": 0,
        "generated_images": 0,
        "failed_images": 0,
        "skipped_images": 0,
        "disabled_reason": None,
        "image_generation_time_seconds": 0.0,
    }

    doc.add_heading(story_data.get("title", "Untitled Story"), level=1)

    meta_para = doc.add_paragraph()
    meta_para.add_run(
        f"Target Age: {story_data.get('target_age_confirmation', '?')} | "
    ).bold = True
    meta_para.add_run(
        f"Category: {story_data.get('story_category', '?')} | "
    ).bold = True
    meta_para.add_run(
        f"Tone: {story_data.get('story_tone', '?')} | "
    ).bold = True
    meta_para.add_run(
        f"Moral: {story_data.get('moral_value', '?')}\n"
    ).bold = True
    meta_para.add_run(f"Synopsis: {story_data.get('synopsis', '')}")

    # Cover image
    cover_prompt = story_data.get("cover_image_prompt")
    if cover_prompt:
        image_report["expected_images"] += 1
        t0 = time.time()
        cover_path = download_page_image(cover_prompt, "cover.jpg", image_dir=image_dir, session_id=session_id)
        image_report["image_generation_time_seconds"] += round(time.time() - t0, 2)
        if cover_path:
            image_report["generated_images"] += 1
            doc.add_picture(cover_path, width=Inches(5.0))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif _image_generation_disabled_reason:
            image_report["skipped_images"] += 1
            image_report["disabled_reason"] = _image_generation_disabled_reason
        else:
            image_report["failed_images"] += 1

    doc.add_page_break()

    # Story pages
    for page in story_data.get("pages", []):
        page_num = page.get("page_number", "?")
        text = page.get("text", "")
        image_prompt = page.get("image_prompt", "")

        image_path = None
        if image_prompt:
            image_report["expected_images"] += 1
            if _image_generation_disabled_reason:
                image_report["skipped_images"] += 1
                image_report["disabled_reason"] = _image_generation_disabled_reason
            else:
                image_filename = f"page_{page_num}.jpg"
                t0 = time.time()
                image_path = download_page_image(image_prompt, image_filename, image_dir=image_dir, session_id=session_id)
                image_report["image_generation_time_seconds"] += round(time.time() - t0, 2)
                if image_path:
                    image_report["generated_images"] += 1
                elif _image_generation_disabled_reason:
                    image_report["skipped_images"] += 1
                    image_report["disabled_reason"] = _image_generation_disabled_reason
                else:
                    image_report["failed_images"] += 1

        table = doc.add_table(rows=1, cols=2)
        table.autofit = False
        table.columns[0].width = Inches(3.5)
        table.columns[1].width = Inches(3.0)

        row = table.rows[0]
        cell_left = row.cells[0]
        if image_path:
            paragraph = cell_left.paragraphs[0]
            run = paragraph.add_run()
            run.add_picture(image_path, width=Inches(3.2))

        cell_right = row.cells[1]
        cell_right.text = f"Page {page_num}\n\n{text}"

        doc.add_page_break()

    os.makedirs(output_dir, exist_ok=True)
    output_filename = story_data.get("output_filename", "Prajna_Story_Sample.docx")
    output_path = os.path.join(output_dir, output_filename)
    doc.save(output_path)

    story_data["_image_generation_report"] = image_report

    print(f"[OK] Document saved: {output_path}")
    print(
        "[IMG] Summary: "
        f"generated={image_report['generated_images']} "
        f"failed={image_report['failed_images']} "
        f"skipped={image_report['skipped_images']} "
        f"expected={image_report['expected_images']}"
    )
    if image_report["disabled_reason"]:
        print(f"[IMG] Disabled reason: {image_report['disabled_reason']}")

    return output_path
