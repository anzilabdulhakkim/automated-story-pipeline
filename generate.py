"""
generate.py — Main CLI entry point for the Prajna Story Pipeline.

Generates N stories with full Word document output (cover + page images + text).

Usage:
    python generate.py [N]              # Generate N stories (default: 1)
    python generate.py 5 --dry-run      # Generate 5 stories without API calls
    python generate.py 10 --text-only   # Generate stories without images/docs

This replaces the old venv/generate_100.py and venv/build_story_doc.py.
All logic now flows through the proper pipeline:
    generate.py → story_generator.py → gemini_client.py → Gemini Flash
    generate.py → doc_builder.py → image generation + Word assembly
"""

import argparse
import asyncio
import io
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for emoji)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from config import CONFIG
from story_generator import generate_story
from doc_builder import create_word_document


def _configure_event_loop_policy() -> None:
    """
    Use selector loop on Windows to avoid noisy Proactor transport shutdown
    callbacks on transient remote connection resets.
    """
    if sys.platform != "win32":
        return
    try:
        policy = asyncio.WindowsSelectorEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)
    except Exception:
        # Non-fatal: keep default policy if unavailable.
        pass


# ---------------------------------------------------------------------------
# Story Configuration Data (from generate_100.py)
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Family", "Friendship", "Love", "Adventure", "Animals", "Fantasy",
    "Courage", "Imagination", "Fairy Tales", "Perseverance", "Folktales",
    "Mythology", "Science Fiction",
]

TONES = [
    "Happy", "Sad", "Exciting", "Calm", "Funny", "Scary",
    "Educational", "Inspirational",
]

MORALS = [
    "Honesty", "Kindness", "Sharing", "Forgiveness", "Patience",
    "Responsibility", "Respect", "Gratitude",
]

ART_STYLES = [
    # Must match the art style prefixes defined in story_generator.py STORY_SYSTEM_PROMPT
    "Disney3D", "Watercolor", "FlatVector", "Anime", "Clay",
]

PROFILES = [
    {"nickname": "Leo",    "gender": "he",   "age": 3},
    {"nickname": "Rose",   "gender": "she",  "age": 4},
    {"nickname": "Mia",    "gender": "she",  "age": 5},
    {"nickname": "Noah",   "gender": "he",   "age": 6},
    {"nickname": "Emma",   "gender": "she",  "age": 7},
    {"nickname": "Sam",    "gender": "they", "age": 8},
    {"nickname": "Liam",   "gender": "he",   "age": 9},
    {"nickname": "Ethan",  "gender": "he",   "age": 10},
    {"nickname": "Olivia", "gender": "she",  "age": 11},
    {"nickname": "Alex",   "gender": "they", "age": 12},
]

USER_TEXT_PROMPTS = [
    "bedtime story about dinosaur",
    "funny story about dragon",
    "princess story with happy ending",
    "story for my daughter age 5 about sharing",
    "lion and rabbit story with moral",
    "space adventure story for kids",
    "short bedtime story about moon",
    "story about brave boy",
    "magic story with unicorn",
    "story about friendship",
    "story for my son Aarav",
    "bedtime story princess and dragon",
    "story about robot and girl",
    "jungle story with animals",
    "story about lost puppy",
    "funny monster story",
    "story about school adventure",
    "story about mermaid",
    "story about superhero kid",
    "story about talking cat",
    "short story for sleep",
    "story about fairy and boy",
    "story about space rocket",
    "moral story about honesty",
    "story about shark but friendly",
    "story about dragon and knight",
    "story about magic forest",
    "story about brave girl",
    "story about flying horse",
    "story about moon and star",
    "story about angry boy learning lesson",
    "story about kid afraid of dark",
    "story about sharing toys",
    "story about kindness",
    "story about helping others",
    "story with my kid name Vihaan",
    "story about birthday adventure",
    "story about pirate treasure",
    "story about alien friend",
    "story about lost robot",
    "story about elephant and mouse",
    "story about fox and crow",
    "story about brave princess",
    "story about ninja kid",
    "story about talking tree",
    "story about bedtime adventure",
    "story about space journey",
    "story about magic school",
    "story about brave lion",
    "story about unicorn and girl",
    "funny bedtime story",
    "short funny story",
    "sleep story kids",
    "story about hero boy",
    "story about brave dog",
    "story about monster but nice",
    "story about dragon baby",
    "story about space kid",
    "story about astronaut",
    "story about jungle king",
    "story about magic wand",
    "story about fairy tale",
    "story about talking car",
    "story about robot friend",
    "story about brave kid",
    "story about friendship moral",
    "story about helping friend",
    "story about sharing lesson",
    "story about confidence",
    "story about fear",
    "story about ocean adventure",
    "story about flying boy",
    "story about invisible boy",
    "story about magic door",
    "story about dream adventure",
    "story about sleeping moon",
    "story about star falling",
    "story about rainbow",
    "story about cloud friend",
    "story about sun adventure",
    "story about princess and unicorn",
    "story about dragon and boy",
    "story about knight",
    "story about fairy kingdom",
    "story about magic castle",
    "story about animal friends",
    "story about school friends",
    "story about jungle adventure",
    "story about forest adventure",
    "story about space friends",
    "story about hero girl",
    "story about brave child",
    "story about smart kid",
    "story about funny kid",
    "story about kind kid",
]


# ---------------------------------------------------------------------------
# Config Generator
# ---------------------------------------------------------------------------

def generate_story_configs(amount: int = 1) -> list[dict[str, Any]]:
    """Generate random story configuration dicts."""
    configs = []
    for _ in range(amount):
        profile = random.choice(PROFILES)
        configs.append({
            "user_text":       random.choice(USER_TEXT_PROMPTS),
            "user_nickname":   profile["nickname"],
            "user_gender":     profile["gender"],
            "target_age":      profile["age"],
            "story_category":  random.choice(CATEGORIES),
            "story_tone":      random.choice(TONES),
            "moral_value":     random.choice(MORALS),
            "language":        "English",
            "art_style":       random.choice(ART_STYLES),
        })
    return configs


# ---------------------------------------------------------------------------
# Single Story Pipeline
# ---------------------------------------------------------------------------

def save_story_as_markdown(story_json: dict, config: dict) -> str:
    """Save the story as a clean Markdown file for visibility."""
    os.makedirs("stories", exist_ok=True)

    safe_title = "".join(
        c for c in story_json.get("title", "Untitled")
        if c.isalnum() or c in " _-"
    ).strip().replace(" ", "_")

    filename = f"Prajna_STORY_{config['target_age']}yr_{config['user_nickname']}_{safe_title}.md"
    filepath = os.path.join("stories", filename)

    md = []
    md.append(f"# {story_json.get('title', 'Untitled')}")
    md.append(f"\n**Target Age:** {story_json.get('target_age_confirmation', '?')} | **Category:** {story_json.get('story_category', '?')} | **Moral:** {story_json.get('moral_value', '?')}")
    md.append(f"\n> **Synopsis:** {story_json.get('synopsis', '')}")
    md.append(f"\n---\n")

    for page in story_json.get("pages", []):
        md.append(f"### Page {page.get('page_number', '?')}")
        md.append(f"\n{page.get('text', '')}")
        md.append(f"\n*Image Prompt:* {page.get('image_prompt', '')}")
        md.append(f"\n---\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return filepath


async def process_single_story(
    index: int,
    config: dict[str, Any],
    session_id: str,
    *,
    dry_run: bool = False,
    text_only: bool = False,
) -> dict[str, Any]:
    """
    Generate one story: text → (optionally) images + Word doc.

    Returns an analytics record dict.
    """
    story_id = f"gen-{session_id}-STORY-{index + 1:03d}"
    analytics = {
        "story_number": index + 1,
        "config": config,
        "text_generation_time_seconds": 0,
        "doc_generation_time_seconds": 0,
        "total_time_seconds": 0,
        "image_report": None,
        "status": "pending",
    }

    try:
        print(f"\n{'='*55}")
        print(f"  Story {index + 1} — {config['user_nickname']} (age {config['target_age']})")
        print(f"  Prompt: {config['user_text']}")
        print(f"{'='*55}")

        # ── 1. Generate story text via pipeline ──────────────────────
        text_start = time.time()
        result = await generate_story(
            config, session_id, story_id, dry_run=dry_run,
        )
        text_end = time.time()
        analytics["text_generation_time_seconds"] = round(text_end - text_start, 2)

        story_json = result["story"]
        print(f"  ✅ Story generated: \"{story_json.get('title', 'Untitled')}\"")
        print(f"     Pages: {story_json.get('total_pages', '?')} | "
              f"Time: {analytics['text_generation_time_seconds']}s")

        # ── 2. Save Markdown Version (Always) ──────────────────────
        md_path = save_story_as_markdown(story_json, config)
        print(f"     📝 Story saved to: {md_path}")

        # ── 2. Build Word document with images (unless text-only) ────
        if not text_only and not dry_run:
            safe_title = "".join(
                c for c in story_json.get("title", f"Story_{index+1}")
                if c.isalnum() or c in " _-"
            ).strip()
            story_json["output_filename"] = (
                f"Prajna_{config['target_age']}yr_"
                f"{config['user_nickname']}_{safe_title}_{story_id[-6:]}.docx"
            )

            # Per-story image folder so images don't overwrite across stories
            image_folder = os.path.join(
                CONFIG.story_images_dir,
                f"{config['user_nickname']}_{safe_title}".replace(" ", "_"),
            )

            doc_start = time.time()
            create_word_document(story_json, output_dir=CONFIG.output_dir, image_dir=image_folder, session_id=session_id)
            doc_end = time.time()
            analytics["doc_generation_time_seconds"] = round(doc_end - doc_start, 2)
            analytics["image_report"] = story_json.get("_image_generation_report")

        analytics["total_time_seconds"] = round(time.time() - text_start, 2)
        if dry_run or text_only:
            analytics["status"] = "success"
        else:
            report = analytics["image_report"] or {}
            expected = int(report.get("expected_images", 0) or 0)
            generated = int(report.get("generated_images", 0) or 0)
            disabled_reason = report.get("disabled_reason")

            if expected == 0 or generated == expected:
                analytics["status"] = "success"
            elif disabled_reason:
                analytics["status"] = "partial_success_images_disabled"
                print(f"  ⚠️  Image generation disabled: {disabled_reason}")
            elif generated == 0:
                analytics["status"] = "partial_success_no_images"
                print("  ⚠️  Story generated, but no images were created.")
            else:
                analytics["status"] = "partial_success_missing_images"
                print(
                    "  ⚠️  Story generated with partial images "
                    f"({generated}/{expected})."
                )

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        analytics["status"] = f"failed: {e}"

    return analytics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(amount: int, dry_run: bool, text_only: bool) -> None:
    """Run the full generation pipeline."""
    # Ensure directories exist
    os.makedirs(CONFIG.log_dir, exist_ok=True)
    os.makedirs(CONFIG.cache_dir, exist_ok=True)
    os.makedirs(CONFIG.configs_dir, exist_ok=True)
    os.makedirs(CONFIG.analytics_dir, exist_ok=True)
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    os.makedirs(CONFIG.story_images_dir, exist_ok=True)
    os.makedirs("stories", exist_ok=True)

    # Generate configs
    print(f"\n🚀 Generating {amount} story configuration(s)...\n")
    configs = generate_story_configs(amount)

    # Timestamped filenames so nothing overwrites
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save configs for audit
    config_file = os.path.join(CONFIG.configs_dir, f"story_configs_batch_{amount}_{ts}.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=4)
    print(f"📋 Configs saved to {config_file}")

    session_id = f"batch-{amount}-{ts}"
    analytics_data = []

    # Process stories.
    # If text_only is True, we can process all concurrently (Issue #21).
    # If generating images, we stick to sequential to avoid aggressive Imagen rate-limiting.
    if text_only:
        print(f"⚡ Processing {amount} stories concurrently (--text-only)...")
        tasks = [
            process_single_story(i, cfg, session_id, dry_run=dry_run, text_only=True)
            for i, cfg in enumerate(configs)
        ]
        # return_exceptions=True for fault isolation (#21)
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Post-process results (convert exceptions to records)
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                print(f"❌ Story {i+1} failed: {r}")
                analytics_data.append({
                    "story_number": i + 1,
                    "status": f"crashed: {r}",
                    "total_time_seconds": 0,
                    "telemetry": None,
                })
            else:
                analytics_data.append(r)
    else:
        # Full run: process each story sequentially
        for i, cfg in enumerate(configs):
            result = await process_single_story(
                i, cfg, session_id,
                dry_run=dry_run, text_only=text_only,
            )
            analytics_data.append(result)

    # Save analytics
    analytics_file = os.path.join(CONFIG.analytics_dir, f"generation_analytics_batch_{amount}_{ts}.json")
    with open(analytics_file, "w", encoding="utf-8") as f:
        json.dump(analytics_data, f, indent=4)

    # Summary
    success = sum(1 for a in analytics_data if a["status"] == "success")
    partial = sum(1 for a in analytics_data if str(a.get("status", "")).startswith("partial_success"))
    failed = amount - success - partial
    print(f"\n{'='*55}")
    print(f"  Pipeline Complete!")
    print(f"  ✅ Success: {success}  ⚠️ Partial: {partial}  ❌ Failed: {failed}")
    print(f"  📊 Analytics: {analytics_file}")
    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prajna Story Pipeline — Generate stories with Word docs.",
    )
    parser.add_argument(
        "amount", nargs="?", type=int, default=1,
        help="Number of stories to generate (default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the pipeline without making real API calls",
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Generate story JSON only — skip image generation and Word doc",
    )
    args = parser.parse_args()

    _configure_event_loop_policy()

    try:
        asyncio.run(main_async(args.amount, args.dry_run, args.text_only))
    except KeyboardInterrupt:
        print("\n⚠️ Cancelled by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
