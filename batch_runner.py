"""
batch_runner.py — Async batch processor.

Reads a JSON array of story configurations (e.g., story_configs_batch_1.json),
processes them concurrently through the `story_generator` and `image_generator`,
and writes the results to `generation_analytics_batch_1.json`.

Usage:
    python batch_runner.py --config story_configs_batch_1.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

from config import CONFIG
from logger import pipeline_logger
from story_generator import generate_story
from image_generator import generate_image

# Set up console logging for the runner
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_runner")


async def process_single_story(
    index: int,
    config: dict[str, Any],
    session_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Process one config into a story and (optionally) an image."""
    story_id = f"batch-{session_id}-STORY-{index+1:03d}"
    log.info(f"[{story_id}] Starting processing (dry_run={dry_run})...")

    t0 = time.perf_counter()

    try:
        # 1. Generate text story
        story_result = await generate_story(
            config, 
            session_id, 
            story_id, 
            dry_run=dry_run
        )
        
        # 2. Attempt image generation for first page (cover)
        image_path = None
        if not dry_run:
            story_json = story_result["story"]
            safe_title = "".join(
                c for c in story_json.get("title", f"Story_{index+1}")
                if c.isalnum() or c in " _-"
            ).strip().replace(" ", "_")
            # Set output_filename so doc_builder doesn't overwrite previous stories
            story_json["output_filename"] = (
                f"Prajna_{config['target_age']}yr_"
                f"{config['user_nickname']}_{safe_title}_{story_id[-6:]}.docx"
            )
            
            image_dir = os.path.join(
                CONFIG.story_images_dir,
                f"{config['user_nickname']}_{safe_title}"
            )
            
            pages = story_json.get("pages", [])
            first_page_prompt = pages[0].get("image_prompt", "A happy scene.") if pages else "A happy scene."
            
            image_path = await generate_image(
                scene_description=first_page_prompt,
                config=config,
                session_id=session_id,
                image_dir=image_dir,
                filename="cover.jpg"
            )
    except Exception as e:
        log.error(f"[{story_id}] Failed: {e}")
        pipeline_logger.log_error(
            task_type="story_generation",
            model_used="unknown",
            error=str(e),
            story_id=story_id
        )
        return {
            "story_number": index + 1,
            "config": config,
            "story_output": None,
            "cover_image_path": None,
            "telemetry": None,
            "status": f"failed: {e}",
            "total_time_seconds": round(time.perf_counter() - t0, 2)
        }

    tt = time.perf_counter() - t0
    log.info(f"[{story_id}] Finished in {tt:.2f}s")

    if dry_run:
        status = "success"
    elif image_path:
        status = "success"
    else:
        status = "partial_success_no_cover_image"
        log.warning(f"[{story_id}] Story generated but cover image is missing.")
    
    return {
        "story_number": index + 1,
        "config": config,
        "story_output": story_result["story"],
        "cover_image_path": image_path,
        "telemetry": story_result["telemetry"],
        "status": status,
        "total_time_seconds": round(tt, 2)
    }


async def main_async(config_path: str, dry_run: bool) -> None:
    if not os.path.exists(config_path):
        log.error(f"Input file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            configs = json.load(f)
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse {config_path}: {e}")
            sys.exit(1)

    if not isinstance(configs, list):
        log.error("Config file must contain a JSON array of objects.")
        sys.exit(1)

    log.info(f"Loaded {len(configs)} story configurations from {config_path}")
    
    # Deriving output and session name
    base_name = os.path.basename(config_path)
    session_name = os.path.splitext(base_name)[0]
    out_name = base_name.replace("story_configs_", "generation_analytics_")
    if out_name == base_name:
        out_name = f"generation_analytics_{base_name}"
    
    out_path = os.path.join(os.path.dirname(config_path), out_name)

    # Process all concurrently.
    # The gemini_client internally caps the concurrency using its semaphore.
    # return_exceptions=True ensures one crashed story never cancels the entire
    # batch — each exception is collected and converted to an error record (#21).
    tasks = [
        process_single_story(i, cfg, session_name, dry_run)
        for i, cfg in enumerate(configs)
    ]

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert any unhandled exceptions to structured error records
    results = []
    for i, r in enumerate(raw_results):
        if isinstance(r, BaseException):
            log.error("Story %d crashed with unhandled exception: %s", i + 1, r)
            results.append({
                "story_number": i + 1,
                "config": configs[i],
                "story_output": None,
                "cover_image_path": None,
                "telemetry": None,
                "status": f"crashed: {r}",
                "total_time_seconds": 0,
            })
        else:
            results.append(r)

    # Write results
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    log.info(f"Batch complete. Results written to: {out_path}")

    if not dry_run:
        pipeline_logger.print_session_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Story AI Pipeline batch processor.")
    parser.add_argument("--config", required=True, help="Path to JSON config array")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline without calling Gemini APIs")
    args = parser.parse_args()

    # Create directories defined in config
    os.makedirs(CONFIG.log_dir, exist_ok=True)
    os.makedirs(CONFIG.cache_dir, exist_ok=True)

    try:
        asyncio.run(main_async(args.config, args.dry_run))
    except KeyboardInterrupt:
        log.info("Batch run cancelled by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
