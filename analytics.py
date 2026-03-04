"""
analytics.py — Post-run cost & observability aggregator.

Reads logs/api_calls.jsonl (text) and analytics/*.json (images)
and outputs a combined billing summary.
Usage:
    python analytics.py
    python analytics.py --since 2026-01-01
    python analytics.py --session batch-5-20260226
"""

import argparse
import glob
import json
import os
from collections import defaultdict

from logger import pipeline_logger

# Vertex AI pricing for gemini-3.1-flash-image-preview (1K / 1MP default resolution)
IMAGE_COST_PER_IMAGE = 0.067


def print_analytics(session_prefix: str = "", since_iso: str = "") -> None:
    records = pipeline_logger.read_all()

    # Apply filters
    if since_iso:
        records = [r for r in records if r.get("timestamp", "") >= since_iso]
    if session_prefix:
        records = [
            r for r in records
            if (r.get("story_id") or "").startswith(session_prefix)
            or (r.get("router_decision") or "").startswith(session_prefix)
        ]

    if not records:
        print("No API logs found matching the specified filters.")
        return

    total_cost = sum(r.get("cost_usd_estimate", 0.0) for r in records)
    total_in   = sum(r.get("input_tokens", 0) for r in records)
    total_out  = sum(r.get("output_tokens", 0) for r in records)
    cache_hits = sum(1 for r in records if r.get("cache_hit"))
    errors     = sum(1 for r in records if r.get("error"))
    repairs    = sum(1 for r in records if r.get("repair_call"))

    # Image cost and timing from generation analytics files
    total_images_generated = 0
    total_image_cost = 0.0
    total_image_time = 0.0
    analytics_files = glob.glob(os.path.join("analytics", "generation_analytics_*.json"))
    for af in analytics_files:
        try:
            with open(af, "r", encoding="utf-8") as fh:
                batch = json.load(fh)
            for entry in batch:
                report = entry.get("image_report") or {}
                generated = int(report.get("generated_images", 0) or 0)
                total_images_generated += generated
                total_image_cost += generated * IMAGE_COST_PER_IMAGE
                total_image_time += float(report.get("image_generation_time_seconds", 0.0) or 0.0)
        except (json.JSONDecodeError, OSError):
            pass

    print("\n" + "=" * 60)
    print("  PRODUCTION PIPELINE ANALYTICS")
    if since_iso or session_prefix:
        print(f"  Filters: session='{session_prefix or "(all)"}' since='{since_iso or "(all)"}'")
    print("=" * 60)

    # Cost by Model
    cost_by_model = defaultdict(float)
    calls_by_model = defaultdict(int)
    for r in records:
        if not r.get("cache_hit") and not r.get("error"):
            model = r.get("model_used", "unknown")
            cost_by_model[model] += r.get("cost_usd_estimate", 0.0)
            calls_by_model[model] += 1

    print("\n--- Cost by Model ---")
    for model, cost in sorted(cost_by_model.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model:<20} : ${cost:.5f} ({calls_by_model[model]} calls)")
    if total_images_generated:
        print(
            f"  gemini-3.1-flash-image-preview : ${total_image_cost:.5f} "
            f"({total_images_generated} images @ ${IMAGE_COST_PER_IMAGE}/img)"
        )

    # Stats by Task Type
    cost_by_task = defaultdict(float)
    for r in records:
        cost_by_task[r.get("task_type", "unknown")] += r.get("cost_usd_estimate", 0.0)

    print("\n--- Cost by Task Type ---")
    for task, cost in sorted(cost_by_task.items(), key=lambda x: x[1], reverse=True):
        print(f"  {task:<20} : ${cost:.5f}")
    if total_images_generated:
        print(f"  image_generation     : ${total_image_cost:.5f}")

    combined_total = total_cost + total_image_cost
    img_mins = total_image_time / 60
    print("\n--- Totals ---")
    print(f"  Total API Calls  : {len(records)} (text) + {total_images_generated} (images)")
    print(f"  Cache Hits       : {cache_hits} ({(cache_hits/len(records))*100:.1f}%)")
    print(f"  Repair Calls     : {repairs} ({(repairs/max(len(records),1))*100:.1f}%)")
    print(f"  Total Input Toks : {total_in:,}")
    print(f"  Total Output Toks: {total_out:,}")
    print(f"  Errors Logged    : {errors}")
    print(f"  Image Gen Time   : {total_image_time:.1f}s ({img_mins:.1f} min) across {total_images_generated} images")
    print(f"  Text Cost        : ${total_cost:.5f}")
    print(f"  Image Cost       : ${total_image_cost:.5f} ({total_images_generated} images @ ${IMAGE_COST_PER_IMAGE})")
    print(f"  TOTAL ESTIMATED  : ${combined_total:.5f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Story AI Pipeline Analytics")
    parser.add_argument(
        "--session",
        default="",
        help="Filter records whose story_id starts with this prefix",
    )
    parser.add_argument(
        "--since",
        default="",
        metavar="ISO_DATE",
        help="Only include records with timestamp >= this ISO date (e.g. 2026-01-01)",
    )
    args = parser.parse_args()
    print_analytics(session_prefix=args.session, since_iso=args.since)
