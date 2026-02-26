"""
analytics.py — Post-run cost & observability aggregator.

Reads logs/api_calls.jsonl and outputs a billing summary.
Usage:
    python analytics.py
    python analytics.py --since 2026-01-01
    python analytics.py --session batch-5-20260226
"""

import argparse
import os
from collections import defaultdict
from logger import pipeline_logger


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

    # Stats by Task Type
    cost_by_task = defaultdict(float)
    for r in records:
        cost_by_task[r.get("task_type", "unknown")] += r.get("cost_usd_estimate", 0.0)

    print("\n--- Cost by Task Type ---")
    for task, cost in sorted(cost_by_task.items(), key=lambda x: x[1], reverse=True):
        print(f"  {task:<20} : ${cost:.5f}")

    print("\n--- Totals ---")
    print(f"  Total API Calls  : {len(records)}")
    print(f"  Cache Hits       : {cache_hits} ({(cache_hits/len(records))*100:.1f}%)")
    print(f"  Repair Calls     : {repairs} ({(repairs/max(len(records),1))*100:.1f}%)")
    print(f"  Total Input Toks : {total_in:,}")
    print(f"  Total Output Toks: {total_out:,}")
    print(f"  Errors Logged    : {errors}")
    print(f"  TOTAL ESTIMATED  : ${total_cost:.5f}")
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
