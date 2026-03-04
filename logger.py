"""
logger.py — Structured observability layer.

Writes JSONL logs for API calls to disk for analytics.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class APICallRecord:
    """Immutable record written for every API call (real or cached)."""
    timestamp:        str
    task_type:        str
    model_used:       str
    input_tokens:     int
    output_tokens:    int
    latency_ms:       float
    cost_usd_estimate: float
    cache_hit:        bool
    story_id:         Optional[str] = None
    router_decision:  Optional[str] = None
    error:            Optional[str] = None

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens



class PipelineLogger:
    """
    Thread-safe, append-only JSONL logger.
    """

    def __init__(self, log_path: str = "logs/api_calls.jsonl") -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # threading.Lock() prevents concurrent writes from multiple threads.
        # Note: acquiring this lock during file I/O will briefly block the
        # event loop thread. This is safe from a data-race standpoint but is
        # not truly async-safe. At current concurrency levels this is fine;
        # switch to asyncio.Lock + aiofiles if write latency becomes an issue.
        self._lock = threading.Lock()



    def log(self, record: APICallRecord) -> None:
        """Append one log line (thread-safe)."""
        data = asdict(record)
        data["total_tokens"] = record.total_tokens   # asdict() skips @property fields
        line = json.dumps(data, ensure_ascii=False)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def log_error(
        self,
        *,
        task_type: str,
        model_used: str,
        error: str,
        story_id: Optional[str] = None,
        router_decision: Optional[str] = None,
    ) -> None:
        """Convenience method to log a failed call with zero tokens/cost."""
        record = APICallRecord(
            timestamp=APICallRecord.now_iso(),
            task_type=task_type,
            model_used=model_used,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cost_usd_estimate=0.0,
            cache_hit=False,
            story_id=story_id,
            router_decision=router_decision,
            error=error,
        )
        self.log(record)



    def read_all(self) -> list[dict]:
        """Read all log records from disk. Returns empty list if file missing."""
        if not os.path.exists(self.log_path):
            return []
        records: list[dict] = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # skip malformed lines
        return records



    def print_session_summary(self) -> None:
        records = self.read_all()
        if not records:
            print("No log records found.")
            return

        total_calls    = len(records)
        cache_hits     = sum(1 for r in records if r.get("cache_hit"))
        total_cost     = sum(r.get("cost_usd_estimate", 0) for r in records)
        total_in_tok   = sum(r.get("input_tokens",  0) for r in records)
        total_out_tok  = sum(r.get("output_tokens", 0) for r in records)
        errors         = sum(1 for r in records if r.get("error"))

        print("\n" + "=" * 60)
        print("  Pipeline Session Summary (CUMULATIVE)")
        print("  Note: Includes all calls in logs/api_calls.jsonl")
        print("=" * 60)
        print(f"  Total API calls  : {total_calls}")
        print(f"  Cache hits       : {cache_hits} ({cache_hits/total_calls*100:.1f}%)")
        print(f"  Errors           : {errors}")
        print(f"  Input  tokens    : {total_in_tok:,}")
        print(f"  Output tokens    : {total_out_tok:,}")
        print(f"  Est. total cost  : ${total_cost:.6f}")
        print("=" * 60)


# Module-level singleton used by all pipeline components
pipeline_logger = PipelineLogger()
