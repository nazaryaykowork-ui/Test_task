"""One JSON line per node span and one per turn, sharing the turn's trace_id, in
logs/agent.jsonl; `summarize` aggregates them for /stats.

A single current span on the tracer is enough because the CLI runs one turn at a time and
the graph runs its nodes one after another. Production uses OpenTelemetry context instead.
"""

import inspect
import json
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path

from langgraph.errors import GraphInterrupt

from agent.pii import mask_pii

SPAN_FIELDS = ("allowed", "intent", "sql", "attempts", "error", "outcome")


class Tracer:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.trace_id: str | None = None
        self._span: dict | None = None
        self._turn: Counter = Counter()

    def start_turn(self) -> str:
        self.trace_id = uuid.uuid4().hex[:12]
        # Explicit zeros, so every turn record has the same fields.
        self._turn = Counter(dict.fromkeys(("llm_calls", "llm_errors", "tokens_in", "tokens_out",
                                            "sql_attempts", "sql_retries"), 0))
        self._started = time.perf_counter()
        return self.trace_id

    def end_turn(self, **fields) -> None:
        self._write(
            {
                "type": "turn",
                "latency_ms": round((time.perf_counter() - self._started) * 1000),
                **self._turn,
                **fields,
            }
        )

    def security_event(self, **fields) -> None:
        self._write({"type": "security", **fields})

    def wrap(self, name: str, node):
        takes_config = "config" in inspect.signature(node).parameters

        # No functools.wraps: LangGraph reads the signature to decide whether to pass
        # `config`, and must see this wrapper's (state, config), not the node's.
        def traced(state, config):
            self._span = Counter()
            started = time.perf_counter()
            outcome, update = "ok", {}
            try:
                update = node(state, config) if takes_config else node(state)
                return update
            except GraphInterrupt:  # interrupt() pauses the graph by raising
                outcome = "interrupted"
                raise
            except Exception as e:
                outcome = f"error: {type(e).__name__}"
                raise
            finally:
                span = {
                    "type": "span",
                    "node": name,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "status": outcome,
                    **self._span,
                    # Empty and zero values are noise, but a False guard verdict is not.
                    **{k: update[k] for k in SPAN_FIELDS if k in update and (k == "allowed" or update[k])},
                }
                self._write(span)
                self._turn.update(self._span)
                self._turn["sql_attempts"] += name == "generate_sql"
                self._turn["sql_retries"] += name == "generate_sql" and update.get("attempts", 0) > 1
                self._span = None

        return traced

    def record_llm_call(self, model: str, tokens_in: int = 0, tokens_out: int = 0, error: str = "") -> None:
        span = self._span
        span["llm_calls"] += 1
        span["tokens_in"] += tokens_in
        span["tokens_out"] += tokens_out
        span["llm_errors"] += bool(error)
        span[f"calls_{model}"] += 1
        if error:  # "llm_error:ReadTimeout" vs "llm_error:ClientError_429": a timeout is not a quota
            span[f"llm_error:{error}"] += 1

    def _write(self, record: dict) -> None:
        # Logs are PII-free too.
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "trace_id": self.trace_id, **record}
        record = {k: mask_pii(v) if isinstance(v, str) else v for k, v in record.items()}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def error_details(e: Exception) -> dict:
    return {"error": f"{type(e).__name__}: {e}", "traceback": "".join(traceback.format_exception(e))[-2000:]}


def summarize(path: Path) -> str:
    if not path.exists():
        return "No turns logged yet."
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    turns = [r for r in records if r["type"] == "turn"]
    spans = [r for r in records if r["type"] == "span"]
    if not turns:
        return "No turns logged yet."

    latencies = sorted(t["latency_ms"] for t in turns)
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    outcomes = Counter(t["outcome"] for t in turns)
    total = Counter()
    for t in turns:
        total.update({k: t.get(k, 0) for k in ("llm_calls", "llm_errors", "tokens_in", "tokens_out",
                                               "sql_attempts", "sql_retries")})
    node_latency: dict[str, list[int]] = {}
    for s in spans:
        node_latency.setdefault(s["node"], []).append(s["latency_ms"])
    slowest = sorted(((sum(v) / len(v), k) for k, v in node_latency.items()), reverse=True)[:3]

    return "\n".join(
        [
            f"Turns:      {len(turns)}  ({', '.join(f'{k} {v}' for k, v in outcomes.most_common())})",
            f"Latency:    avg {sum(latencies) / len(latencies) / 1000:.1f} s, p95 {p95 / 1000:.1f} s",
            f"LLM:        {total['llm_calls']} calls, {total['llm_errors']} failed (retried or fell back), "
            f"tokens in/out {total['tokens_in']:,} / {total['tokens_out']:,}",
            f"SQL:        {total['sql_attempts']} queries written, {total['sql_retries']} self-corrections",
            f"Refusals:   {outcomes.get('refused', 0) + outcomes.get('blocked', 0)}   Failures: "
            f"{sum(v for k, v in outcomes.items() if k in ('sql_failed', 'unavailable', 'error'))}",
            "Slowest nodes (avg): " + ", ".join(f"{name} {ms / 1000:.2f} s" for ms, name in slowest),
        ]
    )
