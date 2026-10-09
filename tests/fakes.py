"""Fakes behind the same interfaces as the real Gemini and BigQuery clients.

Each fake plays back a script of responses in order: a value is returned, an exception
instance is raised. Running past the end of the script fails the test, so an unexpected
extra call (and the quota it would cost) is caught.
"""

import json
from types import SimpleNamespace

import httpx
import pandas as pd
from google.api_core import exceptions as gcp
from google.genai import errors

# Column layout of bigquery-public-data.thelook_ecommerce (the four tables the agent uses).
THELOOK_SCHEMA = {
    "orders": "order_id INTEGER, user_id INTEGER, status STRING, gender STRING, created_at TIMESTAMP, "
    "returned_at TIMESTAMP, shipped_at TIMESTAMP, delivered_at TIMESTAMP, num_of_item INTEGER",
    "order_items": "id INTEGER, order_id INTEGER, user_id INTEGER, product_id INTEGER, inventory_item_id INTEGER, "
    "status STRING, created_at TIMESTAMP, shipped_at TIMESTAMP, delivered_at TIMESTAMP, returned_at TIMESTAMP, "
    "sale_price FLOAT",
    "products": "id INTEGER, cost FLOAT, category STRING, name STRING, brand STRING, retail_price FLOAT, "
    "department STRING, sku STRING, distribution_center_id INTEGER",
    "users": "id INTEGER, first_name STRING, last_name STRING, email STRING, age INTEGER, gender STRING, "
    "state STRING, street_address STRING, postal_code STRING, city STRING, country STRING, latitude FLOAT, "
    "longitude FLOAT, traffic_source STRING, created_at TIMESTAMP, user_geom GEOGRAPHY",
}


class FakeClock:
    """Test time: advanced by fake latency and backoff sleeps, never by real waiting."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeGenaiClient:
    """Stands in for google.genai.Client; the agent only uses client.models.generate_content.

    With a clock, every call takes `latency_s`. If that is longer than the request timeout
    the agent set, the call times out like the real SDK: the clock moves by the timeout and
    httpx.ReadTimeout is raised without consuming a scripted response.
    """

    def __init__(self, *script, clock: FakeClock | None = None):
        self.script = list(script)
        self.clock = clock
        self.latency_s = 0.0
        self.calls = []  # SimpleNamespace(model, prompt, timeout_s) per call
        self.models = self

    def generate_content(self, model, contents, config=None):
        timeout_ms = config and config.http_options and config.http_options.timeout
        timeout_s = timeout_ms / 1000 if timeout_ms else float("inf")
        self.calls.append(SimpleNamespace(model=model, prompt=contents, timeout_s=timeout_s))
        if self.clock:
            if self.latency_s > timeout_s:
                self.clock.advance(timeout_s)
                raise httpx.ReadTimeout("timed out")
            self.clock.advance(self.latency_s)
        assert self.script, f"unexpected LLM call #{len(self.calls)}"
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, SimpleNamespace):  # a prepared response, e.g. gemini_blocked()
            return item
        text = item if isinstance(item, str) or item is None else json.dumps(item)  # dicts: structured output
        usage = SimpleNamespace(prompt_token_count=len(contents) // 4, candidates_token_count=len(text or "") // 4)
        return SimpleNamespace(text=text, usage_metadata=usage, prompt_feedback=None, candidates=[])


class FakeBigQuery:
    """Stands in for BigQueryRunner. Script items: a list of row dicts, a DataFrame, or an exception.

    Dry runs are not scripted: they report `scan_bytes` for every query.
    """

    def __init__(self, *script, scan_bytes: int = 10_000_000):
        self.script = list(script)
        self.scan_bytes = scan_bytes
        self.queries = []  # every SQL string sent for execution
        self.dry_runs = []
        self.job_configs = []
        self.max_results = []

    def dry_run(self, sql_query: str) -> int:
        self.dry_runs.append(sql_query)
        return self.scan_bytes

    def execute_query(self, sql_query: str, job_config=None, max_results=None) -> pd.DataFrame:
        self.queries.append(sql_query)
        self.job_configs.append(job_config)
        self.max_results.append(max_results)
        assert self.script, f"unexpected BigQuery call #{len(self.queries)}"
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, pd.DataFrame) else pd.DataFrame(item)

    def get_table_schema(self, table_name: str) -> list[dict]:
        columns = (c.split() for c in THELOOK_SCHEMA[table_name].split(", "))
        return [{"name": name, "type": type_, "mode": "NULLABLE", "description": ""} for name, type_ in columns]


def bq_query_error(message: str = "Unrecognized name: revenue at [1:8]") -> gcp.BadRequest:
    return gcp.BadRequest(message)


def bq_bytes_limit() -> gcp.InternalServerError:
    """As observed live: BigQuery reports an exceeded bytes-billed limit as HTTP 500."""
    message = "Query exceeded limit for bytes billed: 1000. 10485760 or higher required."
    return gcp.InternalServerError(message, errors=[{"reason": "bytesBilledLimitExceeded", "message": message}])


def bq_job_timeout() -> gcp.Cancelled:
    """As observed live: a job over job_timeout_ms ends with HTTP 499, reason 'stopped'."""
    message = "Job execution was cancelled: Job timed out after 30 sec"
    return gcp.Cancelled(message, errors=[{"reason": "stopped", "message": message}])


def bq_unavailable() -> gcp.ServiceUnavailable:
    return gcp.ServiceUnavailable("Service is temporarily unavailable")


EMPTY = pd.DataFrame()


def gemini_error(code: int) -> errors.APIError:
    """A 429 or 5xx exactly as the SDK raises it."""
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"code": code, "message": f"fake {code}", "status": "FAKE"}})


def gemini_daily_quota() -> errors.ClientError:
    """The 429 observed live when a model's free-tier requests per day are used up."""
    violation = {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [violation]}]
    return errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
                                              "details": details}})


def gemini_blocked(reason: str = "SAFETY") -> SimpleNamespace:
    """A response whose prompt Gemini's safety filters blocked: no text, a block reason."""
    usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=0)
    return SimpleNamespace(text=None, usage_metadata=usage, prompt_feedback=SimpleNamespace(block_reason=reason),
                           candidates=[])


def gemini_timeout() -> httpx.TimeoutException:
    return httpx.ReadTimeout("timed out")


def classified(intent: str, allowed: bool = True, **extra) -> dict:
    """Scripted output of the classify node."""
    return {"allowed": allowed, "reason": "test", "intent": intent, **extra}


def planned(*steps: str) -> dict:
    return {"steps": list(steps)}


def sql(query: str) -> dict:
    return {"sql": query}


def report(title: str, content: str) -> dict:
    return {"title": title, "content": content}
