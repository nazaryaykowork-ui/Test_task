"""BigQuery behind the agent's limits: dry-run cost check, bytes-billed cap, job timeout,
row cap, retries.

Errors are split by who can fix them:
- the query is wrong, too expensive or too slow -> QueryError, back to the model
- BigQuery is briefly unwell (429/5xx)          -> retried with backoff
- anything else, or retries exhausted           -> ServiceUnavailable, a clear message

The HTTP code alone is not enough: BigQuery reports an exceeded bytes-billed limit as 500
and a job timeout as 499, so query errors are recognised by their reason.
"""

import math
import time
from collections.abc import Callable
from typing import TypeVar

import pandas as pd
import requests
from google.api_core import exceptions as gcp
from google.cloud import bigquery

from agent.bq_client import BigQueryRunner
from agent.resilience import ServiceUnavailable, TurnBudget, with_backoff

T = TypeVar("T")

ATTEMPTS = 3
MAX_ERROR_CHARS = 500  # the model only needs the gist
QUERY_ERROR_REASONS = {"invalidQuery", "invalid", "bytesBilledLimitExceeded", "resourcesExceeded", "stopped"}


class QueryError(ValueError):
    """The message goes back to the model."""


def is_query_error(e: Exception) -> bool:
    reasons = {err.get("reason") for err in getattr(e, "errors", None) or []}
    return isinstance(e, gcp.BadRequest) or bool(reasons & QUERY_ERROR_REASONS)


def is_transient(e: Exception) -> bool:
    """Not a RetryError: the client library already retried that request until its deadline."""
    if is_query_error(e):
        return False
    return isinstance(e, gcp.TooManyRequests | gcp.ServerError | TimeoutError | requests.ConnectionError)


class Warehouse:
    def __init__(
        self,
        bq: BigQueryRunner,
        max_bytes_billed: int,
        timeout_s: int,
        max_rows: int,
        sleep: Callable[[float], None] = time.sleep,
        budget: TurnBudget | None = None,
    ) -> None:
        self.bq = bq
        self.budget = budget or TurnBudget(math.inf)
        self.max_bytes_billed = max_bytes_billed
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self.sleep = sleep

    def table_schema(self, table: str) -> list[dict]:
        return self._call(lambda: self.bq.get_table_schema(table))

    def check_cost(self, sql: str) -> None:
        """Dry run: catches invalid SQL and over-budget scans before anything is billed."""
        scanned = self._call(lambda: self.bq.dry_run(sql))
        if scanned > self.max_bytes_billed:
            raise QueryError(
                f"The query would scan {_size(scanned)}, over the {_size(self.max_bytes_billed)} limit. "
                "Select fewer columns, filter by date, or aggregate earlier."
            )

    def run(self, sql: str) -> pd.DataFrame:
        # BigQuery enforces the bytes cap too, in case the dry run was off. The job may not
        # outlast the turn.
        timeout_s = min(self.timeout_s, self.budget.remaining())
        config = bigquery.QueryJobConfig(
            maximum_bytes_billed=self.max_bytes_billed, job_timeout_ms=max(1, int(timeout_s * 1000))
        )
        # One row over the cap tells the caller that the result was cut.
        return self._call(lambda: self.bq.execute_query(sql, job_config=config, max_results=self.max_rows + 1))

    def _call(self, call: Callable[[], T]) -> T:
        def attempt() -> T:
            self.budget.check("BigQuery")
            return call()

        try:
            return with_backoff(attempt, is_transient, ATTEMPTS, self.sleep)
        except (gcp.GoogleAPIError, TimeoutError, requests.ConnectionError) as e:
            if is_query_error(e):
                raise QueryError(_reason_message(e)[:MAX_ERROR_CHARS]) from e
            raise ServiceUnavailable("BigQuery", e) from e


def _reason_message(e: gcp.GoogleAPICallError) -> str:
    """Without the request URL, project and job id around it."""
    errors = getattr(e, "errors", None) or []
    return (errors[0].get("message") if errors else None) or e.message


def _size(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.0f} MB"
