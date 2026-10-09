"""Backoff for transient failures of external services, and a time budget per chat turn."""

import math
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class ServiceUnavailable(RuntimeError):
    """An external service kept failing after retries; the turn ends with a clear message."""

    def __init__(self, service: str, cause: Exception) -> None:
        super().__init__(f"{service} unavailable: {type(cause).__name__}: {cause}")
        self.service = service


class TurnBudget:
    """Each request gets at most the time left, so an outage ends the turn when the budget is
    spent rather than after every retry has timed out. Unlimited until start() is called."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds = seconds
        self.clock = clock
        self.ends_at = math.inf

    def start(self) -> None:
        self.ends_at = self.clock() + self.seconds

    def remaining(self) -> float:
        return self.ends_at - self.clock()

    def check(self, service: str) -> None:
        if self.remaining() <= 0:
            raise ServiceUnavailable(service, TimeoutError(f"turn time budget of {self.seconds:g} s used up"))


def with_backoff(
    call: Callable[[], T],
    is_transient: Callable[[Exception], bool],
    attempts: int,
    sleep: Callable[[float], None] = time.sleep,
    base_delay_s: float = 1.0,
) -> T:
    """Retries transient errors after 1 s, 2 s, 4 s... plus jitter; other errors propagate."""
    for attempt in range(attempts - 1):
        try:
            return call()
        except Exception as e:
            if not is_transient(e):
                raise
            sleep(base_delay_s * 2**attempt + random.uniform(0, base_delay_s / 2))
    return call()
