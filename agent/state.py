"""Graph state, persisted per thread by the SQLite checkpointer."""

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    user_id: str  # owner of the thread
    question: str
    allowed: bool
    guard_reason: str  # logged as a security event on refusal
    intent: str
    reply: str
    outcome: str  # how the turn ended: answered, refused, sql_failed, deleted, ...
    history: Annotated[list[dict], operator.add]  # {"role", "content"}, appended each turn

    # Per turn: classify resets or overwrites these at the start of every turn.
    steps: list[str]  # sub-questions from the planner
    step: int
    sql: str  # the model's query; the retry prompt shows it this way
    scoped_sql: str  # the same query after the SQL guard, as executed
    attempts: int
    error: str  # fed back to the model on the next attempt
    empty_retried: bool  # an empty result is retried once, then accepted
    results: list[dict]  # {"question", "sql", "data"} per finished sub-question
    report_title: str  # set only when the reply is a report to save
    report_action: str
    report_keyword: str  # "" means all reports
    this_conversation: bool
    pending_delete: list[dict]  # the exact reports shown for confirmation


TURN_RESET = {
    "outcome": "",
    "steps": [],
    "step": 0,
    "sql": "",
    "scoped_sql": "",
    "attempts": 0,
    "error": "",
    "empty_retried": False,
    "results": [],
    "report_title": "",
    "pending_delete": [],
}
