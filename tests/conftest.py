import sqlite3
from types import SimpleNamespace

import pytest
import sqlglot
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.chat import Chat
from agent.config import Config
from agent.graph import build_graph
from agent.llm import LLM
from agent.nodes import Nodes
from agent.resilience import TurnBudget
from agent.sql_guard import TABLES
from agent.store import ReportStore
from agent.tracing import Tracer
from agent.warehouse import Warehouse
from tests.fakes import FakeBigQuery, FakeClock, FakeGenaiClient


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        gemini_api_key="test",
        gcp_project_id="test",
        model="primary-model",
        fallback_model="fallback-model",
        llm_timeout_s=30,
        dataset="bigquery-public-data.thelook_ecommerce",
        tables=("orders", "order_items", "products", "users"),
        pii_columns=("first_name", "last_name", "email", "street_address", "postal_code", "latitude",
                     "longitude", "user_geom"),
        max_bytes_billed=1_000_000_000,
        bq_timeout_s=30,
        max_steps=3,
        max_attempts=3,
        max_rows=50,
        turn_budget_s=60,
        sqlite_path=tmp_path / "agent.db",
        log_path=tmp_path / "agent.jsonl",
        user_categories={"alice": ("Jeans", "Pants"), "bob": ("Sweaters",)},
    )


@pytest.fixture
def make_agent(cfg):
    """make_agent(llm=[...], bq=[...]) -> agent with .graph, .llm, .bq, .store, .tracer, .sleeps,
    .clock and .budget.

    SQLite and the JSONL log live in a temp dir. Time is a FakeClock: backoff sleeps are
    recorded in .sleeps and advance the clock instead of slowing the tests down.
    """
    opened = []

    def make(llm=(), bq=(), scan_bytes=10_000_000):
        clock, sleeps = FakeClock(), []
        fake_llm, fake_bq = FakeGenaiClient(*llm, clock=clock), FakeBigQuery(*bq, scan_bytes=scan_bytes)
        budget = TurnBudget(cfg.turn_budget_s, clock=clock)

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock.advance(seconds)

        store = ReportStore(cfg.sqlite_path)
        conn = sqlite3.connect(cfg.sqlite_path, check_same_thread=False)
        opened.extend([store, conn])
        tracer = Tracer(cfg.log_path)
        llm_gateway = LLM(cfg, client=fake_llm, sleep=sleep, on_call=tracer.record_llm_call, budget=budget)
        warehouse = Warehouse(
            fake_bq, cfg.max_bytes_billed, cfg.bq_timeout_s, cfg.max_rows, sleep=sleep, budget=budget
        )
        graph = build_graph(Nodes(cfg, llm_gateway, warehouse, store), SqliteSaver(conn), tracer)
        return SimpleNamespace(graph=graph, llm=fake_llm, bq=fake_bq, store=store, tracer=tracer, sleeps=sleeps,
                               clock=clock, budget=budget)

    yield make
    for resource in opened:
        resource.close()


def turn(agent, text: str, user: str = "alice", thread: str = "t1") -> dict:
    """Invokes the graph directly; exceptions propagate (see `chat` for the user-facing layer)."""
    return agent.graph.invoke({"user_id": user, "question": text}, {"configurable": {"thread_id": thread}})


def chat(agent, user: str = "alice", thread: str = "t1") -> Chat:
    return Chat(agent.graph, agent.tracer, user, thread, agent.budget)


def unscoped(executed_sql: str) -> str:
    """The model's part of an executed query: the SQL without the guard's scope CTEs."""
    query = sqlglot.parse_one(executed_sql, read="bigquery")
    with_ = query.args["with_"]
    own_ctes = [cte for cte in with_.expressions if cte.alias not in TABLES]
    with_.set("expressions", own_ctes)
    if not own_ctes:
        query.set("with_", None)
    return query.sql(dialect="bigquery")
