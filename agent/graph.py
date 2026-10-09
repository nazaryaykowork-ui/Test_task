"""The LangGraph state machine of one turn (docs/agent_flow.mmd)."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agent.bq_client import BigQueryRunner
from agent.config import Config
from agent.llm import LLM
from agent.nodes import Nodes
from agent.resilience import TurnBudget
from agent.state import AgentState
from agent.store import ReportStore
from agent.tracing import Tracer
from agent.warehouse import Warehouse


def build_graph(nodes: Nodes, checkpointer: SqliteSaver, tracer: Tracer):
    def after_classify(state: AgentState) -> str:
        if not state["allowed"]:
            return "refuse"
        return {
            "schema": "describe_data",
            "discussion": "answer",
            "analysis": "plan",
            "report": "plan",
            "manage_reports": "resolve_deletion" if state["report_action"] == "delete" else "list_reports",
        }[state["intent"]]

    def retry_or_fail(state: AgentState) -> str:
        return "generate_sql" if state["attempts"] < nodes.cfg.max_attempts else "fail"

    def after_validate(state: AgentState) -> str:
        return retry_or_fail(state) if state["error"] else "execute"

    def after_execute(state: AgentState) -> str:
        if state["error"]:
            return retry_or_fail(state)
        return "generate_sql" if state["step"] < len(state["steps"]) else "answer"

    def after_resolve(state: AgentState) -> str:
        return "confirm_delete" if state["pending_delete"] else "mask"

    def after_mask(state: AgentState) -> str:
        return "save_report" if state["report_title"] else "respond"

    graph = StateGraph(AgentState)
    for name in ("classify", "refuse", "describe_data", "plan", "generate_sql", "validate", "execute",
                 "fail", "answer", "mask", "save_report", "list_reports", "resolve_deletion", "confirm_delete",
                 "respond"):
        graph.add_node(name, tracer.wrap(name, getattr(nodes, name)))

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", after_classify)
    graph.add_edge("plan", "generate_sql")
    graph.add_edge("generate_sql", "validate")
    graph.add_conditional_edges("validate", after_validate)
    graph.add_conditional_edges("execute", after_execute)
    graph.add_conditional_edges("resolve_deletion", after_resolve)
    for name in ("describe_data", "fail", "answer", "list_reports", "confirm_delete"):
        graph.add_edge(name, "mask")
    graph.add_conditional_edges("mask", after_mask)
    graph.add_edge("save_report", "respond")
    graph.add_edge("refuse", END)
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)


def can_access(graph, thread_id: str, user_id: str) -> bool:
    """Unknown threads are refused like foreign ones."""
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    return state.values.get("user_id") == user_id


@contextmanager
def open_graph(cfg: Config) -> Iterator[tuple]:
    """Yields (graph, tracer, budget). The budget is shared by the LLM gateway and the
    warehouse; Chat starts it at every turn."""
    store = ReportStore(cfg.sqlite_path)
    conn = sqlite3.connect(cfg.sqlite_path, check_same_thread=False)
    tracer = Tracer(cfg.log_path)
    budget = TurnBudget(cfg.turn_budget_s)
    try:
        bq = BigQueryRunner(cfg.gcp_project_id, cfg.dataset)
        warehouse = Warehouse(bq, cfg.max_bytes_billed, cfg.bq_timeout_s, cfg.max_rows, budget=budget)
        llm = LLM(cfg, on_call=tracer.record_llm_call, budget=budget)
        yield build_graph(Nodes(cfg, llm, warehouse, store), SqliteSaver(conn), tracer), tracer, budget
    finally:
        conn.close()
        store.close()
