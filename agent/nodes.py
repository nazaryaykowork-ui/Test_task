"""Graph nodes: each takes the state and returns a partial update."""

import re
from dataclasses import asdict
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel

from agent.config import Config, load_prompts
from agent.llm import LLM
from agent.pii import mask_pii
from agent.sql_guard import SQLRejected, guard_sql
from agent.state import TURN_RESET, AgentState
from agent.store import Report, ReportStore
from agent.warehouse import QueryError, Warehouse

HISTORY_MESSAGES = 6  # enough to recognise a follow-up
PROMPT_TAGS = re.compile(r"<(?=/?\s*(?:user_message|history|user|assistant)\b)", re.IGNORECASE)


class Classification(BaseModel):
    allowed: bool
    reason: str
    intent: Literal["schema", "discussion", "analysis", "report", "manage_reports"]
    # manage_reports arguments, extracted in the same call to save a request.
    report_action: Literal["none", "list", "delete"] = "none"
    report_keyword: str = ""
    this_conversation: bool = False


class Plan(BaseModel):
    # Listing the parts first makes the steps cover all of them; without it the planner
    # dropped one side of comparisons ("California vs Texas") in live runs.
    parts: list[str] = []
    steps: list[str]


class SQLDraft(BaseModel):
    sql: str


class ReportDraft(BaseModel):
    title: str
    content: str


class Nodes:
    def __init__(self, cfg: Config, llm: LLM, warehouse: Warehouse, store: ReportStore) -> None:
        self.cfg = cfg
        self.llm = llm
        self.warehouse = warehouse
        self.store = store
        self.schema = describe_schema(warehouse, cfg.tables, cfg.pii_columns)  # loaded once per process

    def classify(self, state: AgentState) -> dict:
        """Input guard and intent router in a single LLM call."""
        prompt = load_prompts()["classify"].format(history=recent_history(state), question=user_text(state))
        result = self.llm.generate_structured(prompt, Classification)
        return {
            **TURN_RESET,
            "allowed": result.allowed,
            "guard_reason": result.reason,
            "intent": result.intent,
            "report_action": result.report_action,
            "report_keyword": result.report_keyword.strip(),
            "this_conversation": result.this_conversation,
        }

    def refuse(self, state: AgentState) -> dict:
        # Goes straight to END, so a refused message never enters history or later prompts.
        return {"reply": load_prompts()["refusal"].strip(), "outcome": "refused"}

    def describe_data(self, state: AgentState) -> dict:
        prompts = load_prompts()
        prompt = prompts["schema"].format(persona=prompts["persona"], schema=self.schema, question=user_text(state))
        return {"reply": self.llm.generate(prompt), "outcome": "answered"}

    def plan(self, state: AgentState) -> dict:
        prompt = load_prompts()["plan"].format(
            max_steps=self.cfg.max_steps, schema=self.schema, history=recent_history(state), question=user_text(state)
        )
        steps = self.llm.generate_structured(prompt, Plan).steps[: self.cfg.max_steps]
        return {"steps": steps or [state["question"]]}

    def generate_sql(self, state: AgentState) -> dict:
        prompts = load_prompts()
        retry = prompts["sql_retry"].format(sql=state["sql"], error=state["error"]) if state["error"] else ""
        prompt = prompts["sql"].format(
            max_rows=self.cfg.max_rows,
            schema=self.schema,
            question=user_text(state),
            step=state["steps"][state["step"]],
            previous=format_results(state["results"]),
            retry=retry,
        )
        draft = self.llm.generate_structured(prompt, SQLDraft)
        return {"sql": draft.sql, "attempts": state["attempts"] + 1, "error": ""}

    def validate(self, state: AgentState) -> dict:
        """SQL guard (allowed tables, no PII, user's scope), then a BigQuery dry run."""
        categories = self.cfg.user_categories[state["user_id"]]
        try:
            scoped = guard_sql(state["sql"], self.cfg.dataset, categories, self.cfg.pii_columns)
        except SQLRejected as e:
            return {"error": f"The SQL guard rejected the query: {e}"}
        try:
            self.warehouse.check_cost(scoped)
        except QueryError as e:
            return {"error": f"BigQuery rejected the query: {e}"}
        return {"scoped_sql": scoped}

    def execute(self, state: AgentState) -> dict:
        try:
            rows = self.warehouse.run(state["scoped_sql"])
        except QueryError as e:
            return {"error": f"BigQuery rejected the query: {e}"}

        # An empty result is retried once, unless it came on the last attempt: then "no
        # matching data" is the answer.
        if rows.empty and not state["empty_retried"] and state["attempts"] < self.cfg.max_attempts:
            return {
                "error": "The query returned no rows. Check filters, literal values and date ranges.",
                "empty_retried": True,
            }
        data = rows.head(self.cfg.max_rows).to_csv(index=False) if not rows.empty else "(no rows)"
        if len(rows) > self.cfg.max_rows:
            data += (
                f"(truncated: only the first {self.cfg.max_rows} rows are shown. Do not add up or rank these "
                "rows; totals and rankings must come from SQL aggregation.)\n"
            )
        result = {"question": state["steps"][state["step"]], "sql": state["sql"], "data": data}
        return {
            "results": state["results"] + [result],
            "step": state["step"] + 1,
            "attempts": 0,
            "empty_retried": False,
        }

    def fail(self, state: AgentState) -> dict:
        step = state["steps"][state["step"]]
        return {
            "reply": f'I could not get the data for "{step}" after {state["attempts"]} attempts. '
            f"The last attempt failed because: {state['error']}\n"
            "Try rephrasing the question, for example narrow the time range or name the exact metric.",
            "outcome": "sql_failed",
        }

    def answer(self, state: AgentState) -> dict:
        prompts = load_prompts()
        data = format_results(state["results"]) or "(no new data, answer from the conversation)"
        scope = ", ".join(self.cfg.user_categories[state["user_id"]])
        if state["intent"] == "report":
            prompt = prompts["report"].format(
                persona=prompts["persona"], scope=scope, question=user_text(state), data=data
            )
            report = self.llm.generate_structured(prompt, ReportDraft)
            return {"reply": report.content, "report_title": report.title, "outcome": "report_saved"}
        prompt = prompts["answer"].format(
            persona=prompts["persona"], scope=scope, history=recent_history(state), question=user_text(state),
            data=data,
        )
        return {"reply": self.llm.generate(prompt), "outcome": "answered"}

    def mask(self, state: AgentState) -> dict:
        """Runs before anything is shown, saved or kept in history."""
        return {"reply": mask_pii(state["reply"]), "report_title": mask_pii(state["report_title"])}

    def save_report(self, state: AgentState, config: RunnableConfig) -> dict:
        thread_id = config["configurable"]["thread_id"]
        report_id = self.store.save(state["user_id"], thread_id, state["report_title"], state["reply"])
        return {"reply": f'{state["reply"]}\n\nSaved as report #{report_id}: "{state["report_title"]}"'}

    def list_reports(self, state: AgentState, config: RunnableConfig) -> dict:
        reports = self._matching_reports(state, config)
        if not reports:
            return {"reply": "You have no saved reports matching that.", "outcome": "listed"}
        return {"reply": f"Your saved reports ({len(reports)}):\n{format_reports(reports)}", "outcome": "listed"}

    def resolve_deletion(self, state: AgentState, config: RunnableConfig) -> dict:
        """Resolves the targets before asking, so the user confirms an exact list."""
        reports = self._matching_reports(state, config)
        if not reports:
            return {"reply": "You have no saved reports matching that, so nothing was deleted.",
                    "outcome": "nothing_to_delete"}
        return {"pending_delete": [asdict(r) for r in reports]}

    def confirm_delete(self, state: AgentState, config: RunnableConfig) -> dict:
        """Only an explicit "yes" deletes. LangGraph re-runs this node from the top on resume,
        so nothing before interrupt() may have side effects."""
        reports = [Report(**r) for r in state["pending_delete"]]
        answer = interrupt(
            f"I am about to permanently delete {len(reports)} report(s):\n{format_reports(reports)}\n"
            'Reply "yes" to delete them. Any other reply cancels.'
        )
        if answer.strip().lower() != "yes":
            return {"reply": "Cancelled. Nothing was deleted.", "outcome": "cancelled"}
        thread_id = config["configurable"]["thread_id"]
        deleted = self.store.delete(state["user_id"], thread_id, reports, request=state["question"])
        return {"reply": f"Deleted {deleted} report(s).", "outcome": "deleted"}

    def respond(self, state: AgentState) -> dict:
        return {
            "history": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": state["reply"]},
            ]
        }

    def _matching_reports(self, state: AgentState, config: RunnableConfig) -> list[Report]:
        thread_id = config["configurable"]["thread_id"] if state["this_conversation"] else None
        return self.store.find(state["user_id"], state["report_keyword"], thread_id)


def describe_schema(warehouse: Warehouse, tables: tuple[str, ...], hidden: tuple[str, ...]) -> str:
    """PII columns are left out, so the model never asks for them."""
    lines = []
    for table in tables:
        schema = warehouse.table_schema(table)
        columns = ", ".join(f"{c['name']} {c['type']}" for c in schema if c["name"] not in hidden)
        lines.append(f"{table}: {columns}")
    return "\n".join(lines)


def escape_tags(text: str) -> str:
    """User text must not open or close the tags that delimit it in a prompt."""
    return PROMPT_TAGS.sub("&lt;", text)


def user_text(state: AgentState) -> str:
    return escape_tags(state["question"])


def recent_history(state: AgentState) -> str:
    """One tagged message per entry, so a message cannot fake an extra turn."""
    messages = state.get("history", [])[-HISTORY_MESSAGES:]
    return "\n".join(f"<{m['role']}>{escape_tags(m['content'])}</{m['role']}>" for m in messages) or "(empty)"


def format_reports(reports: list[Report]) -> str:
    return "\n".join(f'  #{r.id} "{r.title}" ({r.created_at})' for r in reports)


def format_results(results: list[dict]) -> str:
    return "\n\n".join(f"Sub-question: {r['question']}\n{r['data']}" for r in results)
