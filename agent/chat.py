"""One user's conversation: runs each turn, traces it, and turns any failure into a clear
message so the chat never crashes."""

from langgraph.types import Command

from agent.config import load_prompts
from agent.llm import NoAnswer
from agent.pii import mask_pii
from agent.resilience import ServiceUnavailable, TurnBudget
from agent.tracing import Tracer, error_details

UNAVAILABLE = {
    "Gemini": "The AI service is busy or unavailable right now, so I could not finish this request. "
    "Please try again in a minute.",
    "BigQuery": "The sales database is unavailable right now, so I could not finish this request. "
    "Please try again in a few minutes.",
}
UNEXPECTED = "Something went wrong on my side and the request was not completed. Please try again."


class Chat:
    def __init__(self, graph, tracer: Tracer, user_id: str, thread_id: str, budget: TurnBudget) -> None:
        self.graph = graph
        self.tracer = tracer
        self.budget = budget  # shared with the LLM gateway and the warehouse
        self.user_id = user_id
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}
        # A confirmation is valid only in the session that showed it: a question left
        # pending by an earlier session is dropped, and the next message starts a new turn.
        self.awaiting_confirmation = False

    def send(self, text: str) -> str:
        """A new message, or the answer to a pending confirmation."""
        # Masked on the way in, so PII the user types never reaches the state, history,
        # data/agent.db or a prompt.
        text = mask_pii(text)
        trace_id = self.tracer.start_turn()
        self.budget.start()
        turn = {"user_id": self.user_id, "thread_id": self.thread_id, "message": text,
                "confirmation_answer": self.awaiting_confirmation}
        if self.awaiting_confirmation:
            graph_input = Command(resume=text)
        else:
            graph_input = {"user_id": self.user_id, "question": text}
        self.awaiting_confirmation = False

        try:
            result = self.graph.invoke(graph_input, self.config)
        except NoAnswer as e:  # usually Gemini's safety filters: treated as a refusal
            self.tracer.end_turn(**turn, outcome="blocked", reason=e.reason)
            self.tracer.security_event(**turn, reason=e.reason)
            return load_prompts()["refusal"].strip()
        except ServiceUnavailable as e:
            self.tracer.end_turn(**turn, outcome="unavailable", service=e.service, **error_details(e))
            return f"{UNAVAILABLE[e.service]} (trace id: {trace_id})"
        except Exception as e:
            self.tracer.end_turn(**turn, outcome="error", **error_details(e))
            return f"{UNEXPECTED} (trace id: {trace_id})"

        if pending := result.get("__interrupt__"):
            self.awaiting_confirmation = True
            self.tracer.end_turn(**turn, intent=result["intent"], outcome="awaiting_confirmation")
            return pending[0].value

        self.tracer.end_turn(**turn, intent=result["intent"], outcome=result["outcome"])
        if result["outcome"] == "refused":
            self.tracer.security_event(**turn, reason=result["guard_reason"])
        return result["reply"]
