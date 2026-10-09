"""Self-correction, retry budgets, backoff, fallback model, BigQuery limits, the turn time
budget, and clear messages instead of crashes."""

import json

import pytest
from google.api_core import exceptions as gcp

from agent.chat import UNAVAILABLE, UNEXPECTED
from agent.config import load_prompts
from agent.tracing import summarize
from tests.conftest import chat, turn
from tests.fakes import (
    EMPTY,
    bq_bytes_limit,
    bq_job_timeout,
    bq_query_error,
    bq_unavailable,
    classified,
    gemini_blocked,
    gemini_daily_quota,
    gemini_error,
    gemini_timeout,
    planned,
    sql,
)

ROWS = [{"revenue": 100.0}]


def analysis_script(*sqls: str, answer: str = "Revenue was $100.") -> list:
    return [classified("analysis"), planned("Total revenue"), *[sql(q) for q in sqls], answer]


def test_sql_error_goes_back_to_the_model_and_the_fix_is_used(make_agent):
    agent = make_agent(
        llm=analysis_script("SELECT SUM(price) FROM order_items", "SELECT SUM(sale_price) FROM order_items"),
        bq=[bq_query_error("Unrecognized name: price at [1:12]"), ROWS],
    )
    result = turn(agent, "Total revenue?")

    assert result["reply"] == "Revenue was $100."
    retry_prompt = agent.llm.calls[3].prompt
    assert "Unrecognized name: price" in retry_prompt
    assert "SELECT SUM(price) FROM order_items" in retry_prompt  # the model sees its own query, not the scoped one


def test_empty_result_is_retried_once_then_accepted(make_agent):
    agent = make_agent(llm=analysis_script("SELECT 1", "SELECT 2", answer="No sales matched."), bq=[EMPTY, EMPTY])
    result = turn(agent, "Revenue for category 'jeans'?")

    assert result["reply"] == "No sales matched."
    assert "returned no rows" in agent.llm.calls[3].prompt
    assert "(no rows)" in agent.llm.calls[-1].prompt
    assert len(agent.bq.queries) == 2


def test_gives_up_after_max_attempts_with_a_clear_message(make_agent, cfg):
    agent = make_agent(
        llm=[classified("analysis"), planned("Total revenue"), *[sql("SELECT bad") for _ in range(cfg.max_attempts)]],
        bq=[bq_query_error() for _ in range(cfg.max_attempts)],
    )
    result = turn(agent, "Total revenue?")

    assert result["reply"].startswith(f'I could not get the data for "Total revenue" after {cfg.max_attempts} attempts')
    assert "Try rephrasing" in result["reply"]
    assert len(agent.llm.calls) == 2 + cfg.max_attempts  # no answer call


def test_query_over_the_bytes_limit_is_stopped_by_the_dry_run(make_agent, cfg):
    agent = make_agent(
        llm=[classified("analysis"), planned("Total revenue"), *[sql("SELECT 1") for _ in range(cfg.max_attempts)]],
        scan_bytes=cfg.max_bytes_billed + 1,
    )
    result = turn(agent, "Total revenue?")

    assert agent.bq.queries == []
    assert len(agent.bq.dry_runs) == cfg.max_attempts
    assert "over the 1.00 GB limit" in result["reply"]


def test_queries_run_with_bytes_cap_timeout_and_row_cap(make_agent, cfg):
    agent = make_agent(llm=analysis_script("SELECT 1"), bq=[ROWS])
    turn(agent, "Total revenue?")

    [job_config] = agent.bq.job_configs
    assert job_config.maximum_bytes_billed == cfg.max_bytes_billed
    assert int(job_config.job_timeout_ms) == cfg.bq_timeout_s * 1000
    assert agent.bq.max_results == [cfg.max_rows + 1]  # the extra row reveals truncation


@pytest.mark.parametrize("error", [bq_bytes_limit(), bq_job_timeout()], ids=["bytes limit (500)", "timeout (499)"])
def test_query_limits_go_back_to_the_model_not_retried_as_outages(make_agent, error):
    agent = make_agent(llm=analysis_script("SELECT 1", "SELECT 2"), bq=[error, ROWS])
    assert chat(agent).send("Total revenue?") == "Revenue was $100."

    assert agent.sleeps == []
    assert error.message[:40] in agent.llm.calls[3].prompt


def test_gemini_503_is_retried_with_backoff_on_the_same_model(make_agent):
    agent = make_agent(llm=[gemini_error(503), classified("discussion"), "Flat."])
    assert chat(agent).send("And last year?") == "Flat."

    assert [c.model for c in agent.llm.calls[:2]] == ["primary-model", "primary-model"]
    assert len(agent.sleeps) == 1 and 1.0 <= agent.sleeps[0] < 1.5


def test_gemini_429_switches_to_the_fallback_model(make_agent):
    agent = make_agent(llm=[gemini_error(429), gemini_error(429), classified("discussion"), "Flat."])
    assert chat(agent).send("And last year?") == "Flat."
    assert [c.model for c in agent.llm.calls[:3]] == ["primary-model", "primary-model", "fallback-model"]


def test_daily_quota_429_skips_backoff_and_goes_straight_to_the_fallback(make_agent):
    agent = make_agent(llm=[gemini_daily_quota(), classified("discussion"), "Flat."])
    assert chat(agent).send("And last year?") == "Flat."
    assert [c.model for c in agent.llm.calls[:2]] == ["primary-model", "fallback-model"]
    assert agent.sleeps == []


def test_gemini_down_on_both_models_gives_a_clear_message_and_the_chat_continues(make_agent):
    agent = make_agent(llm=[gemini_timeout()] * 2 + [classified("discussion"), "Flat."])
    conversation = chat(agent)

    assert conversation.send("Revenue by month?").startswith(UNAVAILABLE["Gemini"] + " (trace id: ")
    assert [c.model for c in agent.llm.calls[:2]] == ["primary-model", "fallback-model"]
    assert conversation.send("And last year?") == "Flat."
    history = agent.graph.get_state(conversation.config).values["history"]
    assert [m["content"] for m in history] == ["And last year?", "Flat."]  # no trace of the failed turn


def test_gemini_503_on_both_models_uses_two_attempts_each_then_stops(make_agent):
    agent = make_agent(llm=[gemini_error(503)] * 4)
    assert chat(agent).send("Revenue by month?").startswith(UNAVAILABLE["Gemini"])
    assert [c.model for c in agent.llm.calls] == ["primary-model"] * 2 + ["fallback-model"] * 2


@pytest.mark.parametrize("error", [gemini_timeout(), gemini_error(504)], ids=["client timeout", "504"])
def test_a_timeout_on_the_primary_goes_straight_to_the_fallback(make_agent, error):
    agent = make_agent(llm=[error, classified("discussion"), "Flat."])
    assert chat(agent).send("And last year?") == "Flat."
    assert [c.model for c in agent.llm.calls[:2]] == ["primary-model", "fallback-model"]
    assert agent.sleeps == []


def test_an_outage_ends_the_turn_within_the_budget(make_agent, cfg):
    agent = make_agent(llm=[classified("discussion"), "Flat."])
    agent.llm.latency_s = 45  # every request hangs past its timeout

    assert chat(agent).send("Revenue by month?").startswith(UNAVAILABLE["Gemini"])
    assert agent.clock.now == cfg.turn_budget_s  # 30 s primary timeout + 30 s left for the fallback
    assert [(c.model, c.timeout_s) for c in agent.llm.calls] == [("primary-model", 30), ("fallback-model", 30)]


def test_slow_steps_use_up_the_budget_and_the_last_request_gets_only_what_is_left(make_agent):
    agent = make_agent(llm=[classified("analysis"), planned("Total revenue"), sql("SELECT 1")])
    agent.llm.latency_s = 25  # each call succeeds, slowly

    assert chat(agent).send("Total revenue?").startswith(UNAVAILABLE["Gemini"])
    assert agent.clock.now == 60
    assert [c.timeout_s for c in agent.llm.calls] == [30, 30, 10]  # classify, plan, then 10 s left for SQL


def test_a_bigquery_job_may_not_outlast_the_turn(make_agent):
    agent = make_agent(llm=analysis_script("SELECT 1"), bq=[ROWS])
    agent.llm.latency_s = 15  # classify + plan + SQL = 45 s, so 15 s are left for the query

    assert chat(agent).send("Total revenue?") == "Revenue was $100."
    assert int(agent.bq.job_configs[0].job_timeout_ms) == 15_000


def test_non_transient_gemini_error_is_not_retried(make_agent):
    agent = make_agent(llm=[gemini_error(400)])
    assert chat(agent).send("Revenue by month?").startswith(UNEXPECTED)
    assert len(agent.llm.calls) == 1 and agent.sleeps == []


def test_bigquery_blip_is_retried(make_agent):
    agent = make_agent(llm=analysis_script("SELECT 1"), bq=[bq_unavailable(), ROWS])
    assert chat(agent).send("Total revenue?") == "Revenue was $100."
    assert len(agent.sleeps) == 1


def test_bigquery_request_that_the_client_library_already_retried_is_not_retried_again(make_agent):
    exhausted = gcp.RetryError("Timeout of 20.0s exceeded", cause=ConnectionError("connection refused"))
    agent = make_agent(llm=analysis_script("SELECT 1"), bq=[exhausted])
    assert chat(agent).send("Total revenue?").startswith(UNAVAILABLE["BigQuery"])
    assert agent.sleeps == [] and len(agent.bq.queries) == 1


def test_bigquery_down_gives_a_clear_message(make_agent):
    agent = make_agent(llm=analysis_script("SELECT 1"), bq=[bq_unavailable()] * 3)
    assert chat(agent).send("Total revenue?").startswith(UNAVAILABLE["BigQuery"])
    assert len(agent.bq.queries) == 3


def test_a_blocked_response_is_refused_and_logged_as_a_security_event(make_agent):
    agent = make_agent(llm=[gemini_blocked("SAFETY")])
    conversation = chat(agent)

    assert conversation.send("Some message Gemini's filters block") == load_prompts()["refusal"].strip()
    records = [json.loads(line) for line in agent.tracer.path.read_text().splitlines()]
    turn = next(r for r in records if r["type"] == "turn")
    event = next(r for r in records if r["type"] == "security")
    assert turn["outcome"] == "blocked" and event["reason"] == "prompt blocked (SAFETY)"
    assert agent.graph.get_state(conversation.config).values.get("history", []) == []
    assert "Refusals:   1" in summarize(agent.tracer.path)


def test_an_empty_response_is_handled_like_a_blocked_one(make_agent):
    agent = make_agent(llm=[classified("discussion"), None])
    assert chat(agent).send("And last year?") == load_prompts()["refusal"].strip()
    assert '"reason": "empty response"' in agent.tracer.path.read_text()


def test_structured_output_that_fails_validation_is_asked_for_once_more(make_agent):
    agent = make_agent(llm=["not json", classified("discussion"), "Flat."])
    assert chat(agent).send("And last year?") == "Flat."
    assert len(agent.llm.calls) == 3


def test_structured_output_invalid_twice_ends_the_turn_with_the_error_message(make_agent):
    agent = make_agent(llm=["not json", {"intent": "nonsense"}])
    assert chat(agent).send("And last year?").startswith(UNEXPECTED)


def test_an_empty_result_on_the_last_attempt_is_answered_as_no_data(make_agent, cfg):
    agent = make_agent(
        llm=[classified("analysis"), planned("Revenue for Bermuda"), *[sql(f"SELECT {i}") for i in range(3)],
             "No matching sales were found."],
        bq=[bq_query_error(), bq_query_error(), EMPTY],
    )
    result = turn(agent, "Revenue for Bermuda?")

    assert result["reply"] == "No matching sales were found." and result["outcome"] == "answered"
    assert "(no rows)" in agent.llm.calls[-1].prompt


def test_a_truncated_result_is_flagged_to_the_model(make_agent, cfg):
    rows = [{"product": f"p{i}", "revenue": i} for i in range(cfg.max_rows + 1)]
    agent = make_agent(llm=analysis_script("SELECT product, revenue FROM order_items"), bq=[rows])
    turn(agent, "List product revenue")

    answer_prompt = agent.llm.calls[-1].prompt
    assert f"p{cfg.max_rows - 1}," in answer_prompt and f"p{cfg.max_rows}," not in answer_prompt
    assert f"truncated: only the first {cfg.max_rows} rows are shown" in answer_prompt
