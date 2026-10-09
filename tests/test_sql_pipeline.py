"""Plan -> generate SQL -> execute -> answer; reports, schema questions, follow-ups."""

from tests.conftest import turn, unscoped
from tests.fakes import classified, planned, report, sql

REVENUE_SQL = "SELECT category, SUM(sale_price) AS revenue FROM order_items GROUP BY category"
REVENUE_ROWS = [{"category": "Jeans", "revenue": 1200.0}, {"category": "Pants", "revenue": 800.0}]


def test_analysis_question_runs_sql_and_answers_from_rows(make_agent):
    agent = make_agent(
        llm=[classified("analysis"), planned("Revenue by category"), sql(REVENUE_SQL), "Jeans lead with $1,200."],
        bq=[REVENUE_ROWS],
    )
    result = turn(agent, "Which categories bring the most revenue?")

    assert result["reply"] == "Jeans lead with $1,200."
    assert [unscoped(q) for q in agent.bq.queries] == [REVENUE_SQL]
    assert "Jeans,1200.0" in agent.llm.calls[-1].prompt
    assert "only for these product categories: Jeans, Pants" in agent.llm.calls[-1].prompt


def test_multi_step_question_runs_one_query_per_sub_question(make_agent):
    agent = make_agent(
        llm=[
            classified("analysis"),
            planned("Average spend of users in California", "Average spend of users in Texas"),
            sql("SELECT 1 AS ca"),
            sql("SELECT 2 AS tx"),
            "California users spend less than Texas users.",
        ],
        bq=[[{"avg_spend": 50.0}], [{"avg_spend": 70.0}]],
    )
    turn(agent, "Why are users in California underspending compared to Texas?")

    assert [unscoped(q) for q in agent.bq.queries] == ["SELECT 1 AS ca", "SELECT 2 AS tx"]
    # Earlier results feed later steps.
    assert "Average spend of users in California\navg_spend\n50.0" in agent.llm.calls[3].prompt


def test_plan_is_capped_at_max_steps(make_agent, cfg):
    steps = [f"step {i}" for i in range(cfg.max_steps + 2)]
    agent = make_agent(
        llm=[classified("analysis"), planned(*steps), *[sql(f"SELECT {i}") for i in range(cfg.max_steps)], "Done."],
        bq=[[{"x": i}] for i in range(cfg.max_steps)],
    )
    turn(agent, "A very broad question")
    assert len(agent.bq.queries) == cfg.max_steps


def test_report_is_saved_and_the_user_is_told_its_title(make_agent):
    agent = make_agent(
        llm=[
            classified("report"),
            planned("Revenue by category in Q1"),
            sql(REVENUE_SQL),
            report("Q1 revenue review", "## Summary\nJeans lead.\n## Action items\n- Stock more jeans."),
        ],
        bq=[REVENUE_ROWS],
    )
    result = turn(agent, "Create a Q1 report with insights and action items for Q2")

    assert result["reply"].endswith('Saved as report #1: "Q1 revenue review"')
    saved = agent.store.find("alice")
    assert [(r.id, r.title) for r in saved] == [(1, "Q1 revenue review")]
    assert agent.store.find("bob") == []


def test_schema_question_is_answered_without_querying_data(make_agent):
    agent = make_agent(llm=[classified("schema"), "We have orders, order items, products and users."])
    result = turn(agent, "What data do we have?")

    assert result["reply"] == "We have orders, order items, products and users."
    assert agent.bq.queries == []
    assert "order_items: id INTEGER" in agent.llm.calls[1].prompt


def test_follow_up_is_answered_from_history_without_new_queries(make_agent):
    agent = make_agent(
        llm=[
            classified("analysis"), planned("Revenue by category"), sql(REVENUE_SQL), "Jeans lead with $1,200.",
            classified("discussion"), "Jeans earn 50% more than Pants.",
        ],
        bq=[REVENUE_ROWS],
    )
    turn(agent, "Revenue by category?")
    result = turn(agent, "How much more do Jeans earn than Pants?")

    assert result["reply"] == "Jeans earn 50% more than Pants."
    assert len(agent.bq.queries) == 1
    assert "Jeans lead with $1,200." in agent.llm.calls[-1].prompt
