"""Listing and deleting saved reports: an exact list of the user's own reports, an explicit
"yes", an audit entry."""

import json

import pytest

from tests.conftest import chat
from tests.fakes import classified


def delete_request(keyword: str = "", this_conversation: bool = False) -> dict:
    return classified("manage_reports", report_action="delete", report_keyword=keyword,
                      this_conversation=this_conversation)


@pytest.fixture
def seeded(make_agent):
    """alice has three reports (two in thread t1, one in t2); bob has one that also mentions Acme."""

    def make(llm=()):
        agent = make_agent(llm=llm)
        agent.store.save("alice", "t1", "Acme churn review", "Acme churn rose 4%.")
        agent.store.save("alice", "t1", "Q1 revenue", "Revenue grew 12%.")
        agent.store.save("alice", "t2", "Q2 plan for Acme", "Grow Acme accounts.")
        agent.store.save("bob", "t9", "Acme sweaters", "Acme bought sweaters.")
        return agent

    return make


def titles(agent, user: str) -> list[str]:
    return [r.title for r in agent.store.find(user)]


def audit_rows(agent) -> list[dict]:
    rows = agent.store.conn.execute("SELECT user_id, thread_id, action, details FROM audit_log").fetchall()
    return [{"user_id": u, "thread_id": t, "action": a, **json.loads(d)} for u, t, a, d in rows]


def test_delete_shows_the_exact_list_of_own_reports_and_waits(seeded):
    agent = seeded(llm=[delete_request("acme")])
    conversation = chat(agent)

    question = conversation.send("Delete all reports mentioning Acme")

    assert question.startswith("I am about to permanently delete 2 report(s):")
    assert '#1 "Acme churn review"' in question and '#3 "Q2 plan for Acme"' in question
    assert "Acme sweaters" not in question
    assert conversation.awaiting_confirmation
    assert len(titles(agent, "alice")) == 3


def test_yes_deletes_exactly_the_listed_reports_and_writes_the_audit_log(seeded):
    agent = seeded(llm=[delete_request("acme")])
    conversation = chat(agent)
    conversation.send("Delete all reports mentioning Acme")

    assert conversation.send("yes") == "Deleted 2 report(s)."
    assert titles(agent, "alice") == ["Q1 revenue"]
    assert titles(agent, "bob") == ["Acme sweaters"]
    [entry] = audit_rows(agent)
    assert entry["user_id"] == "alice" and entry["thread_id"] == "t1" and entry["action"] == "delete_reports"
    assert entry["request"] == "Delete all reports mentioning Acme"
    assert [r["id"] for r in entry["reports"]] == [1, 3] and entry["deleted"] == 2


@pytest.mark.parametrize("answer", ["no", "y", "yes please", "sure", "", "delete them"])
def test_anything_but_yes_cancels(seeded, answer):
    agent = seeded(llm=[delete_request("acme")])
    conversation = chat(agent)
    conversation.send("Delete all reports mentioning Acme")

    assert conversation.send(answer) == "Cancelled. Nothing was deleted."
    assert len(titles(agent, "alice")) == 3
    assert audit_rows(agent) == []


def test_yes_is_case_and_whitespace_insensitive(seeded):
    agent = seeded(llm=[delete_request("acme")])
    conversation = chat(agent)
    conversation.send("Delete all reports mentioning Acme")
    assert conversation.send("  YES ") == "Deleted 2 report(s)."


def test_delete_this_conversation_only_targets_this_thread(seeded):
    agent = seeded(llm=[delete_request(this_conversation=True)])
    conversation = chat(agent, thread="t1")
    question = conversation.send("Delete all the reports we made in this conversation")

    assert "2 report(s)" in question and "Q2 plan for Acme" not in question
    conversation.send("yes")
    assert titles(agent, "alice") == ["Q2 plan for Acme"]


def test_nothing_to_delete_does_not_ask(seeded):
    agent = seeded(llm=[delete_request("globex")])
    conversation = chat(agent)
    assert conversation.send("Delete reports about Globex") == (
        "You have no saved reports matching that, so nothing was deleted."
    )
    assert not conversation.awaiting_confirmation


def test_a_confirmation_from_an_earlier_session_is_not_reused(seeded):
    agent = seeded(llm=[delete_request("acme"), classified("discussion", allowed=False)])
    chat(agent).send("Delete all reports mentioning Acme")  # the session ends here

    new_session = chat(agent)
    new_session.send("yes")
    assert len(titles(agent, "alice")) == 3


def test_list_shows_only_own_reports(seeded):
    agent = seeded(llm=[classified("manage_reports", report_action="list")])
    reply = chat(agent).send("Show my reports")

    assert reply.startswith("Your saved reports (3):")
    assert "Acme sweaters" not in reply


def test_store_never_deletes_another_users_report(seeded):
    agent = seeded()
    bobs = agent.store.find("bob")
    assert agent.store.delete("alice", "t1", bobs, request="crafted") == 0
    assert titles(agent, "bob") == ["Acme sweaters"]
