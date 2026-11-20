"""Source list and specialist artifacts — no OpenAI call, no retrieval."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents import specialists
from agents.specialists import SOURCES_MARKER, chunk_metadata, format_sources

CITED = "cbp_prohibited_restricted_items__014"
NOT_CITED = "cbp_prohibited_restricted_items__037"


def _turn(answer: str, retrieved: list[str]) -> list:
    return [
        HumanMessage(content="Can I bring back ivory?"),
        AIMessage(content="", tool_calls=[{"name": "call_compliance_agent", "args": {}, "id": "c1"}]),
        ToolMessage(content=answer, artifact=retrieved, name="call_compliance_agent", tool_call_id="c1"),
    ]


def test_sources_list_only_cited_chunks_with_title_and_url():
    sources = format_sources(_turn(f"Ivory is restricted [{CITED}].", [CITED, NOT_CITED]))
    assert sources.startswith(SOURCES_MARKER)
    assert chunk_metadata()[CITED]["url"] in sources
    assert chunk_metadata()[CITED]["title"] in sources
    assert sources.count("\n- ") == 1


def test_no_sources_when_nothing_cited():
    assert format_sources(_turn("The knowledge base doesn't cover this.", [CITED])) == ""


def test_sources_ignore_earlier_turns():
    messages = _turn(f"Ivory is restricted [{CITED}].", [CITED]) + [HumanMessage(content="thanks")]
    assert format_sources(messages) == ""


def test_compliance_tool_returns_retrieved_ids_as_artifact(monkeypatch):
    monkeypatch.setattr(specialists, "compliance_agent_run", lambda _msg: ("an answer", [CITED, NOT_CITED]))
    state = {"messages": [
        HumanMessage(content="Can I bring back ivory?"),
        AIMessage(content="", tool_calls=[{"name": "call_compliance_agent", "args": {}, "id": "c1"}]),
    ]}
    tool_call = {"type": "tool_call", "name": "call_compliance_agent", "args": {"state": state}, "id": "c1"}
    tool_message = specialists.call_compliance_agent.invoke(tool_call)
    assert tool_message.content == "an answer"
    assert tool_message.artifact == [CITED, NOT_CITED]


def test_tracking_tool_returns_lookup_results_as_artifact(monkeypatch):
    monkeypatch.setattr(specialists, "tracking_agent_run", lambda _msg: ("in transit", ["TRK1001: {...}"]))
    state = {"messages": [
        HumanMessage(content="Where is TRK1001?"),
        AIMessage(content="", tool_calls=[{"name": "call_tracking_agent", "args": {}, "id": "t1"}]),
    ]}
    tool_call = {"type": "tool_call", "name": "call_tracking_agent", "args": {"state": state}, "id": "t1"}
    tool_message = specialists.call_tracking_agent.invoke(tool_call)
    assert tool_message.content == "in transit"
    assert tool_message.artifact == ["TRK1001: {...}"]
