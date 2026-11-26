"""The compiled graph carries a hard loop limit (no OpenAI call: the graph is only built)."""

from agents.graph import RECURSION_LIMIT, build_graph


def test_build_graph_binds_recursion_limit():
    assert RECURSION_LIMIT == 10
    assert build_graph().config["recursion_limit"] == RECURSION_LIMIT
