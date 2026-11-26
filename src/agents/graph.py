"""The supervisor graph: one LLM with the three specialists bound as tools.

Standard LangGraph tool-calling loop (ToolNode + tools_condition), just applied to
specialist agents instead of plain functions:

    supervisor --(tool calls present)--> specialists --> supervisor --(none)--> END

The model itself decides whether to call zero, one, or more than one specialist —
routing is decided by the model at runtime, not by hardcoded rules.

build_graph() takes an optional checkpointer (see agents/converse.py for the
multi-turn chatbot that uses one); with none, it's single-turn, as used by chat.py.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.prompts import get_prompt_template
from agents.specialists import (
    call_compliance_agent,
    call_escalation_agent,
    call_tracking_agent,
    format_sources,
)

load_dotenv()
MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")

# Hard stop for the supervisor <-> specialists loop. Every node run is one step, so 10
# allows at most 5 supervisor/specialist rounds per customer message. Kept even on
# purpose: the 10th step is always the specialists node, so a stopped run never leaves
# the supervisor's tool calls unanswered in the conversation's saved history.
RECURSION_LIMIT = 10
FALLBACK_REPLY = (
    "Sorry, I couldn't resolve this automatically. "
    "Please contact our support team so a colleague can help you."
)

_tools = [call_tracking_agent, call_compliance_agent, call_escalation_agent]
_supervisor_llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(_tools)


def supervisor_node(state: MessagesState) -> dict:
    chain = get_prompt_template("supervisor") | _supervisor_llm
    response = chain.invoke({"messages": state["messages"]})
    if not response.tool_calls:  # final reply: append the sources the specialist cited
        sources = format_sources(state["messages"])
        if sources:
            response = response.model_copy(update={"content": response.content + sources})
    return {"messages": [response]}


def build_graph(checkpointer=None):
    graph = StateGraph(MessagesState) #here you define the shape of the state (“it’s a list of messages”), not the content
    graph.add_node("supervisor", supervisor_node) # i have just 2 nodes : supervisor and specialists. 
    graph.add_node("specialists", ToolNode(_tools))
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", tools_condition, {"tools": "specialists", END: END}) # After supervisor, ask tools_condition where to go. If it says "tools", go to "specialists". If it says END, go to END.
    graph.add_edge("specialists", "supervisor") # after specialist always come back to supervisor
    # with_config binds the limit to every invoke() that doesn't pass its own
    return graph.compile(checkpointer=checkpointer).with_config(recursion_limit=RECURSION_LIMIT)
