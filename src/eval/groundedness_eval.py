"""Groundedness eval: for each test message, does the supervisor's final reply only
state claims actually backed by (a) the KB excerpts the compliance specialist
retrieved and/or (b) the mocked shipment data the tracking specialist used — or
does it slip in outside knowledge?

Method: LLM-as-judge. The judge sees the final reply plus the real source text
and flags any claim in the reply not backed by that source text. The KB excerpts
are exactly the chunks the compliance specialist was given in this run (read from
the artifact on its ToolMessage, not retrieved again). The source list appended to
the reply is removed before judging: it is built in code from chunk metadata, not
written by the LLM. The judge never writes its own answer to compare
against — it only grades ours. This is a faithfulness/groundedness check, not an
answer-correctness check, because there's no single correct reference answer to
compare against for most of these questions.

The judge can itself be wrong — hand-verify a sample of its verdicts rather than
trust it blindly.

Usage (from repo root, with .venv active):
    python -m eval.groundedness_eval
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agents.graph import build_graph
from agents.specialists import SOURCES_MARKER, chunk_metadata, compliance_results
from agents.tools import check_shipment_status
from eval.routing_eval import called_specialists

load_dotenv()
MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = REPO_ROOT / "data" / "eval" / "groundedness_cases.jsonl"

TRACKING_REF_RE = re.compile(r"TRK\d+")
CHUNK_ID_RE = re.compile(r"\b[a-z0-9_]+__\d{3}\b")

JUDGE_SYSTEM_PROMPT = (
    "You are a strict fact-checker reviewing a customer-support assistant's reply. "
    "You are given SOURCE TEXT (the only information the assistant was allowed to "
    "use) and the assistant's REPLY. Read the SOURCE TEXT carefully and completely "
    "before judging — check every field in it, not just the first sentence. Check "
    "every factual claim in the REPLY. A claim is UNSUPPORTED only if it states "
    "something as fact about the world (a rule, a status, a number, a date) that "
    "is not stated in, or directly implied by, the SOURCE TEXT — including "
    "plausible-sounding outside knowledge, invented specifics, or more certainty "
    "than the source actually gives.\n\n"
    "Two things are NEVER unsupported claims, even though they aren't literally "
    "written in the source — never flag either of these:\n"
    "1. The assistant honestly saying the source doesn't cover something, or that "
    "it couldn't find relevant information. That is a true observation about the "
    "source, not a claim that itself needs backing.\n"
    "2. Generic, non-factual advice or recommendations — e.g. suggesting the "
    "customer contact CBP, a customs broker, the sender, or check local "
    "regulations. This is boilerplate guidance, not a factual claim about policy.\n\n"
    "Only flag claims that assert a specific fact as true without support.\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: GROUNDED\n"
    "or\n"
    "VERDICT: UNGROUNDED\n"
    "UNSUPPORTED CLAIMS:\n"
    "- <claim>\n"
    "- <claim>"
)


def load_cases() -> list[dict]:
    with CASES_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_sources(message: str, messages: list, specialists: set[str]) -> str:
    sources = []
    metadata = chunk_metadata()

    retrieved_ids = [cid for _, ids in compliance_results(messages) for cid in ids]
    if retrieved_ids:
        unique_ids = list(dict.fromkeys(retrieved_ids))
        excerpt_text = "\n\n".join(f"[{cid}] {metadata[cid]['text']}" for cid in unique_ids)
        sources.append(f"KB EXCERPTS:\n{excerpt_text}")

    if "call_tracking_agent" in specialists:
        match = TRACKING_REF_RE.search(message)
        if match:
            result = check_shipment_status.invoke({"tracking_ref": match.group()})
            sources.append(f"MOCKED SHIPMENT LOOKUP for {match.group()}: {result}")

    return "\n\n".join(sources)


def citation_check(messages: list) -> tuple[int, int]:
    """(chunk ids the compliance specialist cited, of those how many it was NOT given)."""
    cited = invented = 0
    for answer, retrieved_ids in compliance_results(messages):
        for cid in set(CHUNK_ID_RE.findall(answer)):
            cited += 1
            invented += cid not in retrieved_ids
    return cited, invented


def judge(llm: ChatOpenAI, source_text: str, reply: str) -> tuple[bool, str]:
    prompt = f"SOURCE TEXT:\n{source_text}\n\nREPLY:\n{reply}"
    response = llm.invoke([
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    verdict_text = response.content
    is_grounded = verdict_text.strip().startswith("VERDICT: GROUNDED")
    return is_grounded, verdict_text


def main() -> None:
    cases = load_cases()
    graph = build_graph()
    judge_llm = ChatOpenAI(model=MODEL, temperature=0)

    graded = 0
    grounded_count = 0
    total_cited = 0
    total_invented = 0

    for case in cases:
        result = graph.invoke({"messages": [HumanMessage(content=case["message"])]})
        specialists = called_specialists(result["messages"])
        reply = result["messages"][-1].content
        reply_text = reply.split(SOURCES_MARKER)[0]  # judge the LLM's words, not the code-built source list

        print(f"[{case['id']}] ({case['category']}) {case['message']}")
        print(f"  reply: {reply}")

        if not specialists & {"call_compliance_agent", "call_tracking_agent"}:
            print("  skipped — no compliance/tracking specialist called, nothing to check\n")
            continue

        source_text = build_sources(case["message"], result["messages"], specialists)
        is_grounded, verdict_text = judge(judge_llm, source_text, reply_text)
        graded += 1
        grounded_count += is_grounded

        cited, invented = citation_check(result["messages"])
        total_cited += cited
        total_invented += invented

        print(f"  {verdict_text}")
        print(f"  citations: {cited} chunk id(s) cited by the specialist, {invented} not among its retrieved chunks\n")

    if graded:
        print(f"Grounded: {grounded_count}/{graded} graded cases ({grounded_count / graded:.0%})")
        print(f"Citations: {total_cited} cited chunk ids, {total_invented} not among the retrieved chunks")
    else:
        print("No cases were graded.")


if __name__ == "__main__":
    main()
