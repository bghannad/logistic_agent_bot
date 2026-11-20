"""Groundedness eval: for each test message, does the bot only state claims backed by
what its specialists were actually given — the KB chunks the compliance specialist
retrieved and the shipment data the tracking specialist looked up — or does it slip
in outside knowledge?

Method: LLM-as-judge. The judge sees a text plus the source text it should be based
on, and flags any claim not backed by that source. The sources are read from the
artifacts on the specialists' ToolMessages (exactly what this run used; nothing is
retrieved or looked up again). Four checks per message:
- end-to-end: final reply vs everything the specialists were given
- compliance specialist: its answer vs its retrieved chunks
- tracking specialist: its answer vs its lookup results
- supervisor: final reply vs the specialists' answers
The per-hop checks locate where an unsupported claim came in. The source list
appended to the reply is removed before judging: it is built in code from chunk
metadata, not written by the LLM. The judge never writes its own answer to compare
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
from agents.specialists import SOURCES_MARKER, chunk_metadata, specialist_results
from eval.routing_eval import called_specialists

load_dotenv()
MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = REPO_ROOT / "data" / "eval" / "groundedness_cases.jsonl"

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


COMPLIANCE = "call_compliance_agent"
TRACKING = "call_tracking_agent"
ESCALATION = "call_escalation_agent"


def kb_text(chunk_ids: list[str]) -> str:
    metadata = chunk_metadata()
    return "\n\n".join(f"[{cid}] {metadata[cid]['text']}" for cid in dict.fromkeys(chunk_ids))


def lookup_text(lookups: list[str]) -> str:
    return "\n".join(lookups) if lookups else "(no shipment lookup was made)"


def original_sources(messages: list) -> str:
    """Everything the specialists were given in this run: chunks, lookup results, escalation decision."""
    parts = []
    chunk_ids = [cid for _, ids in specialist_results(messages, COMPLIANCE) for cid in ids]
    if chunk_ids:
        parts.append(f"KB EXCERPTS:\n{kb_text(chunk_ids)}")
    lookups = [x for _, found in specialist_results(messages, TRACKING) for x in found]
    if lookups:
        parts.append(f"SHIPMENT LOOKUPS:\n{lookup_text(lookups)}")
    for decision, _ in specialist_results(messages, ESCALATION):
        parts.append(f"ESCALATION SPECIALIST DECISION:\n{decision}")
    return "\n\n".join(parts)


def specialist_answers(messages: list) -> str:
    """What the supervisor was given: the specialists' answers."""
    parts = []
    for name, label in [(COMPLIANCE, "COMPLIANCE"), (TRACKING, "TRACKING"), (ESCALATION, "ESCALATION")]:
        for answer, _ in specialist_results(messages, name):
            parts.append(f"{label} SPECIALIST ANSWER:\n{answer}")
    return "\n\n".join(parts)


def citation_check(messages: list) -> tuple[int, int]:
    """(chunk ids the compliance specialist cited, of those how many it was NOT given)."""
    cited = invented = 0
    for answer, retrieved_ids in specialist_results(messages, COMPLIANCE):
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

    # per check: [grounded, graded]
    totals = {"end-to-end": [0, 0], "compliance specialist": [0, 0],
              "tracking specialist": [0, 0], "supervisor": [0, 0]}
    total_cited = total_invented = 0

    def run_check(name: str, source_text: str, text: str) -> bool:
        is_grounded, verdict_text = judge(judge_llm, source_text, text)
        totals[name][0] += is_grounded
        totals[name][1] += 1
        print(f"  [{name}] {verdict_text}")
        return is_grounded

    for case in cases:
        result = graph.invoke({"messages": [HumanMessage(content=case["message"])]})
        messages = result["messages"]
        specialists = called_specialists(messages)
        reply = messages[-1].content
        reply_text = reply.split(SOURCES_MARKER)[0]  # judge the LLM's words, not the code-built source list

        print(f"[{case['id']}] ({case['category']}) {case['message']}")
        print(f"  reply: {reply}")

        if not specialists & {COMPLIANCE, TRACKING}:
            print("  skipped — no compliance/tracking specialist called, nothing to check\n")
            continue

        # 1. End-to-end: final reply vs everything the specialists were given.
        end_to_end_ok = run_check("end-to-end", original_sources(messages), reply_text)

        # 2. Each hop on its own, to locate where an unsupported claim came in.
        failed_hops = []
        for answer, chunk_ids in specialist_results(messages, COMPLIANCE):
            if not run_check("compliance specialist", f"KB EXCERPTS:\n{kb_text(chunk_ids)}", answer):
                failed_hops.append("compliance specialist")
        for answer, lookups in specialist_results(messages, TRACKING):
            if not run_check("tracking specialist", f"SHIPMENT LOOKUPS:\n{lookup_text(lookups)}", answer):
                failed_hops.append("tracking specialist")
        if not run_check("supervisor", specialist_answers(messages), reply_text):
            failed_hops.append("supervisor")

        if not end_to_end_ok:
            where = ", ".join(failed_hops) if failed_hops else "not located (the per-hop checks all passed)"
            print(f"  -> unsupported claim introduced by: {where}")

        cited, invented = citation_check(messages)
        total_cited += cited
        total_invented += invented
        print(f"  citations: {cited} chunk id(s) cited by the compliance specialist, "
              f"{invented} not among its retrieved chunks\n")

    print("Grounded (one run):")
    for name, (ok, n) in totals.items():
        if n:
            print(f"  {name:<22} {ok}/{n} ({ok / n:.0%})")
    print(f"Citations: {total_cited} cited chunk ids, {total_invented} not among the retrieved chunks")


if __name__ == "__main__":
    main()
