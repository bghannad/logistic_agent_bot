# Logistic Agent Bot

**Status: Work in Progress.** The multi-agent system, retrieval pipeline, evaluation harness, tests, and CI are built and verified. Azure deployment has not happened yet (Docker is built and tested locally; the actual cloud deployment is the next step).

A multi-agent chatbot for delivery and customs support questions ("my package is stuck in customs," "can I import this item," "my order never arrived"). A supervisor agent routes each message to the specialist(s) it actually needs, rather than following one fixed sequence of steps.

## Architecture

```
customer message
      |
      v
   SUPERVISOR  --  decides which specialist(s) to call (zero, one, or more)
      |
  --------------------------------------------------
  |                  |                              |
TRACKING         COMPLIANCE                     ESCALATION
mocked tool      hybrid search (BM25 +          human-handoff
(shipment         embeddings) + reranking        judgment call,
 status lookup)   over a real CBP/FTC            summary written by
                  knowledge base                  the supervisor
  |                  |                              |
  --------------------------------------------------
                      |
                      v
                 SUPERVISOR
         merges specialist findings into
         one reply, or asks a clarifying
          question if it needs more info
```

The model itself decides which specialists to call (real tool-calling, not a hardcoded `if`), The tracking and compliance specialists read the customer's original message from LangGraph's injected state rather than from a model-generated argument, so a compound question can't get silently split or paraphrased on its way to them. The escalation specialist instead receives a short summary written by the supervisor, because its decision depends on the whole conversation, not only the latest message.

When a reply uses the knowledge base, it ends with a source list (page title, section and URL) for the chunks the compliance specialist cited. The list is built in code from the chunk metadata, not written by the model, so links can't be invented. The compliance and tracking tools also return what they were given (the retrieved chunk ids, the raw lookup results) as LangChain tool artifacts, stored on their ToolMessages but not sent to the model. The groundedness eval reads these, so it judges each specialist's answer and the supervisor's final reply separately, against exactly what that run used.

## Tech stack

- **Orchestration**: LangGraph (supervisor + specialist agents, tool-calling, conversation memory via checkpointing)
- **LLM**: OpenAI `gpt-4o-mini`
- **Retrieval**: hybrid search — BM25 (keyword) + `text-embedding-3-small` (semantic), merged by Reciprocal Rank Fusion, then reranked by a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Knowledge base**: real, public-domain CBP and FTC documents (not self-authored), chunked with a heading-aware heuristic
- **Web**: FastAPI backend, a plain HTML/JS chat page
- **Observability** (optional): Langfuse — tracing and versioned prompt management
- **Testing**: pytest, GitHub Actions CI
- **Deployment**: Docker (built and tested locally); Azure Container Instances deployment is planned, not yet done

## Results

All three evaluations use small, hand-labeled test sets written against the real knowledge base and run through the real agent system.

**Retrieval** (18 hand-labeled questions, each with a known-correct chunk): Hit@5 17/18 (94%), Recall@5 0.92, Precision@5 0.26 against a ceiling of 0.29 for this test set. (Precision@5 is capped low by construction — each question has only 1–3 correct chunks out of 5 returned slots — so 0.26 is close to the best this metric could show here; Hit@5 and Recall@5 are the more meaningful numbers.)

**Routing** (18 hand-labeled customer messages, each with the specialist(s) that should be called): 17/18 (94%) exact match.

**Groundedness / hallucination check** (12 hand-labeled messages, LLM-as-judge): 11/12 after one round of fixing the judge's own prompt (its first pass had penalized the system for honestly saying "not in my knowledge base," and for generic advice, as if those were unsupported claims). The one remaining flag is borderline, not a clear invention: the reply said "gold items" where the source specifically says "gold coins, medals and bullion" — a mild broadening. Limitations: the judge prompt was tuned on the same 12 cases it scores, and replies vary between runs (on a separate run, one reply made a claim the bot had no way to verify), so this is a sample, not a guarantee. The judge's verdicts were checked by hand.

All test cases were written for this project, not drawn from real customer messages.

## Known limitations

- **Tracking data is mocked.** `check_shipment_status` returns deterministic fake data for four tracking numbers, not a real carrier API.
- **Eval sets are small** (12-18 cases each) and self-authored, as noted above.
- **Conversation memory is in-memory only.** It is lost on process restart; no persistent checkpointer is wired up yet.
- **No access control.** Anyone who can reach the running app can use it. An optional access-code gate is designed but not built, since it only matters once this has a public URL.
- **Knowledge base is US-import-focused** (CBP and FTC documents only) and does not cover other countries' import rules.

## Data sources

The knowledge base is built from real U.S. federal government publications (CBP and FTC), in the public domain under 17 U.S.C. § 105, not self-authored synthetic documents. Full source list, URLs, and retrieval dates: [`data/kb/sources/SOURCES.md`](data/kb/sources/SOURCES.md).

## Running locally

```
git clone git@github.com:bghannad/logistic_agent_bot.git
cd logistic_agent_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY at minimum
```

Single-turn CLI:
```
PYTHONPATH=src python3 -m agents.chat "my tracking number is TRK1001, where is it?"
```

Multi-turn chat in the terminal:
```
PYTHONPATH=src python3 -m agents.converse
```

Web UI:
```
PYTHONPATH=src uvicorn web.main:app --reload --app-dir src
```
then open `http://localhost:8000`.

## Tests

```
pip install -r requirements-dev.txt
pytest -q
```

No network calls, no API key spend — a dummy OpenAI key is set automatically for tests that only need one to construct, and any real Langfuse keys in `.env` are stripped for the test session.

## Langfuse (optional)

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` (free tier at [cloud.langfuse.com](https://cloud.langfuse.com)) to get:
- full traces of every conversation (which specialist was called, why, what it returned)
- versioned prompt management — prompts live in `prompts/*.txt`, sync them to Langfuse with:
  ```
  python scripts/sync_prompts.py          # sync
  python scripts/sync_prompts.py --dry-run  # preview only
  ```

Without these keys, everything works identically, just untraced, reading prompts from the local `.txt` files directly.

## Docker

```
docker build -t logistic_agent_bot .
docker run --env-file .env -p 8000:8000 logistic_agent_bot
```
Image is approximately 2.2 GB (torch, transformers, and the baked-in reranker model account for most of it). The cross-encoder model is downloaded once at build time, so the running container needs no Hugging Face network access.

## Deployment

Azure Container Instances deployment is planned but not yet implemented.
