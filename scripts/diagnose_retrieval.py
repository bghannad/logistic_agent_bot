"""Diagnose retrieval stage by stage: where do the expected chunks rank in BM25,
in the embedding search, after RRF, and after the cross-encoder reranker?

Usage (from repo root, with .venv active):
    python scripts/diagnose_retrieval.py            # one case in detail (default ret_13)
    python scripts/diagnose_retrieval.py ret_05     # another case in detail
    python scripts/diagnose_retrieval.py all        # every case + stage-wise metrics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retrieval.rerank import _get_model  # noqa: E402
from retrieval.retrieve import CANDIDATE_K, FINAL_K  # noqa: E402
from retrieval.search import HybridIndex, _tokenize, reciprocal_rank_fusion  # noqa: E402

CASES_PATH = REPO_ROOT / "data" / "eval" / "retrieval_cases.jsonl"


def load_cases() -> list[dict]:
    with CASES_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_stages(index: HybridIndex, query: str) -> dict:
    """Same steps as retrieve(), but keeps every intermediate ranking."""
    bm25 = index._bm25_ranking(query)                       # all 63, best first
    bm25_scores = dict(zip(index.chunk_ids, index.bm25.get_scores(_tokenize(query))))
    vector = index._vector_ranking(query)                   # all 63, best first (1 API call)
    fused = reciprocal_rank_fusion([bm25, vector])          # all 63, best first
    candidates = fused[:CANDIDATE_K]                        # what the reranker sees
    scores = _get_model().predict([(query, index.by_id[c]["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return {
        "bm25": bm25,
        "bm25_scores": bm25_scores,
        "vector": vector,
        "fused": fused,
        "reranked": [cid for cid, _ in ranked],             # the 15 candidates, reranked
        "ce_scores": {cid: float(s) for cid, s in ranked},
    }


def rank_of(ranking: list[str], chunk_id: str) -> int | None:
    return ranking.index(chunk_id) + 1 if chunk_id in ranking else None


def fmt(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def diagnose_one(index: HybridIndex, case: dict) -> None:
    query = case["query"]
    expected = case["expected_chunk_ids"]
    s = run_stages(index, query)

    print(f"[{case['id']}] {query}")
    print(f"expected: {expected}\n")

    print("Query words: BM25 IDF, and whether the word appears in each expected chunk")
    chunk_words = {cid: set(_tokenize(index.by_id[cid]["text"])) for cid in expected}
    print(f"  {'word':<12}{'IDF':>8}   " + "   ".join(cid.split('__')[-1] for cid in expected))
    for word in dict.fromkeys(_tokenize(query)):
        idf = index.bm25.idf.get(word)
        idf_text = "not in KB" if idf is None else f"{idf:.2f}"
        marks = "   ".join(("yes" if word in chunk_words[cid] else " . ").center(5) for cid in expected)
        print(f"  {word:<12}{idf_text:>9}  {marks}")

    print(f"\nRank of each expected chunk at each stage (1 = best; out of {len(index.chunk_ids)})")
    print(f"  {'chunk':<8}{'BM25':>6}{'BM25 score':>12}{'Embed':>7}{'RRF':>6}"
          f"{'in top ' + str(CANDIDATE_K) + '?':>13}{'Rerank':>8}{'CE score':>10}{'final top ' + str(FINAL_K) + '?':>15}")
    for cid in expected:
        rr = rank_of(s["reranked"], cid)
        ce = s["ce_scores"].get(cid)
        print(f"  {cid.split('__')[-1]:<8}{fmt(rank_of(s['bm25'], cid)):>6}{s['bm25_scores'][cid]:>12.2f}"
              f"{fmt(rank_of(s['vector'], cid)):>7}{fmt(rank_of(s['fused'], cid)):>6}"
              f"{('yes' if rr else 'NO'):>13}{fmt(rr):>8}{('-' if ce is None else f'{ce:.2f}'):>10}"
              f"{('yes' if rr and rr <= FINAL_K else 'NO'):>15}")

    def show(title: str, ranking: list[str], n: int = 5) -> None:
        print(f"\n{title}")
        for i, cid in enumerate(ranking[:n], 1):
            mark = "  <- expected" if cid in expected else ""
            print(f"  {i}. {cid}  ({index.by_id[cid].get('section_heading')}){mark}")

    show("BM25 top 5", s["bm25"])
    show("Embedding top 5", s["vector"])
    show("Final top 5 (= what retrieve() returns), with cross-encoder score", s["reranked"], FINAL_K)


def diagnose_all(index: HybridIndex, cases: list[dict]) -> None:
    stage_names = ["BM25 only", "Embeddings only", "RRF, no rerank", "Full (RRF + rerank)"]
    totals = {name: {"hit": 0, "recall": 0.0, "rr": 0.0} for name in stage_names}
    recall_candidates = 0.0

    print(f"{'case':<8}{'BM25':>6}{'Embed':>7}{'RRF':>6}{'Final':>7}   (rank of the FIRST expected chunk; '-' = not in list)")
    for case in cases:
        expected = set(case["expected_chunk_ids"])
        s = run_stages(index, case["query"])
        top_lists = {
            "BM25 only": s["bm25"][:FINAL_K],
            "Embeddings only": s["vector"][:FINAL_K],
            "RRF, no rerank": s["fused"][:FINAL_K],
            "Full (RRF + rerank)": s["reranked"][:FINAL_K],
        }
        for name, top in top_lists.items():
            found = [i for i, cid in enumerate(top, 1) if cid in expected]
            totals[name]["hit"] += bool(found)
            totals[name]["recall"] += len(found) / len(expected)
            totals[name]["rr"] += 1 / found[0] if found else 0.0
        recall_candidates += len(set(s["fused"][:CANDIDATE_K]) & expected) / len(expected)

        def first(ranking: list[str]) -> int | None:
            return next((i for i, cid in enumerate(ranking, 1) if cid in expected), None)

        print(f"{case['id']:<8}{fmt(first(s['bm25'])):>6}{fmt(first(s['vector'])):>7}"
              f"{fmt(first(s['fused'])):>6}{fmt(first(s['reranked'])):>7}")

    n = len(cases)
    print(f"\nStage-wise metrics at k={FINAL_K} ({n} cases, one run)")
    print(f"  {'stage':<22}{'Hit@5':>9}{'Recall@5':>10}{'MRR':>7}")
    for name in stage_names:
        t = totals[name]
        print(f"  {name:<22}{t['hit']:>5}/{n:<3}{t['recall'] / n:>10.2f}{t['rr'] / n:>7.2f}")
    print(f"\nRecall@{CANDIDATE_K} of the RRF candidates (the reranker's ceiling): {recall_candidates / n:.2f}")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "ret_13"
    cases = load_cases()
    index = HybridIndex()
    if arg == "all":
        diagnose_all(index, cases)
        return
    case = next((c for c in cases if c["id"] == arg), None)
    if case is None:
        sys.exit(f"No case with id {arg!r}")
    diagnose_one(index, case)


if __name__ == "__main__":
    main()
