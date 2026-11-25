"""Does retrieve() return the same 5 chunks every time for the same message?
Runs retrieve() several times on each groundedness message and compares the IDs.

Usage (from repo root, with .venv active):
    python scripts/check_retrieval_stability.py        # 3 runs per message
    python scripts/check_retrieval_stability.py 5      # 5 runs per message
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retrieval.retrieve import retrieve  # noqa: E402
from retrieval.search import HybridIndex  # noqa: E402

CASES_PATH = REPO_ROOT / "data" / "eval" / "groundedness_cases.jsonl"


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    index = HybridIndex()
    with CASES_PATH.open(encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]

    stable = 0
    for case in cases:
        results = [[c["chunk_id"] for c in retrieve(index, case["message"])] for _ in range(runs)]
        same = all(r == results[0] for r in results)
        stable += same
        print(f"[{case['id']}] {'SAME' if same else 'DIFFERENT'}")
        if not same:
            for i, r in enumerate(results, 1):
                print(f"  run {i}: {r}")

    print(f"\nIdentical top 5 (same chunks, same order) in all {runs} runs: {stable}/{len(cases)} messages")


if __name__ == "__main__":
    main()
