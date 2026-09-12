"""Deterministically generates a larger, harder benchmark corpus + gold task set.

Run with: python benchmarks/corpus/generate_corpus.py

Why a generator instead of more hand-written docs (docs/META_RSI.md "Current
status"): the old 10-doc corpus let BM25 find the right document regardless of
k1/b, and the verifier's min_overlap never bound, so the meta-loop had no headroom
to show a win. This corpus deliberately builds in:

- Real lexical ambiguity: each topic chain gets a decoy document (`_d`) that shares
  surface vocabulary (an adjective, a neighboring chain's term) with the correct
  answer but is not it — so `verifier_min_overlap` actually trades precision for
  recall instead of being a no-op.
- Varied document length: the context document (`_c`) is padded with filler text,
  so BM25's length-normalization parameter `b` actually has something to normalize
  against.
- Enough chains/tasks (12 chains -> 48 docs, 24 tasks) that a train/meta_val/
  held_out split of 8/8/8 is large enough for accept/reject decisions in the
  meta-loop to mean something.
"""
from __future__ import annotations

import json
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent

ADJ1 = [
    "Ravensbrook", "Azoren", "Kestrel", "Doventry", "Marrow", "Cindral",
    "Halworth", "Thessaly", "Grendmoor", "Ashvale", "Corrin", "Ludmere",
]
NOUN1 = [
    "Protocol", "Bank", "Flats", "Accord", "Institute", "Guild",
    "Consortium", "Registry", "Bureau", "Trust", "Syndicate", "Chapter",
]
ADJ2 = [
    "Meridian", "Verdant", "Solace", "Kessler", "Northgate", "Ferrous",
    "Amberline", "Solenne", "Cobalt", "Windmere", "Talbrook", "Ironvale",
]
NOUN2 = [
    "Framework", "Alliance", "Board", "Compact", "Council", "Foundry",
    "Assembly", "Charter", "Bureau", "Coalition", "Guild", "Order",
]
TERM = [
    "latency budget", "thermal drift index", "soil moisture gradient",
    "water rights allocation", "signal noise floor", "canopy cover ratio",
    "credit risk buffer", "migration corridor width", "spectral leakage margin",
    "aquifer recharge rate", "population turnover rate", "load balancing threshold",
]
DESC = [
    "the maximum delay a relay hop could add",
    "the seasonal variance in surface temperature readings",
    "the seasonal change in topsoil water content",
    "the allocation of shared water usage among users",
    "the background interference level in a channel",
    "the fraction of ground shaded by tree canopy",
    "the reserve held against expected loan defaults",
    "the minimum width required for safe wildlife passage",
    "the unwanted signal escaping a filter band",
    "the speed at which groundwater is replenished",
    "the rate at which members leave and rejoin a population",
    "the point at which incoming requests are redistributed",
]
REDEF = [
    "a per-service allowance rather than a per-hop one",
    "a rolling average instead of a single reading",
    "a monthly rather than seasonal measurement",
    "a per-basin rather than per-river allocation",
    "a normalized ratio instead of an absolute value",
    "a canopy density score instead of a raw ratio",
    "a dynamic reserve tied to real-time exposure",
    "a minimum corridor length instead of a width",
    "a frequency-weighted margin instead of a flat one",
    "a basin-wide instead of well-level rate",
    "an annualized instead of monthly rate",
    "a queue-depth trigger instead of a fixed threshold",
]
DOMAINS = ["networking", "ecology", "policy", "genetics"]
FILLER = (
    "It published annual reports, held public consultations, and coordinated "
    "with neighboring bodies on a range of unrelated administrative matters "
    "over the following decade, expanding its staff and regional presence."
)

N = len(TERM)
assert len(ADJ1) == len(NOUN1) == len(ADJ2) == len(NOUN2) == len(DESC) == len(REDEF) == N


def build():
    documents = []
    gold_tasks = []
    splits = {"train": [], "meta_val": [], "held_out_eval": []}
    split_names = ["train", "meta_val", "held_out_eval"]

    for i in range(N):
        entity1 = f"{ADJ1[i]} {NOUN1[i]}"
        entity2 = f"{ADJ2[i]} {NOUN2[i]}"
        term = TERM[i]
        neighbor_term = TERM[(i + 1) % N]
        neighbor_noun1 = NOUN1[(i + 1) % N]
        year1 = 1960 + i * 3
        year2 = year1 + 7
        domain = DOMAINS[i % len(DOMAINS)]

        doc_a = f"doc_{i:02d}a"
        doc_b = f"doc_{i:02d}b"
        doc_c = f"doc_{i:02d}c"
        doc_d = f"doc_{i:02d}d"

        documents.append({
            "doc_id": doc_a,
            "text": f"The {entity1} was established in {year1}. It introduced the term "
                    f"{term} to describe {DESC[i]}.",
        })
        documents.append({
            "doc_id": doc_b,
            "text": f"{term.capitalize()} first appeared in work from the {entity1} and "
                    f"was later adopted by the {entity2}, which redefined it as {REDEF[i]}.",
        })
        documents.append({
            "doc_id": doc_c,
            "text": f"The {entity2} was founded in {year2} as a successor body. {FILLER} "
                    f"It also briefly discussed {neighbor_term} in an unrelated appendix.",
        })
        documents.append({
            "doc_id": doc_d,
            "text": f"The {ADJ1[i]} {neighbor_noun1} was a distinct initiative from "
                    f"{year1}, focused on {neighbor_term} and unaffiliated with the {entity1}.",
        })

        q1 = f"origin of the term {term} which organization introduced it"
        q2 = f"{term} redefinition by successor organization {entity2}"

        task_idx_base = len(gold_tasks)
        gold_tasks.append({
            "question": q1,
            "gold_doc_ids": [doc_a, doc_b],
            "gold_answer": doc_a,
            "tool_budget": 4,
            "task_type": "lookup",
            "domain": domain,
        })
        gold_tasks.append({
            "question": q2,
            "gold_doc_ids": [doc_a, doc_b],
            "gold_answer": doc_b,
            "tool_budget": 4,
            "task_type": "multihop",
            "domain": domain,
        })

        # Balanced 3-way split by chain index so every split gets every domain
        # and both lookup/multihop task types (docs/META_RSI.md Guardrail 2).
        split = split_names[i % 3]
        splits[split].extend([task_idx_base, task_idx_base + 1])

    return documents, gold_tasks, splits


def main() -> None:
    documents, gold_tasks, splits = build()
    (CORPUS_DIR / "docs.json").write_text(json.dumps(documents, indent=2) + "\n")
    (CORPUS_DIR / "gold_tasks.json").write_text(json.dumps(gold_tasks, indent=2) + "\n")
    (CORPUS_DIR / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    print(f"wrote {len(documents)} documents, {len(gold_tasks)} gold tasks")
    for name, idxs in splits.items():
        print(f"  {name}: {len(idxs)} tasks")


if __name__ == "__main__":
    main()
