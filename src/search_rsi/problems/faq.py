"""Customer-support FAQ retrieval: paraphrased questions against canonical answers.

What makes it hard, and which pipeline choices it rewards:
- Users paraphrase ("terminate my membership") what the help center words
  canonically ("cancel my subscription") — the classic semantic-matching gap,
  closed here by learned rewrites rather than an embedding model.
- Stopword-heavy phrasing ("is it possible to ...") adds noise that a
  stopword-aware analyzer drops.
- Near-duplicate articles (the business-account variant of the same task) need
  full query-term coverage to rank correctly — the reranker's job.
"""
from __future__ import annotations

import random

from search_rsi.problems.base import Query, SearchProblem, round_robin_splits
from search_rsi.types import Document

ACTIONS = {
    "cancel": ["terminate", "end"],
    "update": ["change", "modify"],
    "reset": ["recover", "restore"],
    "download": ["export", "retrieve"],
    "pause": ["suspend", "freeze"],
    "delete": ["erase", "remove"],
    "verify": ["confirm", "authenticate"],
}
OBJECTS = {
    "subscription": ["membership", "plan"],
    "password": ["passcode", "login"],
    "invoice": ["bill", "receipt"],
    "payment method": ["card", "billing details"],
    "email address": ["mail", "contact email"],
    "shipping address": ["delivery location", "postal destination"],
    "order": ["purchase", "checkout"],
}
VALID = {
    "cancel": ["subscription", "order"],
    "update": ["payment method", "email address", "shipping address", "password"],
    "reset": ["password"],
    "download": ["invoice"],
    "pause": ["subscription"],
    "delete": ["payment method", "order", "shipping address"],
    "verify": ["email address", "payment method"],
}
SECTIONS = {
    "subscription": "Billing > Plan", "password": "Security", "invoice": "Billing > History",
    "payment method": "Billing > Wallet", "email address": "Profile", "shipping address": "Profile > Addresses",
    "order": "Orders",
}
TEMPLATES = [
    "i want to {a} my {o}",
    "how can i {a} {o}",
    "is it possible to {a} the {o} on my profile",
    "{a} {o} help",
    "steps to {a} my {o} please",
]


def build(seed: int = 13) -> SearchProblem:
    rng = random.Random(seed)
    docs: list[Document] = []
    intents = [(a, o) for a, objs in VALID.items() for o in objs]
    for a, o in intents:
        for variant in ("personal", "business"):
            suffix = " for a business account" if variant == "business" else ""
            extra = (" Business accounts require an administrator role and a second approval."
                     if variant == "business" else " Changes apply immediately to your personal account.")
            docs.append(Document(
                doc_id=f"faq_{a}_{o.replace(' ', '_')}_{variant}",
                title=f"How do I {a} my {o}{suffix}?",
                text=f"To {a} your {o}{suffix}, open Settings, go to {SECTIONS[o]}, and select {a.title()}.{extra}",
                metadata={"action": a, "object": o, "variant": variant},
            ))

    queries: list[Query] = []
    for a, o in intents:
        for j in range(5):
            # j == 0 uses the help center's own words; the rest paraphrase with
            # user vocabulary drawn from a shared pool, so the same user word recurs
            # across intents and splits (and a rule learned on train can transfer).
            a_word = a if j == 0 else rng.choice([a] + ACTIONS[a] * 2)
            o_word = o if j == 0 else rng.choice([o] + OBJECTS[o] * 2)
            business = j == 4
            text = TEMPLATES[len(queries) % len(TEMPLATES)].format(a=a_word, o=o_word)
            if business:
                text += " for our business account"
            want, other = ("business", "personal") if business else ("personal", "business")
            key = f"faq_{a}_{o.replace(' ', '_')}"
            queries.append(Query(
                qid=f"fq{len(queries):03d}", text=text,
                qrels={f"{key}_{want}": 2, f"{key}_{other}": 1},
                intent=f"{a_word}|{o_word}",
            ))

    problem = SearchProblem(
        name="faq_support",
        use_case="customer-support FAQ matching under paraphrase",
        description=__doc__.strip().splitlines()[0],
        documents=docs,
        queries=queries,
        splits=round_robin_splits(queries, group=lambda q: q.intent.split("|")[1]),
        primary_metric="ndcg@10",
    )
    problem.validate()
    return problem
