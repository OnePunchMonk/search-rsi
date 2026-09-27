"""Code search: natural-language questions against function source.

What makes it hard, and which pipeline choices it rewards:
- Identifiers are camelCase / snake_case (`parseCfgFile`), so a plain word
  tokenizer sees one opaque token — the `code` analyzer splits them.
- Code abbreviates (`cfg`, `auth`, `img`, `db`) where people write it out
  ("configuration", "authentication") — learned rewrites bridge that.
- People say "remove" / "download" / "check" where code says delete / fetch /
  validate: a second vocabulary gap on the verb side.
"""
from __future__ import annotations

import random

from search_rsi.problems.base import Query, SearchProblem, round_robin_splits
from search_rsi.types import Document

# identifier form, natural-language form
OBJECTS = [
    ("cfg_file", "configuration file"),
    ("auth_token", "authentication token"),
    ("img_thumb", "image thumbnail"),
    ("db_conn", "database connection"),
    ("msg_queue", "message queue"),
    ("env_vars", "environment variables"),
    ("user_pwd", "user password"),
    ("http_req", "http request"),
    ("csv_report", "csv report"),
    ("ts_index", "timestamp index"),
]
# identifier verb, natural-language synonyms
VERBS = [
    ("parse", ["parse", "read and interpret"]),
    ("load", ["load", "read in"]),
    ("save", ["save", "persist"]),
    ("validate", ["validate", "check"]),
    ("delete", ["delete", "remove"]),
    ("fetch", ["fetch", "download"]),
    ("encode", ["encode", "serialize"]),
    ("merge", ["merge", "combine"]),
]
BODIES = {
    "parse": "tokens = lexer.split(raw)\n    return build_tree(tokens)",
    "load": "with open(path) as fh:\n        return reader.read(fh)",
    "save": "with open(path, 'w') as fh:\n        fh.write(dump(obj))",
    "validate": "if not schema.check(obj):\n        raise ValueError('invalid')\n    return True",
    "delete": "store.remove(key)\n    log.info('removed %s', key)",
    "fetch": "resp = client.get(url, timeout=10)\n    return resp.body",
    "encode": "buf = bytearray()\n    return codec.pack(obj, buf)",
    "merge": "out = dict(left)\n    out.update(right)\n    return out",
}


def _camel(verb: str, ident: str) -> str:
    return verb + "".join(p.title() for p in ident.split("_"))


def build(seed: int = 11) -> SearchProblem:
    rng = random.Random(seed)
    docs: list[Document] = []
    combos = [(v, o) for v, _ in VERBS for o in OBJECTS]
    rng.shuffle(combos)
    combos = combos[:72]
    for verb, (ident, _) in combos:
        body = BODIES[verb].replace("obj", ident.split("_")[0]).replace("raw", f"raw_{ident.split('_')[-1]}")
        if rng.random() < 0.5:
            name = f"{verb}_{ident}"
            text = f"def {name}(path, strict=False):\n    {body}\n"
        else:
            name = _camel(verb, ident)
            text = f"function {name}(path, strict) {{\n    {body}\n}}\n"
        docs.append(Document(doc_id=f"fn{len(docs):03d}", title=name, text=text,
                             metadata={"verb": verb, "object": ident}))

    queries: list[Query] = []
    verb_syns = dict(VERBS)
    templates = ["how to {v} a {o}", "{v} {o}", "function that can {v} the {o}"]
    for k, doc in enumerate(docs):
        verb, ident = doc.metadata["verb"], doc.metadata["object"]
        nl_obj = dict(OBJECTS)[ident]
        nl_verb = verb_syns[verb][k % 2]
        text = templates[k % 3].format(v=nl_verb, o=nl_obj)
        queries.append(Query(qid=f"cq{len(queries):03d}", text=text, qrels={doc.doc_id: 2},
                             intent=ident))

    problem = SearchProblem(
        name="code_search",
        use_case="natural-language to code search over mixed-style identifiers",
        description=__doc__.strip().splitlines()[0],
        documents=docs,
        queries=queries,
        splits=round_robin_splits(queries, group=lambda q: q.intent),
        primary_metric="mrr@10",
    )
    problem.validate()
    return problem
