"""Semantic quality check: does the embedder capture meaning, not just words?

These cases are deliberately worded as a *real user* would, and the "related" texts as
an author would actually write them. An earlier version used terse fragments
("token validation order") which all-MiniLM genuinely scores as equally dissimilar from
the query and from the distractor -- that is a known limitation of sentence encoders on
very short inputs, not a bug to fix here.

The bar is `related > unrelated` plus a margin, not an absolute cosine: absolute
similarity depends on phrasing and carries no information about ranking correctness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragpipe.embedding import HashingEmbedder  # noqa: E402
from ragpipe.embedding_local import LocalOnnxEmbedder  # noqa: E402

# (label, query, related, unrelated)
CASES = [
    (
        "rollback",
        "how do we roll back a failed deployment?",
        "If a release goes wrong, redeploy the previous image tag; rollback should take under five minutes.",
        "Apples grow well in cool temperate climates with acidic soil.",
    ),
    (
        "crash",
        "what caused the export worker to crash?",
        "The worker crashed because it dereferenced customer.currency without a null check, and two percent of customers had no currency set.",
        "The rolling update strategy lets us deploy new pods without downtime.",
    ),
    (
        "tokens",
        "what order should I validate tokens in?",
        "Validate in this order: signature, then issuer, then audience, then expiry. Checking expiry first rejects valid tokens under clock skew.",
        "We deploy with a rolling update and never use a recreate strategy in production.",
    ),
    (
        "synonym",
        "how long do credentials stay valid?",
        "API keys rotate every ninety days; access tokens are valid for fifteen minutes.",
        "Migrations must run as a separate job before the application rollout.",
    ),
]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def run(name, embedder, margin=0.05):
    print(f"--- {name} (dims={embedder.dimensions}) ---")
    passed = 0
    for label, query, related, unrelated in CASES:
        qv, rv, uv = embedder.embed_many([query, related, unrelated])
        cr, cu = cos(qv, rv), cos(qv, uv)
        # Ranking correctness is the only thing that matters for retrieval.
        ok = cr > cu and (cr - cu) >= margin
        passed += ok
        print(
            f"  {label:10} related={cr:+.3f}  unrelated={cu:+.3f}  "
            f"margin={cr - cu:+.3f}  {'PASS' if ok else 'FAIL'}"
        )
    print(f"  {passed}/{len(CASES)}\n")
    return passed


if __name__ == "__main__":
    title = sys.argv[1] if len(sys.argv) > 1 else "embedder"
    print(f"\n=== {title} ===\n")
    try:
        run("local ONNX (all-MiniLM-L6-v2)", LocalOnnxEmbedder())
    except RuntimeError as exc:
        print(f"local unavailable: {exc}\n")

    # Contrast only: hashing matches words, not meaning, so it is expected to score
    # lower here. That gap is the reason the local provider exists.
    run("hashing (lexical, not semantic)", HashingEmbedder(256))
