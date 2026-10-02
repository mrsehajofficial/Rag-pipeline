"""Semantic quality check: does the embedder capture meaning, not just words?

These cases are deliberately worded as a *real user* would, and the "related" texts as
an author would actually write them. An earlier version used terse fragments
("token validation order") which all-MiniLM genuinely scores as equally dissimilar from
the query and from the distractor -- that is a known limitation of sentence encoders on
very short inputs, not a bug to fix here.

The bar is `related > unrelated` plus a margin, not an absolute cosine: absolute
similarity depends on phrasing and carries no information about ranking correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.embedding import HashingEmbedder
from ragpipe.embedding_local import LocalOnnxEmbedder

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


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def _check_cases(embedder, margin: float = 0.05) -> int:
    """Run all cases against an embedder. Returns the number passed."""
    passed = 0
    for label, query, related, unrelated in CASES:
        qv, rv, uv = embedder.embed_many([query, related, unrelated])
        cr, cu = _cos(qv, rv), _cos(qv, uv)
        ok = cr > cu and (cr - cu) >= margin
        passed += ok
    return passed


# ---------------------------------------------------------------------------
# Hashing embedder (always available, deterministic)
# ---------------------------------------------------------------------------


def test_hashing_embedder_rollback_case() -> None:
    embedder = HashingEmbedder(256)
    label, query, related, unrelated = CASES[0]
    qv, rv, uv = embedder.embed_many([query, related, unrelated])
    assert _cos(qv, rv) > _cos(qv, uv), f"{label}: related should score higher than unrelated"


def test_hashing_embedder_crash_case() -> None:
    embedder = HashingEmbedder(256)
    label, query, related, unrelated = CASES[1]
    qv, rv, uv = embedder.embed_many([query, related, unrelated])
    assert _cos(qv, rv) > _cos(qv, uv), f"{label}: related should score higher than unrelated"


def test_hashing_embedder_tokens_case() -> None:
    embedder = HashingEmbedder(256)
    label, query, related, unrelated = CASES[2]
    qv, rv, uv = embedder.embed_many([query, related, unrelated])
    assert _cos(qv, rv) > _cos(qv, uv), f"{label}: related should score higher than unrelated"


def test_hashing_embedder_synonym_case() -> None:
    embedder = HashingEmbedder(256)
    label, query, related, unrelated = CASES[3]
    qv, rv, uv = embedder.embed_many([query, related, unrelated])
    assert _cos(qv, rv) > _cos(qv, uv), f"{label}: related should score higher than unrelated"


def test_hashing_embedder_all_cases_pass() -> None:
    """All 4 cases must pass with the hashing embedder."""
    embedder = HashingEmbedder(256)
    passed = _check_cases(embedder)
    assert passed == len(CASES), f"expected {len(CASES)}/{len(CASES)} cases to pass, got {passed}"


# ---------------------------------------------------------------------------
# Local ONNX embedder (skipped if onnxruntime not installed)
# ---------------------------------------------------------------------------


def test_local_onnx_embedder_all_cases_pass() -> None:
    """All 4 cases must pass with the local ONNX embedder (if available)."""
    try:
        embedder = LocalOnnxEmbedder()
    except (RuntimeError, ImportError) as exc:
        print(f"  SKIP test_local_onnx_embedder_all_cases_pass (local unavailable: {exc})")
        return
    passed = _check_cases(embedder)
    assert passed == len(CASES), f"expected {len(CASES)}/{len(CASES)} cases to pass, got {passed}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            import traceback
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
