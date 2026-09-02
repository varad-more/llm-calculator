"""The same fixtures packages/core/test/golden.test.ts executes. If the two ports drift,
one of these two suites goes red."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from llmsize import size  # noqa: E402

GOLDEN = ROOT / "fixtures" / "golden"
FILES = sorted(f for f in GOLDEN.glob("*.json") if not f.name.startswith("_"))


def actual_for(request):
    r = size(request)
    p = r["plan"]
    return {
        "weightBytes": p["weights"]["totalBytes"],
        "weightMethod": p["weights"]["method"],
        "totalParams": p["weights"]["params"]["total"],
        "weightBytesPerDevice": p["weightBytesPerDevice"],
        "kvPerTokenBytesPerDevice": p["kv"]["perTokenBytesPerDevice"],
        "kvBytesPerDevice": p["kv"]["totalBytesPerDevice"],
        "activationBytes": p["overhead"]["activationBytes"],
        "overheadTotalBytes": p["overhead"]["totalBytes"],
        "usableVramBytes": p["usableVramBytes"],
        "availableKvBytes": p["availableKvBytes"],
        "numBlocks": p["numBlocks"],
        "maxTokens": p["maxTokens"],
        "fits": p["fits"],
        "autofix": p["autofix"],
        "warningCodes": sorted(w["code"] for w in p["warnings"]),
        "flags": r["flags"],
        "label": r["label"],
        "decodeStepSeconds": r["throughput"]["decode"]["stepSeconds"],
        "decodeTokensPerSecond": r["throughput"]["decode"]["tokensPerSecond"],
        "prefillFlops": r["throughput"]["prefill"]["flops"],
        "ttftSeconds": r["throughput"]["prefill"]["ttftSeconds"],
        "bound": r["throughput"]["bound"],
    }


def check(path):
    g = json.loads(path.read_text())
    actual = actual_for(g["request"])
    problems = []
    for key, want in g["expected"].items():
        got = actual[key]
        if isinstance(want, bool) or isinstance(got, bool):
            ok = got == want
        elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
            ok = abs(got - want) <= max(abs(want) * 1e-9, 1e-12)
        else:
            ok = got == want
        if not ok:
            problems.append("%s: got %r, expected %r" % (key, got, want))
    return problems


def test_golden_fixtures():
    assert len(FILES) >= 16, "only %d fixtures found" % len(FILES)
    failures = []
    for path in FILES:
        for p in check(path):
            failures.append("%s -> %s" % (path.name, p))
    assert not failures, "TS/Python parity broken:\n  " + "\n  ".join(failures)


def test_generated_data_matches_repo():
    """_generated.py must be regenerated (pnpm gen) whenever data/ changes."""
    from llmsize._generated import DATA
    repo = json.loads((ROOT / "data" / "gpus.json").read_text())["gpus"]
    assert [g["id"] for g in DATA["gpus"]] == [g["id"] for g in repo]


if __name__ == "__main__":
    test_golden_fixtures()
    test_generated_data_matches_repo()
    print("golden parity ok across %d fixtures" % len(FILES))
