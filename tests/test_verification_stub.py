"""Verification dispatch + stub/error behavior (no physics deps, no LLM, no network).

Run from the repo root:  python -m pytest tests/test_verification_stub.py
"""
from app.verification import Verifier, register, run_verification, unregister, verdict


def test_unknown_kind_uses_stub():
    res = run_verification({"kind": "___definitely_not_registered___"})
    assert res["checkable"] is False
    assert res["claim_supported"] is None
    assert res["error"] is None
    assert res["kind"] == "___definitely_not_registered___"


def test_non_dict_spec_is_error():
    res = run_verification(None)
    assert res["checkable"] is False
    assert res["claim_supported"] is None
    assert res["error"]  # populated with a message


def test_register_custom_and_dispatch():
    register(Verifier(kind="toy", description="toy",
                      run=lambda s: verdict("toy", True, {"v": s.get("v")}, "ok")))
    try:
        res = run_verification({"kind": "toy", "v": 7})
        assert res["checkable"] is True
        assert res["claim_supported"] is True
        assert res["evidence"]["v"] == 7
    finally:
        unregister("toy")


def test_verifier_exception_is_captured():
    def boom(spec):
        raise RuntimeError("kaboom")

    register(Verifier(kind="boom", description="b", run=boom))
    try:
        res = run_verification({"kind": "boom"})
        assert res["checkable"] is False
        assert "kaboom" in (res["error"] or "")
    finally:
        unregister("boom")
