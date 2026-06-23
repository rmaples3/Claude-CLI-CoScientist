"""End-to-end (offline): the loop computes a real verification verdict, and the tournament
rewards it. This is the contribution's core mechanism -- verification-weighted ranking --
exercised without any LLM call.
"""
import pytest

pytest.importorskip("numpy")

from app import tournament
from app.models import Hypothesis
from app.sdk.loop import CoScientistWorkflow

LIEB = {
    "kind": "tight_binding", "dim_k": 2, "onsite": [0, 0, 0],
    "hoppings": [
        {"amp": -1.0, "i": 0, "j": 1, "R": [0, 0]},
        {"amp": -1.0, "i": 0, "j": 1, "R": [-1, 0]},
        {"amp": -1.0, "i": 0, "j": 2, "R": [0, 0]},
        {"amp": -1.0, "i": 0, "j": 2, "R": [0, -1]},
    ],
    "claims": {"flat_band": {"band_index": 1, "max_bandwidth": 0.01}},
}


def test_loop_verify_marks_supported():
    """A hypothesis carrying a real (correct) tight-binding spec is verified by the loop."""
    wf = CoScientistWorkflow()
    h = Hypothesis("G1", "Lieb flat band", "A Lieb lattice hosts an exact flat band.")
    h.spec = LIEB
    wf._verify(h)
    assert h.verification["checkable"] is True
    assert h.verification["claim_supported"] is True
    assert h.verification["evidence"]["flat_band"]["bandwidth"] < 1e-6


def test_verified_beats_unverified_on_merit():
    """All else equal, a verification-supported hypothesis outranks an unverified one."""
    verified = Hypothesis("A", "verified", "x")
    verified.novelty_review = verified.feasibility_review = "MEDIUM"
    verified.verification = {"kind": "tight_binding", "checkable": True, "claim_supported": True,
                             "evidence": {}, "reason": "", "error": None}

    plain = Hypothesis("B", "unverified", "y")
    plain.novelty_review = plain.feasibility_review = "MEDIUM"
    plain.verification = {"kind": "none", "checkable": False, "claim_supported": None,
                          "evidence": {}, "reason": "", "error": None}

    assert tournament.merit(verified) > tournament.merit(plain)
    assert tournament.compare(verified, plain) is verified


def test_refuted_loses_on_merit():
    """A verification-refuted hypothesis is pushed below an equally-reviewed unverified one."""
    refuted = Hypothesis("A", "refuted", "x")
    refuted.novelty_review = refuted.feasibility_review = "HIGH"
    refuted.verification = {"kind": "tight_binding", "checkable": True, "claim_supported": False,
                            "evidence": {}, "reason": "", "error": None}

    plain = Hypothesis("B", "unverified", "y")
    plain.novelty_review = plain.feasibility_review = "MEDIUM"
    plain.verification = {"kind": "none", "checkable": False, "claim_supported": None,
                          "evidence": {}, "reason": "", "error": None}

    # refuted: HIGH+HIGH=6 minus penalty 3 = 3; plain: MEDIUM+MEDIUM=4 -> plain wins
    assert tournament.merit(refuted) < tournament.merit(plain)
    assert tournament.compare(refuted, plain) is plain
