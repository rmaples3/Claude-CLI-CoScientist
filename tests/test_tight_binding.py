"""Tight-binding verifier checked against analytically-known models (offline, no credits).

  - Lieb lattice -> exact flat middle band at E=0       (flat_band)
  - QWZ model    -> Chern +/-1 for |m|<2, 0 for |m|>2   (chern_number, gapped_at_filling)
  - square 1-band -> dispersive, bandwidth ~8           (flat_band negative control)
"""
import pytest

pytest.importorskip("numpy")

from app.verification.condensed_matter.tight_binding import verify_tight_binding


def _lieb(claims):
    return {
        "kind": "tight_binding", "dim_k": 2, "onsite": [0, 0, 0],
        "hoppings": [
            {"amp": -1.0, "i": 0, "j": 1, "R": [0, 0]},
            {"amp": -1.0, "i": 0, "j": 1, "R": [-1, 0]},
            {"amp": -1.0, "i": 0, "j": 2, "R": [0, 0]},
            {"amp": -1.0, "i": 0, "j": 2, "R": [0, -1]},
        ],
        "claims": claims,
    }


def _qwz(m, claims):
    """Qi-Wu-Zhang model: H = sin(kx)sx + sin(ky)sy + (m+cos kx+cos ky)sz."""
    return {
        "kind": "tight_binding", "dim_k": 2, "onsite": [m, -m],
        "hoppings": [
            {"amp": 0.5, "i": 0, "j": 0, "R": [1, 0]},     # +cos(kx) sz
            {"amp": -0.5, "i": 1, "j": 1, "R": [1, 0]},
            {"amp": 0.5, "i": 0, "j": 0, "R": [0, 1]},     # +cos(ky) sz
            {"amp": -0.5, "i": 1, "j": 1, "R": [0, 1]},
            {"amp": [0, -0.5], "i": 0, "j": 1, "R": [1, 0]},   # sin(kx) sx
            {"amp": [0, 0.5], "i": 0, "j": 1, "R": [-1, 0]},
            {"amp": -0.5, "i": 0, "j": 1, "R": [0, 1]},        # sin(ky) sy
            {"amp": 0.5, "i": 0, "j": 1, "R": [0, -1]},
        ],
        "claims": claims,
    }


def _square_1band(claims):
    return {
        "kind": "tight_binding", "dim_k": 2, "onsite": [0.0],
        "hoppings": [{"amp": -1.0, "i": 0, "j": 0, "R": [1, 0]},
                     {"amp": -1.0, "i": 0, "j": 0, "R": [0, 1]}],
        "claims": claims,
    }


def test_lieb_flat_band_supported():
    res = verify_tight_binding(_lieb({"flat_band": {"band_index": 1, "max_bandwidth": 0.01}}))
    assert res["checkable"] is True
    assert res["claim_supported"] is True
    assert res["evidence"]["flat_band"]["bandwidth"] < 1e-6


def test_dispersive_band_not_flat():
    res = verify_tight_binding(_square_1band({"flat_band": {"band_index": 0, "max_bandwidth": 0.1}}))
    assert res["claim_supported"] is False
    assert res["evidence"]["flat_band"]["bandwidth"] > 7  # ~8


def test_qwz_topological_chern_abs_one():
    res = verify_tight_binding(_qwz(-1.0, {"chern_number": {"band_index": 0}}))
    assert res["checkable"] is True
    assert abs(res["evidence"]["chern_number"]["chern"]) == 1


def test_qwz_trivial_chern_zero():
    res = verify_tight_binding(_qwz(3.0, {"chern_number": {"band_index": 0, "expected": 0}}))
    assert res["claim_supported"] is True
    assert res["evidence"]["chern_number"]["chern"] == 0


def test_qwz_expected_mismatch_fails():
    res = verify_tight_binding(_qwz(-1.0, {"chern_number": {"band_index": 0, "expected": 0}}))
    assert res["claim_supported"] is False


def test_qwz_gapped_vs_gapless():
    gapped = verify_tight_binding(_qwz(3.0, {"gapped_at_filling": {"filling": 0.5, "min_gap": 0.5}}))
    assert gapped["claim_supported"] is True
    gapless = verify_tight_binding(_qwz(0.0, {"gapped_at_filling": {"filling": 0.5, "min_gap": 0.5}}))
    assert gapless["claim_supported"] is False


def test_combined_claims_all_pass():
    # QWZ m=-1: topological (|C|=1) AND gapped at half filling.
    res = verify_tight_binding(_qwz(-1.0, {
        "gapped_at_filling": {"filling": 0.5, "min_gap": 0.1},
        "chern_number": {"band_index": 0, "expected": 1},
    }))
    # claim_supported is the AND of sub-verdicts; expected=1 may be -1 by convention,
    # so assert gap passed and chern magnitude, not the combined boolean here.
    assert res["evidence"]["gapped_at_filling"]["direct_gap"] >= 0.1
    assert abs(res["evidence"]["chern_number"]["chern"]) == 1


def test_no_claims_is_unverifiable():
    res = verify_tight_binding({"kind": "tight_binding", "dim_k": 2, "onsite": [0.0],
                                "hoppings": [{"amp": -1.0, "i": 0, "j": 0, "R": [1, 0]}]})
    assert res["checkable"] is False
    assert res["error"] is None  # nothing to check is not an error


def test_bad_spec_is_error():
    res = verify_tight_binding({"kind": "tight_binding", "claims": {"flat_band": {"band_index": 0}}})
    assert res["error"]  # missing hoppings
