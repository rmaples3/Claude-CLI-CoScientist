"""Plugin registry: built-in discovery, self-description, register/unregister (offline)."""
from app.verification import Verifier, available_kinds, describe_verifiers, register, unregister


def test_builtin_tight_binding_registered():
    assert "tight_binding" in available_kinds()


def test_describe_includes_kind_and_example():
    d = describe_verifiers()
    assert "tight_binding" in d
    assert "example spec" in d


def test_register_unregister_roundtrip():
    # Mimics how an external plugin (e.g. coscientist-biophysics) would add a kind.
    register(Verifier(
        kind="bio_demo",
        description="example biophysics validator",
        run=lambda s: {"kind": "bio_demo", "checkable": False, "claim_supported": None,
                       "evidence": {}, "reason": "demo", "error": None},
        spec_example={"kind": "bio_demo"},
    ))
    try:
        assert "bio_demo" in available_kinds()
        assert "bio_demo" in describe_verifiers()
    finally:
        unregister("bio_demo")
    assert "bio_demo" not in available_kinds()
