"""Contract for the pluggable verification layer.

A *verifier* takes a structured, machine-checkable hypothesis ``spec`` (a plain dict
carrying a ``"kind"`` discriminator plus kind-specific fields) and returns a
``VerificationResult`` dict. Both the Claude Agent SDK frontend (``app/sdk``) and the
OpenAI/vLLM loop (``app/agents.py``) consume the same shape, so verification-weighted
ranking is apples-to-apples across backends -- and across verifier domains.

Result schema (all keys always present):

    kind:            str             # the spec["kind"] that was dispatched on
    checkable:       bool            # True if a verifier exists for this kind and ran
    claim_supported: Optional[bool]  # True/False verdict when checkable; None otherwise
    evidence:        dict            # numeric/structured evidence (e.g. bandwidths, gaps, chern)
    reason:          str             # human-readable explanation of the verdict
    error:           Optional[str]   # populated only if the verifier raised

A ``Verifier`` bundles the run callable with self-description (so the Generation agent can
be told which kinds are machine-checkable and how to format a spec). Verifiers register by
``kind`` -- see ``app/verification/registry.py`` and docs/verifiers.md.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass
class Verifier:
    """A pluggable physics validator for one hypothesis ``kind``."""
    kind: str
    description: str                                       # what claims it checks + spec format
    run: Callable[[Dict[str, Any]], Dict[str, Any]]        # spec -> VerificationResult dict
    spec_schema: Dict[str, Any] = field(default_factory=dict)   # optional JSON-schema-ish hint
    spec_example: Dict[str, Any] = field(default_factory=dict)  # a concrete example spec


def unverifiable(kind: str, reason: str) -> Dict[str, Any]:
    """No verifier could evaluate this hypothesis (e.g. not machine-checkable, or no claims)."""
    return {
        "kind": kind,
        "checkable": False,
        "claim_supported": None,
        "evidence": {},
        "reason": reason,
        "error": None,
    }


def verdict(kind: str, claim_supported: bool, evidence: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """A verifier ran and reached a verdict on the hypothesis's central claim."""
    return {
        "kind": kind,
        "checkable": True,
        "claim_supported": bool(claim_supported),
        "evidence": evidence,
        "reason": reason,
        "error": None,
    }


def error(kind: str, message: str) -> Dict[str, Any]:
    """A verifier exists but failed to run (bad spec, numerical failure, missing dep)."""
    return {
        "kind": kind,
        "checkable": False,
        "claim_supported": None,
        "evidence": {},
        "reason": f"verification failed: {message}",
        "error": message,
    }
