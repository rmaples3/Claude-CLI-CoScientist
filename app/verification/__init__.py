"""Backend-agnostic, pluggable verification layer.

``run_verification(spec)`` is the single entry point shared by every frontend (the Claude
Agent SDK workflow and the OpenAI/vLLM loop). Verifiers are registered by hypothesis
``kind``: condensed-matter verifiers ship built in (``app/verification/condensed_matter``),
and external packages can plug in their own (e.g. a future biophysics package) via the
``coscientist.verifiers`` entry-point group -- see docs/verifiers.md. Unregistered kinds
fall through to the stub (reported as not-yet-checkable).
"""
from .base import Verifier, error, unverifiable, verdict
from .registry import (
    available_kinds,
    available_verifiers,
    describe_verifiers,
    register,
    run_verification,
    unregister,
)

__all__ = [
    "run_verification",
    "register",
    "unregister",
    "available_kinds",
    "available_verifiers",
    "describe_verifiers",
    "Verifier",
    "unverifiable",
    "verdict",
    "error",
]
