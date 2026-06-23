"""Placeholder verifier used until the real PythTB/Kwant checks land (setup doc Sec. 6).

It accepts any spec and reports the hypothesis as not-yet-checkable, so the rest of
the pipeline (reflection, ranking, meta-review) can run end-to-end now and simply
gain real physics evidence later without any interface change.
"""
from typing import Any, Dict

from .base import unverifiable


def verify_stub(spec: Dict[str, Any]) -> Dict[str, Any]:
    kind = spec.get("kind", "unknown")
    return unverifiable(
        kind,
        f"no automated verifier is implemented for kind '{kind}' yet; "
        "treat this hypothesis as not-yet-checkable.",
    )
