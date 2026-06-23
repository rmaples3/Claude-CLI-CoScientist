"""Verifier registry + plugin discovery.

Verifiers register by hypothesis ``kind``. Two sources are loaded lazily on first use:
  1. Built-ins shipped in this repo (currently condensed_matter).
  2. External plugins advertised under the ``coscientist.verifiers`` entry-point group --
     any pip-installed package (e.g. a future ``coscientist-biophysics``) can add verifiers
     without touching this code. See docs/verifiers.md for the plugin contract.

Unregistered kinds fall through to the stub (reported as not-yet-checkable). Nothing here
raises: a bad spec or a verifier failure is captured and returned as evidence.
"""
import importlib.metadata as importlib_metadata
import json
import logging
from typing import Any, Callable, Dict, List

from .base import Verifier, error
from .stub import verify_stub

logger = logging.getLogger("coscientist.verification")

ENTRYPOINT_GROUP = "coscientist.verifiers"

_REGISTRY: Dict[str, Verifier] = {}
_loaded = False


def register(verifier: Verifier) -> None:
    """Register (or replace) a verifier for its kind."""
    _REGISTRY[verifier.kind] = verifier


def unregister(kind: str) -> None:
    _REGISTRY.pop(kind, None)


def _load_builtins() -> None:
    from .condensed_matter import get_verifiers
    for v in get_verifiers():
        register(v)


def _load_plugins() -> None:
    try:
        eps = importlib_metadata.entry_points(group=ENTRYPOINT_GROUP)
    except TypeError:  # importlib.metadata < 3.10 API
        eps = importlib_metadata.entry_points().get(ENTRYPOINT_GROUP, [])
    for ep in eps:
        try:
            factory: Callable[[], List[Verifier]] = ep.load()
            for v in factory():
                register(v)
            logger.info("Loaded verifier plugin '%s' (%s)", ep.name, ep.value)
        except Exception as exc:  # a broken plugin must not break the core
            logger.warning("Failed to load verifier plugin %s: %s", getattr(ep, "name", ep), exc)


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True  # set first so a plugin importing this module can't recurse
    try:
        _load_builtins()
    except Exception as exc:
        logger.warning("Failed to load built-in verifiers: %s", exc)
    _load_plugins()


def available_verifiers() -> Dict[str, Verifier]:
    _ensure_loaded()
    return dict(_REGISTRY)


def available_kinds() -> List[str]:
    _ensure_loaded()
    return sorted(_REGISTRY)


def describe_verifiers() -> str:
    """Prompt-friendly catalogue of registered verifiers (kind, description, example spec)."""
    _ensure_loaded()
    if not _REGISTRY:
        return "(no machine verifiers available)"
    lines: List[str] = []
    for v in _REGISTRY.values():
        lines.append(f'- kind "{v.kind}": {v.description}')
        if v.spec_example:
            lines.append(f"  example spec: {json.dumps(v.spec_example)}")
    return "\n".join(lines)


def run_verification(spec: Any) -> Dict[str, Any]:
    """Verify a structured hypothesis spec; always returns a VerificationResult dict."""
    _ensure_loaded()
    if not isinstance(spec, dict):
        return error("unknown", f"spec must be a dict, got {type(spec).__name__}")
    kind = spec.get("kind", "unknown")
    verifier = _REGISTRY.get(kind)
    if verifier is None:
        return verify_stub(spec)
    try:
        return verifier.run(spec)
    except Exception as exc:  # keep the loop alive; surface the failure as evidence
        return error(kind, str(exc))
