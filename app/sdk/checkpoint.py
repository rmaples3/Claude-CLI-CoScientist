"""Checkpoint (de)serialization for the SDK workflow.

Lets a run persist its full ContextMemory after every step so a mid-run failure --
most importantly a Claude usage-limit cutoff -- never loses generated hypotheses,
reviews, Elo scores, or tournament history. Writes are atomic (temp file + os.replace)
so an interrupted write can't corrupt the checkpoint. No model calls, no network.
"""
import json
import os
from typing import Any, Dict, Optional

from ..models import ContextMemory, Hypothesis


def hypothesis_from_dict(d: Dict[str, Any]) -> Hypothesis:
    """Rebuild a Hypothesis from its to_dict() form (no models.py change needed)."""
    h = Hypothesis(d["id"], d.get("title", ""), d.get("text", ""))
    h.novelty_review = d.get("novelty_review")
    h.feasibility_review = d.get("feasibility_review")
    h.elo_score = float(d.get("elo_score", 1200.0))
    h.review_comments = list(d.get("review_comments") or [])
    h.references = list(d.get("references") or [])
    h.is_active = bool(d.get("is_active", True))
    h.parent_ids = list(d.get("parent_ids") or [])
    h.spec = d.get("spec")
    h.verification = d.get("verification")
    return h


def context_to_dict(ctx: ContextMemory) -> Dict[str, Any]:
    return {
        "iteration_number": ctx.iteration_number,
        "hypotheses": [h.to_dict() for h in ctx.hypotheses.values()],
        "tournament_results": ctx.tournament_results,
        "meta_review_feedback": ctx.meta_review_feedback,
    }


def context_from_dict(data: Dict[str, Any]) -> ContextMemory:
    ctx = ContextMemory()
    ctx.iteration_number = int(data.get("iteration_number", 0))
    for d in data.get("hypotheses", []):
        ctx.add_hypothesis(hypothesis_from_dict(d))
    ctx.tournament_results = list(data.get("tournament_results") or [])
    ctx.meta_review_feedback = list(data.get("meta_review_feedback") or [])
    return ctx


def save_checkpoint(path: str, ctx: ContextMemory, meta: Optional[Dict[str, Any]] = None) -> None:
    """Atomically write the full context to `path` (temp file + os.replace)."""
    payload = {"meta": meta or {}, "context": context_to_dict(ctx)}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)  # atomic on the same filesystem


def load_checkpoint(path: str) -> ContextMemory:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    # Accept either {"meta":.., "context":..} or a bare context dict.
    return context_from_dict(payload.get("context", payload))
