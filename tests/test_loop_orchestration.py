"""Deterministic test of the orchestration loop with the LLM roles stubbed out.

Validates the parts of CoScientistWorkflow.run_cycle that are pure logic -- generation
bookkeeping, deterministic verification, the Elo tournament, evolution (child + parents),
and meta-review capture -- without any model calls or network. Complements the live
smoke test (which exercises the real Claude Agent SDK wiring).

Run from the repo root:  python -m pytest tests/test_loop_orchestration.py
"""
import asyncio

import app.sdk.roles as roles
from app.models import ContextMemory, ResearchGoal
from app.sdk.loop import CoScientistWorkflow, ModelConfig, WorkflowConfig


async def _fake_generate(*, goal, constraints, existing_titles, n, model):
    # First hypothesis carries a (stub-unverifiable) spec; the rest have none.
    return [
        {"title": f"H{i}", "text": f"hypothesis text {i}",
         "spec": ({"kind": "tight_binding"} if i == 0 else None)}
        for i in range(n)
    ]


async def _fake_reflect(*, hypothesis_text, goal, model):
    return {"novelty_review": "HIGH", "feasibility_review": "MEDIUM",
            "comment": "looks reasonable", "references": ["2301.00001"]}


async def _fake_debate(*, goal, hypo_a, hypo_b, model):
    return "A"  # deterministic: first of each pair wins


async def _fake_evolve(*, goal, parents, model):
    return {"title": "evolved child", "text": "synthesis of parents", "spec": None}


async def _fake_meta_review(*, goal, ranked, model):
    return {"summary": "overview", "critique": ["c1"], "suggested_next_steps": ["next1"]}


def test_run_cycle_orchestration(monkeypatch):
    monkeypatch.setattr(roles, "generate", _fake_generate)
    monkeypatch.setattr(roles, "reflect", _fake_reflect)
    monkeypatch.setattr(roles, "debate", _fake_debate)
    monkeypatch.setattr(roles, "evolve", _fake_evolve)
    monkeypatch.setattr(roles, "meta_review", _fake_meta_review)

    cfg = WorkflowConfig(num_hypotheses=3, top_k=2, max_debate_pairs=10, models=ModelConfig())
    wf = CoScientistWorkflow(cfg)
    ctx = ContextMemory()
    cycle = asyncio.run(wf.run_cycle(ResearchGoal(description="goal"), ctx))

    # 3 generated + 1 evolved = 4 hypotheses tracked
    assert len(ctx.hypotheses) == 4

    # Evolution produced exactly one child, with two recorded parents
    children = [h for h in ctx.hypotheses.values() if h.parent_ids]
    assert len(children) == 1
    assert len(children[0].parent_ids) == 2

    # The tournament ran and Elo diverged from the 1200 starting point
    assert ctx.tournament_results, "tournament recorded no matches"
    assert any(abs(h.elo_score - 1200.0) > 1e-6 for h in ctx.hypotheses.values()), "Elo never updated"

    # Reviews were applied
    assert all(h.novelty_review == "HIGH" for h in ctx.hypotheses.values())

    # Deterministic verification ran: the spec'd hypothesis is checkable=False (stub),
    # the spec-less ones are marked "none".
    specd = [h for h in ctx.hypotheses.values() if isinstance(h.spec, dict)]
    assert specd and specd[0].verification["checkable"] is False
    no_spec = [h for h in ctx.hypotheses.values() if h.spec is None]
    assert all(h.verification["kind"] == "none" for h in no_spec)

    # Meta-review captured into the cycle result and context
    assert cycle["meta_review"]["summary"] == "overview"
    assert cycle["ranked"]
    assert ctx.meta_review_feedback and ctx.meta_review_feedback[-1]["summary"] == "overview"


def test_smoke_path(monkeypatch):
    """run_smoke generates + reflects 2 hypotheses without tournament/evolution."""
    monkeypatch.setattr(roles, "generate", _fake_generate)
    monkeypatch.setattr(roles, "reflect", _fake_reflect)

    wf = CoScientistWorkflow(WorkflowConfig())
    ctx = ContextMemory()
    out = asyncio.run(wf.run_smoke(ResearchGoal(description="goal"), ctx))

    assert len(out["hypotheses"]) == 2
    assert not ctx.tournament_results  # smoke does not run a tournament
    assert all(h["novelty_review"] == "HIGH" for h in out["hypotheses"])
