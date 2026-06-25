"""Offline tests for checkpointing + graceful mid-run failure (no LLM, no network, no credits).

Run from the repo root:  python -m pytest tests/test_checkpoint.py
"""
import asyncio
import json
import os

import app.sdk.roles as roles
from app.models import ContextMemory, Hypothesis, ResearchGoal
from app.sdk import run as runmod
from app.sdk.checkpoint import context_from_dict, context_to_dict, load_checkpoint, save_checkpoint
from app.sdk.loop import CoScientistWorkflow, WorkflowConfig


def _make_hypo(hid, elo=1200.0, checkable=False):
    h = Hypothesis(hid, f"title {hid}", f"text {hid}")
    h.novelty_review = "HIGH"
    h.feasibility_review = "MEDIUM"
    h.elo_score = elo
    h.references = ["2301.00001"]
    h.spec = {"kind": "tight_binding"} if checkable else None
    h.verification = {"kind": "none", "checkable": False, "claim_supported": None,
                      "evidence": {}, "reason": "stub", "error": None}
    return h


def test_context_roundtrip():
    ctx = ContextMemory()
    ctx.iteration_number = 2
    ctx.add_hypothesis(_make_hypo("G1", 1216.0, checkable=True))
    ctx.add_hypothesis(_make_hypo("G2", 1184.0))
    ctx.tournament_results = [{"winner": "G1", "loser": "G2"}]
    ctx.meta_review_feedback = [{"summary": "s"}]

    restored = context_from_dict(context_to_dict(ctx))
    assert restored.iteration_number == 2
    assert set(restored.hypotheses) == {"G1", "G2"}
    assert restored.hypotheses["G1"].elo_score == 1216.0
    assert restored.hypotheses["G1"].spec == {"kind": "tight_binding"}
    assert restored.hypotheses["G1"].novelty_review == "HIGH"
    assert restored.tournament_results == [{"winner": "G1", "loser": "G2"}]
    assert restored.meta_review_feedback[0]["summary"] == "s"


def test_save_load_file(tmp_path):
    ctx = ContextMemory()
    ctx.add_hypothesis(_make_hypo("G1"))
    path = str(tmp_path / "ckpt.json")
    save_checkpoint(path, ctx, meta={"goal": "g", "stage": "generation"})
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")  # atomic write cleaned up
    loaded = load_checkpoint(path)
    assert "G1" in loaded.hypotheses
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["meta"]["stage"] == "generation"


# --- stubbed roles for the resilience tests ---
async def _fake_generate(*, goal, constraints, existing_titles, n, model):
    return [{"title": f"H{i}", "text": f"t{i}", "spec": None} for i in range(n)]


async def _fake_reflect(*, hypothesis_text, goal, model):
    return {"novelty_review": "HIGH", "feasibility_review": "MEDIUM", "comment": "ok", "references": []}


async def _fake_debate(*, goal, hypo_a, hypo_b, model):
    return "A"


async def _boom_evolve(*, goal, parents, model):
    raise RuntimeError("usage limit reached")


def test_failure_midcycle_preserves_checkpoint(monkeypatch, tmp_path):
    """A mid-cycle failure (evolution) leaves a checkpoint with all prior work."""
    monkeypatch.setattr(roles, "generate", _fake_generate)
    monkeypatch.setattr(roles, "reflect", _fake_reflect)
    monkeypatch.setattr(roles, "debate", _fake_debate)
    monkeypatch.setattr(roles, "evolve", _boom_evolve)

    ckpt = str(tmp_path / "ck.json")

    def cb(ctx, stage):
        save_checkpoint(ckpt, ctx, meta={"stage": stage})

    wf = CoScientistWorkflow(WorkflowConfig(num_hypotheses=3, max_debate_pairs=5), checkpoint_cb=cb)
    ctx = ContextMemory()

    raised = False
    try:
        asyncio.run(wf.run_cycle(ResearchGoal(description="g"), ctx))
    except RuntimeError:
        raised = True
    assert raised  # evolution failure propagated

    loaded = load_checkpoint(ckpt)
    assert len(loaded.hypotheses) == 3              # all generated hypotheses preserved
    assert all(h.novelty_review == "HIGH" for h in loaded.hypotheses.values())  # reviews preserved
    assert loaded.tournament_results                # tournament ran before the failure


def test_run_returns_partial_on_failure(monkeypatch, tmp_path):
    """_run captures the usage-limit error, saves a checkpoint, and returns partial results."""
    monkeypatch.setattr(roles, "generate", _fake_generate)
    monkeypatch.setattr(roles, "reflect", _fake_reflect)
    monkeypatch.setattr(roles, "debate", _fake_debate)
    monkeypatch.setattr(roles, "evolve", _boom_evolve)

    ckpt = str(tmp_path / "ck.json")
    args = runmod._parse_args(["--cycles", "1", "--num-hypotheses", "3"])
    out = asyncio.run(runmod._run(args, ckpt))

    assert out["completed"] is False
    assert "usage limit" in out["error"].lower()
    assert len(out["hypotheses_ranked"]) == 3       # hypotheses present in output despite failure
    assert out["checkpoint"] == ckpt
    assert os.path.exists(ckpt)                       # checkpoint written for --resume


def test_resume_preserves_prior_hypotheses(monkeypatch, tmp_path):
    """Resuming from a checkpoint loads prior hypotheses; a new cycle builds on them."""
    monkeypatch.setattr(roles, "generate", _fake_generate)
    monkeypatch.setattr(roles, "reflect", _fake_reflect)
    monkeypatch.setattr(roles, "debate", _fake_debate)

    async def _ok_evolve(*, goal, parents, model):
        return {"title": "child", "text": "synthesis", "spec": None}

    async def _ok_meta(*, goal, ranked, model):
        return {"summary": "ov", "critique": [], "suggested_next_steps": []}

    monkeypatch.setattr(roles, "evolve", _ok_evolve)
    monkeypatch.setattr(roles, "meta_review", _ok_meta)

    ckpt = str(tmp_path / "ck.json")
    # Seed a checkpoint with two prior hypotheses.
    seed = ContextMemory()
    seed.iteration_number = 1
    seed.add_hypothesis(_make_hypo("G1"))
    seed.add_hypothesis(_make_hypo("G2"))
    save_checkpoint(ckpt, seed, meta={"stage": "seed"})

    args = runmod._parse_args(["--cycles", "1", "--num-hypotheses", "2", "--resume", ckpt])
    out = asyncio.run(runmod._run(args, ckpt))

    assert out["completed"] is True
    ids = {h["id"] for h in out["hypotheses_ranked"]}
    assert {"G1", "G2"}.issubset(ids)                # prior hypotheses retained
    assert len(out["hypotheses_ranked"]) >= 4        # 2 prior + 2 new + 1 evolved


def test_resume_continues_partial_reflection_without_regeneration(monkeypatch, tmp_path):
    """A reflection-stage checkpoint resumes the unfinished item, not Generation."""
    calls = {"generate": 0, "reflected": []}

    async def _must_not_generate(**kwargs):
        calls["generate"] += 1
        raise AssertionError("resume incorrectly restarted Generation")

    async def _record_reflect(*, hypothesis_text, goal, model):
        calls["reflected"].append(hypothesis_text)
        return {"novelty_review": "HIGH", "feasibility_review": "MEDIUM",
                "comment": "ok", "references": []}

    async def _ok_meta(*, goal, ranked, model):
        return {"summary": "resumed", "critique": [], "suggested_next_steps": []}

    monkeypatch.setattr(roles, "generate", _must_not_generate)
    monkeypatch.setattr(roles, "reflect", _record_reflect)
    monkeypatch.setattr(roles, "meta_review", _ok_meta)

    ckpt = str(tmp_path / "partial-reflection.json")
    seed = ContextMemory()
    reviewed = _make_hypo("G1")
    reviewed.novelty_review = "HIGH"
    reviewed.feasibility_review = "MEDIUM"
    pending = _make_hypo("G2")
    pending.novelty_review = None
    pending.feasibility_review = None
    seed.add_hypothesis(reviewed)
    seed.add_hypothesis(pending)
    save_checkpoint(
        ckpt,
        seed,
        meta={
            "goal": "g",
            "stage": "reflection",
            "error": "RuntimeError: usage limit reached",
            "workflow_state": {
                "version": 1,
                "iteration": 1,
                "phase": "reflection_new",
                "new_ids": ["G1", "G2"],
                "reflection_index": 1,
            },
        },
    )

    args = runmod._parse_args([
        "--cycles", "1", "--num-hypotheses", "2", "--top-k", "1",
        "--max-debate-pairs", "0", "--resume", ckpt,
    ])
    out = asyncio.run(runmod._run(args, ckpt))

    assert out["completed"] is True
    assert calls["generate"] == 0
    assert calls["reflected"] == ["text G2"]
    assert seed.iteration_number == 0  # the saved object itself is not mutated
    assert {h["id"] for h in out["hypotheses_ranked"]} == {"G1", "G2"}
