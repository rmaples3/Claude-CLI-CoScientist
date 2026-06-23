"""Code-orchestrated Co-Scientist loop (generate -> reflect -> rank -> evolve -> meta-review).

Mirrors the structure of the fork's SupervisorAgent.run_cycle (app/agents.py) but async,
backed by the Claude Agent SDK roles, and with a verification-gated Elo tournament. Python
owns the loop and the tournament so the experiment stays deterministic and measurable; the
LLM is used only for the per-role reasoning steps.

Every expensive step calls an optional checkpoint callback so a mid-run failure (e.g. a
Claude usage-limit cutoff) loses at most the single in-flight call -- see app/sdk/checkpoint.py.
"""
import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from ..models import ContextMemory, Hypothesis, ResearchGoal
from ..tournament import compare, update_elo
from ..verification import run_verification, unverifiable
from . import roles

logger = logging.getLogger("coscientist.sdk")

# Called as cb(context, stage) after each expensive step to persist progress.
CheckpointCB = Callable[[ContextMemory, str], None]


@dataclass
class ModelConfig:
    """Per-role model aliases. Cost-managed defaults: frontier where quality matters most,
    cheaper for the call-heavy roles (the setup doc's token-burn pitfall)."""
    generation: str = "opus"
    reflection: str = "sonnet"
    ranking: str = "sonnet"
    evolution: str = "opus"
    meta_review: str = "sonnet"


@dataclass
class WorkflowConfig:
    num_hypotheses: int = 6
    elo_k_factor: int = 32
    top_k: int = 2
    max_debate_pairs: int = 12  # cap pairwise debates per tournament to bound token burn
    models: ModelConfig = field(default_factory=ModelConfig)


def _new_id(prefix: str, existing: Set[str]) -> str:
    """Local id generator (avoids importing app.utils, which pulls in torch/openai)."""
    while True:
        hid = f"{prefix}{random.randint(1000, 9999)}"
        if hid not in existing:
            return hid


class CoScientistWorkflow:
    def __init__(self, config: Optional[WorkflowConfig] = None, checkpoint_cb: Optional[CheckpointCB] = None):
        self.config = config or WorkflowConfig()
        self._checkpoint_cb = checkpoint_cb

    # -- helpers -----------------------------------------------------------------------
    def _checkpoint(self, context: ContextMemory, stage: str) -> None:
        if self._checkpoint_cb is None:
            return
        try:
            self._checkpoint_cb(context, stage)
        except Exception as exc:  # never let checkpointing crash the run
            logger.warning("Checkpoint failed at stage '%s': %s", stage, exc)

    def _verify(self, h: Hypothesis) -> None:
        """Deterministically compute the score-affecting verification verdict for h."""
        if isinstance(h.spec, dict):
            h.verification = run_verification(h.spec)
        else:
            h.verification = unverifiable("none", "no machine-checkable spec provided")
        logger.info(
            "Verified %s: checkable=%s claim_supported=%s",
            h.hypothesis_id, h.verification.get("checkable"), h.verification.get("claim_supported"),
        )

    async def _reflect_all(
        self, hypos: List[Hypothesis], goal: str, model: str, context: ContextMemory
    ) -> None:
        for h in hypos:
            try:
                review = await roles.reflect(hypothesis_text=h.text, goal=goal, model=model)
            except Exception as exc:
                logger.warning("Reflection failed for %s: %s", h.hypothesis_id, exc)
                raise  # propagate so a usage-limit cutoff stops the run (state already saved)
            h.novelty_review = review["novelty_review"]
            h.feasibility_review = review["feasibility_review"]
            if review["comment"]:
                h.review_comments.append(review["comment"])
            if review["references"]:
                h.references.extend(review["references"])
            logger.info(
                "Reviewed %s: novelty=%s feasibility=%s",
                h.hypothesis_id, h.novelty_review, h.feasibility_review,
            )
            self._checkpoint(context, "reflection")  # save after each review

    async def _tournament(
        self, active: List[Hypothesis], goal: str, model: str, context: ContextMemory
    ) -> None:
        cfg = self.config
        if len(active) < 2:
            logger.info("Not enough active hypotheses for a tournament (%d).", len(active))
            return
        pairs = list(itertools.combinations(active, 2))
        random.shuffle(pairs)
        pairs = pairs[: cfg.max_debate_pairs]
        logger.info("Running tournament: %d debate(s).", len(pairs))
        for a, b in pairs:
            try:
                pick = await roles.debate(goal=goal, hypo_a=a.to_dict(), hypo_b=b.to_dict(), model=model)
                winner = a if pick == "A" else b
            except ValueError as exc:
                # Unparseable debate output -> deterministic fallback (not a usage problem).
                logger.warning("Debate %s vs %s unparseable (%s); using deterministic compare.",
                               a.hypothesis_id, b.hypothesis_id, exc)
                winner = compare(a, b)
            loser = b if winner is a else a
            update_elo(winner, loser, cfg.elo_k_factor)
            context.tournament_results.append({
                "iteration": context.iteration_number,
                "winner": winner.hypothesis_id,
                "loser": loser.hypothesis_id,
                "winner_score_after": winner.elo_score,
                "loser_score_after": loser.elo_score,
            })
            self._checkpoint(context, "ranking")  # save after each match

    # -- entry points ------------------------------------------------------------------
    async def run_cycle(self, goal: ResearchGoal, context: ContextMemory) -> Dict[str, Any]:
        cfg, m = self.config, self.config.models
        cycle: Dict[str, Any] = {"iteration": context.iteration_number + 1, "steps": {}}
        logger.info("=== Cycle %d ===", cycle["iteration"])

        # 1. Generation (+ deterministic verification of each new hypothesis)
        existing_titles = [h.title for h in context.hypotheses.values()]
        ideas = await roles.generate(
            goal=goal.description, constraints=goal.constraints,
            existing_titles=existing_titles, n=cfg.num_hypotheses, model=m.generation,
        )
        new_hypos: List[Hypothesis] = []
        for idea in ideas:
            hid = _new_id("G", set(context.hypotheses.keys()))
            h = Hypothesis(hid, idea["title"], idea["text"])
            h.spec = idea.get("spec")
            self._verify(h)
            context.add_hypothesis(h)
            new_hypos.append(h)
        cycle["steps"]["generation"] = [h.to_dict() for h in new_hypos]
        logger.info("Generated %d hypotheses.", len(new_hypos))
        self._checkpoint(context, "generation")

        # 2. Reflection
        await self._reflect_all(new_hypos, goal.description, m.reflection, context)
        cycle["steps"]["reflection"] = [h.to_dict() for h in new_hypos]

        # 3. Ranking (tournament 1)
        active = context.get_active_hypotheses()
        await self._tournament(active, goal.description, m.ranking, context)

        # 4. Evolution (synthesize the top parents, then reflect on the child)
        ranked = sorted(context.get_active_hypotheses(), key=lambda x: x.elo_score, reverse=True)
        parents = ranked[: cfg.top_k]
        evolved: Optional[Hypothesis] = None
        if len(parents) >= 2:
            child = await roles.evolve(
                goal=goal.description, parents=[p.to_dict() for p in parents], model=m.evolution
            )
            if child:
                hid = _new_id("E", set(context.hypotheses.keys()))
                evolved = Hypothesis(hid, child["title"], child["text"])
                evolved.spec = child.get("spec")
                evolved.parent_ids = [p.hypothesis_id for p in parents]
                self._verify(evolved)
                context.add_hypothesis(evolved)
                self._checkpoint(context, "evolution")
                await self._reflect_all([evolved], goal.description, m.reflection, context)
        cycle["steps"]["evolution"] = [evolved.to_dict()] if evolved else []

        # 5. Ranking (tournament 2, now including the evolved hypothesis)
        active = context.get_active_hypotheses()
        await self._tournament(active, goal.description, m.ranking, context)

        # 6. Meta-review
        ranked = sorted(active, key=lambda x: x.elo_score, reverse=True)
        overview = await roles.meta_review(
            goal=goal.description, ranked=[h.to_dict() for h in ranked], model=m.meta_review
        )
        context.meta_review_feedback.append(overview)
        cycle["meta_review"] = overview
        cycle["ranked"] = [h.to_dict() for h in ranked]

        context.iteration_number += 1
        self._checkpoint(context, "meta_review")
        logger.info("=== Cycle %d complete ===", context.iteration_number)
        return cycle

    async def run_smoke(self, goal: ResearchGoal, context: ContextMemory) -> Dict[str, Any]:
        """Cheap wiring check: generate 2 hypotheses + reflect on them. No tournament/evolution.

        Confirms the SDK connects, the arXiv tool is reachable, and JSON parses end-to-end.
        """
        m = self.config.models
        logger.info("=== Smoke test ===")
        ideas = await roles.generate(
            goal=goal.description, constraints=goal.constraints,
            existing_titles=[], n=2, model=m.generation,
        )
        new_hypos: List[Hypothesis] = []
        for idea in ideas:
            hid = _new_id("G", set(context.hypotheses.keys()))
            h = Hypothesis(hid, idea["title"], idea["text"])
            h.spec = idea.get("spec")
            self._verify(h)
            context.add_hypothesis(h)
            new_hypos.append(h)
        await self._reflect_all(new_hypos, goal.description, m.reflection, context)
        logger.info("Smoke test produced %d reviewed hypotheses.", len(new_hypos))
        return {"hypotheses": [h.to_dict() for h in new_hypos]}
