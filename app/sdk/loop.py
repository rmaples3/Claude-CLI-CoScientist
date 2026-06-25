"""Code-orchestrated Co-Scientist loop (generate -> reflect -> rank -> evolve -> meta-review).

Mirrors the structure of the fork's SupervisorAgent.run_cycle (app/agents.py) but async,
backed by the Claude Agent SDK roles, and with a verification-gated Elo tournament. Python
owns the loop and the tournament so the experiment stays deterministic and measurable; the
LLM is used only for the per-role reasoning steps.

Every expensive step calls an optional checkpoint callback so a mid-run failure (e.g. a
Claude usage-limit cutoff) loses at most the single in-flight call -- see app/sdk/checkpoint.py.
"""
import copy
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
        self._cycle_state: Dict[str, Any] = {}

    @property
    def resume_state(self) -> Dict[str, Any]:
        """Serializable position of the in-flight cycle for checkpoint metadata."""
        return copy.deepcopy(self._cycle_state)

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

    async def _reflect_one(self, h: Hypothesis, goal: str, model: str) -> None:
        try:
            review = await roles.reflect(hypothesis_text=h.text, goal=goal, model=model)
        except Exception as exc:
            logger.warning("Reflection failed for %s: %s", h.hypothesis_id, exc)
            raise
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

    async def _reflect_all(
        self, hypos: List[Hypothesis], goal: str, model: str, context: ContextMemory
    ) -> None:
        """Non-resumable reflection helper retained for the smoke-test path."""
        for h in hypos:
            await self._reflect_one(h, goal, model)
            self._checkpoint(context, "reflection")

    async def _tournament(
        self, active: List[Hypothesis], goal: str, model: str, context: ContextMemory,
        *, state: Optional[Dict[str, Any]] = None, phase_key: str = "ranking",
    ) -> None:
        cfg = self.config
        if len(active) < 2:
            logger.info("Not enough active hypotheses for a tournament (%d).", len(active))
            return

        pairs_key = f"{phase_key}_pairs"
        index_key = f"{phase_key}_index"
        if state is not None and pairs_key in state:
            pair_ids = list(state[pairs_key])
        else:
            pairs = list(itertools.combinations(active, 2))
            random.shuffle(pairs)
            pairs = pairs[: cfg.max_debate_pairs]
            pair_ids = [[a.hypothesis_id, b.hypothesis_id] for a, b in pairs]
            if state is not None:
                state[pairs_key] = pair_ids
                state[index_key] = 0

        start = int(state.get(index_key, 0)) if state is not None else 0
        logger.info("Running tournament: %d remaining debate(s) of %d.",
                    len(pair_ids) - start, len(pair_ids))
        by_id = {h.hypothesis_id: h for h in active}
        for idx in range(start, len(pair_ids)):
            a_id, b_id = pair_ids[idx]
            if a_id not in by_id or b_id not in by_id:
                raise ValueError(f"checkpoint tournament pair references missing hypothesis: {a_id}, {b_id}")
            a, b = by_id[a_id], by_id[b_id]
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
            if state is not None:
                state[index_key] = idx + 1
            self._checkpoint(context, phase_key)

    # -- entry points ------------------------------------------------------------------
    async def run_cycle(
        self, goal: ResearchGoal, context: ContextMemory,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run or resume one cycle from its last completed expensive operation.

        The state contains stable hypothesis IDs, reflection indices, and the exact shuffled
        tournament pair lists.  Persisting the pair lists is essential: recreating them on
        resume would both repeat Elo updates and change the experiment.
        """
        cfg, m = self.config, self.config.models
        resumable = isinstance(resume_state, dict) and resume_state.get("phase") not in (None, "complete")
        if resumable:
            state = copy.deepcopy(resume_state)
            expected_iteration = context.iteration_number + 1
            if int(state.get("iteration", expected_iteration)) != expected_iteration:
                raise ValueError(
                    "checkpoint workflow iteration does not match context: "
                    f"state={state.get('iteration')} expected={expected_iteration}"
                )
            logger.info("=== Resuming cycle %d at phase %s ===",
                        expected_iteration, state.get("phase"))
        else:
            state = {
                "version": 1,
                "iteration": context.iteration_number + 1,
                "phase": "generation",
                "new_ids": [],
                "reflection_index": 0,
            }
            logger.info("=== Cycle %d ===", state["iteration"])
        self._cycle_state = state
        cycle: Dict[str, Any] = {"iteration": int(state["iteration"]), "steps": {}}

        # 1. Generation (+ deterministic verification of each new hypothesis)
        if state["phase"] == "generation":
            existing_titles = [h.title for h in context.hypotheses.values()]
            ideas = await roles.generate(
                goal=goal.description, constraints=goal.constraints,
                existing_titles=existing_titles, n=cfg.num_hypotheses, model=m.generation,
            )
            new_ids: List[str] = []
            for idea in ideas:
                hid = _new_id("G", set(context.hypotheses.keys()))
                h = Hypothesis(hid, idea["title"], idea["text"])
                h.spec = idea.get("spec")
                self._verify(h)
                context.add_hypothesis(h)
                new_ids.append(hid)
            state["new_ids"] = new_ids
            state["reflection_index"] = 0
            state["phase"] = "reflection_new"
            logger.info("Generated %d hypotheses.", len(new_ids))
            self._checkpoint(context, "generation")

        new_ids = list(state.get("new_ids") or [])
        try:
            new_hypos = [context.hypotheses[hid] for hid in new_ids]
        except KeyError as exc:
            raise ValueError(f"checkpoint references missing generated hypothesis: {exc}") from exc

        # 2. Reflection, resuming at the first unfinished generated hypothesis.
        if state["phase"] == "reflection_new":
            start = int(state.get("reflection_index", 0))
            for idx in range(start, len(new_hypos)):
                await self._reflect_one(new_hypos[idx], goal.description, m.reflection)
                state["reflection_index"] = idx + 1
                self._checkpoint(context, "reflection")
            state["phase"] = "ranking_1"
            self._checkpoint(context, "reflection_complete")

        cycle["steps"]["generation"] = [h.to_dict() for h in new_hypos]
        cycle["steps"]["reflection"] = [h.to_dict() for h in new_hypos]

        # 3. Ranking (tournament 1). Pair order and next index are checkpointed.
        if state["phase"] == "ranking_1":
            await self._tournament(
                context.get_active_hypotheses(), goal.description, m.ranking, context,
                state=state, phase_key="ranking_1",
            )
            state["phase"] = "evolution"
            self._checkpoint(context, "ranking_1_complete")

        # 4. Evolution (synthesize the top parents, then reflect on the child).
        if state["phase"] == "evolution":
            if "parent_ids" not in state:
                ranked = sorted(context.get_active_hypotheses(), key=lambda x: x.elo_score, reverse=True)
                state["parent_ids"] = [h.hypothesis_id for h in ranked[: cfg.top_k]]
            try:
                parents = [context.hypotheses[hid] for hid in state["parent_ids"]]
            except KeyError as exc:
                raise ValueError(f"checkpoint references missing evolution parent: {exc}") from exc

            evolved_id = state.get("evolved_id")
            if len(parents) >= 2 and not evolved_id:
                child = await roles.evolve(
                    goal=goal.description, parents=[p.to_dict() for p in parents], model=m.evolution
                )
                if child:
                    evolved_id = _new_id("E", set(context.hypotheses.keys()))
                    evolved = Hypothesis(evolved_id, child["title"], child["text"])
                    evolved.spec = child.get("spec")
                    evolved.parent_ids = [p.hypothesis_id for p in parents]
                    self._verify(evolved)
                    context.add_hypothesis(evolved)
                    state["evolved_id"] = evolved_id
                    state["evolved_reflected"] = False
            state["phase"] = "reflection_evolved" if evolved_id else "ranking_2"
            self._checkpoint(context, "evolution")

        if state["phase"] == "reflection_evolved":
            evolved_id = state.get("evolved_id")
            if not evolved_id or evolved_id not in context.hypotheses:
                raise ValueError("checkpoint is missing its evolved hypothesis")
            if not state.get("evolved_reflected", False):
                await self._reflect_one(context.hypotheses[evolved_id], goal.description, m.reflection)
                state["evolved_reflected"] = True
            state["phase"] = "ranking_2"
            self._checkpoint(context, "reflection_evolved")

        evolved_id = state.get("evolved_id")
        evolved = context.hypotheses.get(evolved_id) if evolved_id else None
        cycle["steps"]["evolution"] = [evolved.to_dict()] if evolved else []

        # 5. Ranking (tournament 2), independently resumable from tournament 1.
        if state["phase"] == "ranking_2":
            await self._tournament(
                context.get_active_hypotheses(), goal.description, m.ranking, context,
                state=state, phase_key="ranking_2",
            )
            state["phase"] = "meta_review"
            self._checkpoint(context, "ranking_2_complete")

        # 6. Meta-review
        if state["phase"] == "meta_review":
            ranked = sorted(context.get_active_hypotheses(), key=lambda x: x.elo_score, reverse=True)
            overview = await roles.meta_review(
                goal=goal.description, ranked=[h.to_dict() for h in ranked], model=m.meta_review
            )
            context.meta_review_feedback.append(overview)
            context.iteration_number += 1
            state["phase"] = "complete"
            self._checkpoint(context, "meta_review")
        else:
            ranked = sorted(context.get_active_hypotheses(), key=lambda x: x.elo_score, reverse=True)
            overview = context.meta_review_feedback[-1] if context.meta_review_feedback else {}

        cycle["meta_review"] = overview
        cycle["ranked"] = [h.to_dict() for h in ranked]
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
