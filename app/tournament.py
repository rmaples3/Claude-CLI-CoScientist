"""Elo tournament math + verification-weighted comparison.

Single source of truth for the pairwise-tournament scoring used by both frontends
(the Claude Agent SDK workflow and the OpenAI/vLLM loop). Kept dependency-light on
purpose -- only stdlib + app.models -- so the SDK path can import it without pulling
in the openai/torch/sentence-transformers stack that app.agents/app.utils carry.
"""
import math
import random

from .models import Hypothesis

# How much a passed/failed automated physics check shifts a hypothesis's tournament
# score, in the same units as the novelty+feasibility score (HIGH/MED/LOW = 3/2/1).
# This is the lever the contribution turns: a verified-correct hypothesis should win
# ties and edge out an unverified rival of equal review quality; a verified-wrong one
# should be pushed down hard.
VERIFICATION_BONUS = 2.0
VERIFICATION_PENALTY = 3.0

_REVIEW_POINTS = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def expected_score(rating: float, opponent_rating: float) -> float:
    """Standard Elo expected score for ``rating`` vs ``opponent_rating``."""
    return 1.0 / (1.0 + math.pow(10, (opponent_rating - rating) / 400.0))


def update_elo(winner: Hypothesis, loser: Hypothesis, k_factor: int) -> None:
    """Update both hypotheses' Elo scores in place after a decided match."""
    expected_winner = expected_score(winner.elo_score, loser.elo_score)
    expected_loser = 1.0 - expected_winner
    winner.elo_score = winner.elo_score + k_factor * (1 - expected_winner)
    loser.elo_score = loser.elo_score + k_factor * (0 - expected_loser)


def review_score(h: Hypothesis) -> int:
    """Novelty + feasibility points (HIGH/MEDIUM/LOW -> 3/2/1, missing -> 0)."""
    nov = _REVIEW_POINTS.get(h.novelty_review, 0) if isinstance(h.novelty_review, str) else 0
    feas = _REVIEW_POINTS.get(h.feasibility_review, 0) if isinstance(h.feasibility_review, str) else 0
    return nov + feas


def verification_adjustment(h: Hypothesis) -> float:
    """Score shift from the automated physics check; 0 when not checkable."""
    v = h.verification
    if not isinstance(v, dict) or not v.get("checkable"):
        return 0.0
    if v.get("claim_supported") is True:
        return VERIFICATION_BONUS
    if v.get("claim_supported") is False:
        return -VERIFICATION_PENALTY
    return 0.0


def merit(h: Hypothesis) -> float:
    """Verification-weighted merit used by the deterministic comparator."""
    return review_score(h) + verification_adjustment(h)


def compare(hypo_a: Hypothesis, hypo_b: Hypothesis) -> Hypothesis:
    """Deterministic verification-weighted comparison; returns the winner.

    Mirrors the fork's run_pairwise_debate but adds the verification adjustment, so a
    checkably-correct hypothesis beats an equally-reviewed but unverified one. Used as
    the tie-breaker / fallback when an LLM debate is unavailable or unparseable.
    """
    score_a, score_b = merit(hypo_a), merit(hypo_b)
    if score_a > score_b:
        return hypo_a
    if score_b > score_a:
        return hypo_b
    return random.choice([hypo_a, hypo_b])
