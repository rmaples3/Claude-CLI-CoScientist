"""CLI entry point for the Claude Agent SDK condensed-matter Co-Scientist (Stage 1).

Run from the repo root (open-ai-co-scientist/) so config.yaml resolves:

    python -m app.sdk.run --smoke
    python -m app.sdk.run --cycles 1 --num-hypotheses 4

Resilience: the run checkpoints its full state after every step. If it's interrupted
(most importantly a Claude usage-limit cutoff), partial work is saved to both the results
JSON and a checkpoint file, and you can continue with:

    python -m app.sdk.run --resume results/sdk_run_<ts>.checkpoint.json

Auth: the Claude Agent SDK uses your Claude Code CLI login (Claude Pro/Max) or the
ANTHROPIC_API_KEY environment variable. No OpenRouter key is needed for this path.
"""
import argparse
import asyncio
import datetime
import json
import logging
import os
import sys

from ..models import ContextMemory, ResearchGoal
from .checkpoint import load_checkpoint, save_checkpoint
from .loop import CoScientistWorkflow, ModelConfig, WorkflowConfig

# Machine-verifiable goal from the setup doc (Sec. 3) -- written so its central claim
# can eventually be checked numerically (the wedge for the verification layer).
DEFAULT_GOAL = (
    "Propose minimal tight-binding models (≤4 sites/cell, 2D, nearest- and next-nearest-neighbor hopping) "
    "hosting an exactly or nearly flat band (bandwidth < 0.05t) with Chern number C=2 or C=-2. "
    "C=1 flat Chern bands are well-catalogued (checkerboard, decorated honeycomb); C=2 is much rarer. "
    "For each model, provide a tight_binding spec with flat_band + chern_number claims so the verifier "
    "can confirm. Prefer models with a realization pathway in cold atoms or 2D materials."
)

RESULTS_DIR = "results"

# Substrings that suggest the interruption was a usage/rate/quota limit (best-effort).
_USAGE_LIMIT_HINTS = ("usage limit", "rate limit", "rate_limit", "quota", "429", "resets at",
                      "limit reached", "overloaded", "credit")


def _setup_logging(log_path: str) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Condensed-matter Co-Scientist (Claude Agent SDK).")
    p.add_argument("--goal", default=DEFAULT_GOAL, help="Research goal text.")
    p.add_argument("--cycles", type=int, default=1, help="Number of full cycles to run.")
    p.add_argument("--num-hypotheses", type=int, default=6, help="Hypotheses generated per cycle.")
    p.add_argument("--top-k", type=int, default=2, help="Top hypotheses fed to Evolution.")
    p.add_argument("--max-debate-pairs", type=int, default=12, help="Max pairwise debates per tournament.")
    p.add_argument("--smoke", action="store_true", help="Cheap wiring check (generate + reflect only).")
    p.add_argument("--resume", default=None, metavar="CHECKPOINT_JSON",
                   help="Continue from a saved checkpoint (preserves prior hypotheses/Elo/reviews).")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="Where to write the checkpoint (default: results/sdk_run_<ts>.checkpoint.json).")
    p.add_argument("--generation-model", default="opus")
    p.add_argument("--reflection-model", default="sonnet")
    p.add_argument("--ranking-model", default="sonnet")
    p.add_argument("--evolution-model", default="opus")
    p.add_argument("--meta-review-model", default="sonnet")
    return p.parse_args(argv)


async def _run(args, ckpt_path: str) -> dict:
    models = ModelConfig(
        generation=args.generation_model,
        reflection=args.reflection_model,
        ranking=args.ranking_model,
        evolution=args.evolution_model,
        meta_review=args.meta_review_model,
    )
    config = WorkflowConfig(
        num_hypotheses=args.num_hypotheses,
        top_k=args.top_k,
        max_debate_pairs=args.max_debate_pairs,
        models=models,
    )
    goal = ResearchGoal(description=args.goal)
    context = load_checkpoint(args.resume) if args.resume else ContextMemory()

    def _cb(ctx: ContextMemory, stage: str) -> None:
        save_checkpoint(ckpt_path, ctx, meta={"goal": args.goal, "stage": stage})

    workflow = CoScientistWorkflow(config, checkpoint_cb=_cb)

    if args.smoke:
        return {"mode": "smoke", "goal": args.goal, "result": await workflow.run_smoke(goal, context)}

    cycles: list = []
    error = None
    try:
        for _ in range(args.cycles):
            cycles.append(await workflow.run_cycle(goal, context))
    except (Exception, KeyboardInterrupt) as exc:  # save partial work on ANY interruption
        error = f"{type(exc).__name__}: {exc}"
        logging.getLogger("coscientist.sdk").error("Run interrupted: %s", error)

    # Always persist a final checkpoint with whatever we have.
    save_checkpoint(ckpt_path, context, meta={"goal": args.goal, "stage": "final", "error": error})

    ranked = sorted(context.get_active_hypotheses(), key=lambda h: h.elo_score, reverse=True)
    return {
        "mode": "cycles",
        "goal": args.goal,
        "completed": error is None,
        "error": error,
        "checkpoint": ckpt_path,
        "cycles": cycles,
        # Snapshot from context so hypotheses are saved even if the cycle didn't finish.
        "hypotheses_ranked": [h.to_dict() for h in ranked],
        "tournament_results": context.tournament_results,
        "final_overview": context.meta_review_feedback[-1] if context.meta_review_feedback else None,
    }


def _print_summary(output: dict) -> None:
    line = "=" * 72
    print("\n" + line)
    if output["mode"] == "smoke":
        print("SMOKE TEST -- reviewed hypotheses:")
        for h in output["result"]["hypotheses"]:
            v = h.get("verification") or {}
            print(f"  - {h['title']} (novelty={h['novelty_review']}, "
                  f"feasibility={h['feasibility_review']}, checkable={v.get('checkable')})")
        print(line)
        return
    if output.get("error"):
        print(f"!! INTERRUPTED: {output['error']}")
        print(f"!! Partial work saved. Resume with:\n"
              f"     py -m app.sdk.run --resume \"{output.get('checkpoint')}\"")
    ranked = output.get("hypotheses_ranked", [])
    print("RANKED HYPOTHESES:" if ranked else "No hypotheses produced.")
    for i, h in enumerate(ranked, 1):
        v = h.get("verification") or {}
        print(f"  {i}. [{h['elo_score']:.0f}] {h['title']} (nov={h['novelty_review']}, "
              f"feas={h['feasibility_review']}, checkable={v.get('checkable')})")
    overview = output.get("final_overview") or {}
    if overview:
        print("\nMETA-REVIEW SUMMARY:")
        print(f"  {overview.get('summary', '(none)')}")
        for step in overview.get("suggested_next_steps", []):
            print(f"   - next: {step}")
    print(line)


def main(argv=None) -> None:
    args = _parse_args(argv)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = "smoke" if args.smoke else "run"
    log_path = os.path.join(RESULTS_DIR, f"sdk_{tag}_{ts}.log")
    json_path = os.path.join(RESULTS_DIR, f"sdk_{tag}_{ts}.json")
    ckpt_path = args.checkpoint or os.path.join(RESULTS_DIR, f"sdk_{tag}_{ts}.checkpoint.json")
    _setup_logging(log_path)
    log = logging.getLogger("coscientist.sdk")

    try:
        import claude_agent_sdk  # noqa: F401  (import check for a clear error message)
    except ImportError:
        log.error("claude-agent-sdk is not installed. Run: pip install -r requirements-sdk.txt")
        sys.exit(1)

    log.info("Goal: %s", args.goal)
    if args.resume:
        log.info("Resuming from checkpoint: %s", args.resume)
    if not args.smoke:
        log.info("Checkpoint (resume target if interrupted): %s", ckpt_path)

    try:
        output = asyncio.run(_run(args, ckpt_path))
    except Exception:
        log.exception("Fatal error setting up the run.")
        sys.exit(1)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    log.info("Wrote results to %s", json_path)

    err = output.get("error")
    if err:
        if any(hint in err.lower() for hint in _USAGE_LIMIT_HINTS):
            log.error("This looks like a Claude usage limit. Partial work was saved; "
                      "retry after your limit resets.")
        log.error("Resume after reset with:  py -m app.sdk.run --resume \"%s\"",
                  output.get("checkpoint", ckpt_path))

    _print_summary(output)
    if err:
        sys.exit(2)


if __name__ == "__main__":
    main()
