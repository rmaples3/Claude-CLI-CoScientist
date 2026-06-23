# Claude Agent SDK workflow (Stage 1)

A code-orchestrated **generate → reflect → rank → evolve → meta-review** loop for
condensed-matter hypothesis discovery, built on the **Claude Agent SDK**. Python owns the
loop and the Elo tournament (so the experiment stays deterministic and measurable); the
Claude Agent SDK supplies the per-role reasoning and exposes **arXiv search** + an
**automated physics verifier** as in-process MCP tools.

This is the fast, frontier-quality path described in `coscientist-condensed-matter-setup.md`
§3. It shares the domain models, Elo math, and verification layer with the OpenAI/vLLM loop,
so nothing here is throwaway.

## Prerequisites

- The **Claude Code CLI** (Node-based) must be installed and logged in — the Agent SDK drives
  it under the hood. It uses your **Claude Pro/Max** subscription auth, or an `ANTHROPIC_API_KEY`
  environment variable if set. No OpenRouter key is needed for this path.
- Python deps (lightweight — no torch/openai/gradio):

  ```bash
  pip install -r requirements-sdk.txt
  ```

## Run

Always run from the repo root (`open-ai-co-scientist/`) so `config.yaml` resolves.

```bash
# Cheap wiring check: 2 generations + reflection, confirms the SDK connects,
# the arXiv tool is reachable, and JSON parses. Start here.
python -m app.sdk.run --smoke

# A full cycle (small, to keep token usage modest while iterating):
python -m app.sdk.run --cycles 1 --num-hypotheses 4

# Defaults: 6 hypotheses, 1 cycle, the flat-band/topological goal from the setup doc.
python -m app.sdk.run
```

Each run writes a timestamped `.log` and `.json` to `results/`.

### Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--goal "<text>"` | flat-band goal | The research goal. |
| `--cycles N` | 1 | Number of full cycles. |
| `--num-hypotheses N` | 6 | Hypotheses generated per cycle. |
| `--top-k N` | 2 | Top hypotheses fed to Evolution. |
| `--max-debate-pairs N` | 12 | Cap on pairwise LLM debates per tournament (token control). |
| `--smoke` | off | Generate + reflect only; no tournament/evolution. |
| `--{generation,reflection,ranking,evolution,meta-review}-model` | opus/sonnet | Per-role model alias. |

Cost-managed defaults put the frontier model (`opus`) on Generation/Evolution and `sonnet` on
the call-heavy Reflection/Ranking/Meta-review roles. Drop everything to `sonnet`/`haiku` for
cheap iteration.

## How it maps to the architecture

| Piece | Where |
|-------|-------|
| In-process MCP tools (arXiv, verifier) | `app/sdk/tools.py` |
| Role prompts + `run_role()` + JSON parsing | `app/sdk/roles.py` |
| Orchestration + tournament | `app/sdk/loop.py` |
| CLI / results | `app/sdk/run.py` |
| Domain models (shared) | `app/models.py` |
| Elo + verification weighting (shared) | `app/tournament.py` |
| Verification layer (shared, stubbed for now) | `app/verification/` |

The **verification verdict that affects ranking is computed in Python** from a hypothesis's
structured `spec` (`CoScientistWorkflow._verify`), not parsed from model output — so the
verification-weighted comparison is reproducible. The Ranking debate is additionally *shown*
the evidence so it can weight correctness on the LLM path.

## Current limitation (by design)

The physics verifier is a **stub**: every hypothesis comes back `checkable: false`
(not-yet-checkable). The loop runs fully and the verification plumbing is exercised end-to-end;
swapping in the real PythTB/Kwant verifier later (register it in `app/verification/__init__.py`)
needs no changes to the roles, the loop, or the tournament. See the setup doc §6.
