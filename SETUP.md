# Setup (lab onboarding)

Condensed-matter Co-Scientist: a multi-agent loop that generates, debates (Elo), evolves, and
**automatically physics-checks** research hypotheses. There are two ways to run it.

> **Auth, up front:** no credentials are stored in this repo. **Each person authenticates with
> their own Claude account locally.** Nothing you do below commits a key or login.

## A. Claude Agent SDK workflow — recommended (this is the active work)

Frontier-quality, code-orchestrated `generate → reflect → rank → evolve → meta-review` loop,
with a live arXiv tool and a pluggable physics verifier. Details: [docs/sdk_workflow.md](docs/sdk_workflow.md)
and [docs/verifiers.md](docs/verifiers.md).

### Prerequisites
1. **Python 3.11+.** Invoke it as `py` on Windows, `python3` on macOS/Linux.
2. **Node.js + the Claude Code CLI** (the Agent SDK drives it under the hood):
   ```
   npm install -g @anthropic-ai/claude-code
   ```
3. **Your own Claude auth** — pick one:
   - Log in with your Claude account (Pro/Max): run `claude` once and use `/login`; **or**
   - Set `ANTHROPIC_API_KEY=<your key>` (uses metered API billing instead of a subscription).

### Install & run (from this directory)
```
py -m pip install -r requirements-sdk.txt          # macOS/Linux: python3 -m pip install ...
py -m app.sdk.run --smoke                           # cheap wiring check (~2 model calls)
py -m app.sdk.run --num-hypotheses 3 --max-debate-pairs 3   # a small full run
```
- Results + **resumable checkpoints** are written to `results/` (gitignored). If a run is
  interrupted (e.g. you hit a usage limit), it prints a `--resume results/...checkpoint.json`
  command — rerun with that to continue without repeating spent work.
- Per-role models default to Opus (generation/evolution) + Sonnet (reflection/ranking); drop to
  `--reflection-model haiku --ranking-model haiku` for cheap iteration.

### Tests (pure compute — no API calls, no cost)
```
py -m pytest tests/test_verification_stub.py tests/test_plugin_registry.py tests/test_tight_binding.py tests/test_verification_integration.py tests/test_loop_orchestration.py tests/test_checkpoint.py -q
```

## B. Legacy Gradio web app (the upstream LLNL fork) — optional

A browser UI that uses **OpenRouter** (not your Claude login). Needs `OPENROUTER_API_KEY` and the
heavier `requirements.txt`. See [README.md](README.md). Not needed for the SDK workflow.

## Project layout
```
app/sdk/            Claude Agent SDK workflow (tools, roles, loop, run, checkpoint)
app/verification/   Pluggable physics verifiers; condensed_matter/ tight-binding is built in
app/tournament.py   Elo + verification-weighted ranking (shared by both frontends)
app/agents.py, app.py   Legacy OpenRouter/Gradio loop
docs/               sdk_workflow.md, verifiers.md, condensed-matter-setup.md (the project plan)
```

## Gotchas
- Run SDK commands **from the repo root** so `config.yaml` resolves.
- `config.yaml` configures only the legacy app; the SDK workflow is configured by CLI flags and
  the dataclasses in `app/sdk/loop.py`.
- Windows: use `py`, not `python` (the bare `python` is a Microsoft Store stub that fails).
