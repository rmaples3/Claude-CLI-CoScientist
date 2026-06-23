"""Claude Agent SDK frontend for the condensed-matter Co-Scientist (Stage 1).

A code-orchestrated generate->reflect->rank->evolve->meta-review loop that uses the
Claude Agent SDK for the per-role LLM calls and exposes arXiv search + the physics
verifier as in-process MCP tools. Shares the domain models (app.models), Elo math
(app.tournament), and verification layer (app.verification) with the OpenAI/vLLM loop.
"""
