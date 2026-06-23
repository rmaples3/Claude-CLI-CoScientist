"""The six Co-Scientist roles, expressed as Claude Agent SDK calls.

Each role is a single ``query()`` with a role-specific system prompt and (optionally)
access to the in-process ``coscientist`` MCP tools. The Python orchestrator in
``loop.py`` owns the generate->reflect->rank->evolve->meta-review loop and the Elo
tournament; these functions are the per-role LLM steps it composes.

Design choices that matter:
  * ``run_role`` is **hermetic** -- ``tools=[]`` removes all built-in Bash/Read/Write
    tools so a role can only touch our two MCP tools (and only when we allow them).
  * Roles return strict JSON, parsed by ``parse_json`` (fence- and prose-tolerant).
  * The score-affecting verification verdict is computed by the loop directly from a
    hypothesis ``spec`` (deterministic), not parsed from model output. The Ranking
    debate is still *shown* the verification evidence so it can weight it -- that is
    the contribution's lever on the LLM path.
"""
import json
from typing import Any, Dict, List, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from ..verification import describe_verifiers
from .tools import ALL_TOOLS, SERVER_NAME, coscientist_server


# --------------------------------------------------------------------------------------
# Core SDK call + JSON parsing
# --------------------------------------------------------------------------------------
async def run_role(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    allowed_tools: Optional[List[str]] = None,
    max_turns: int = 12,
) -> str:
    """Run one role to completion and return its final text.

    When ``allowed_tools`` is empty the MCP server is omitted entirely, so the role has
    no tools at all (used for pure-synthesis roles like meta-review).
    """
    allowed = allowed_tools or []
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=model,
        mcp_servers={SERVER_NAME: coscientist_server} if allowed else {},
        allowed_tools=allowed,
        tools=[],  # hermetic: no built-in Bash/Read/Write/Edit/WebFetch -- only our MCP tools
        permission_mode="bypassPermissions",  # non-interactive batch; our tools are read-only
        setting_sources=None,  # do not load filesystem CLAUDE.md / project agents
        max_turns=max_turns,
    )
    chunks: List[str] = []
    final: Optional[str] = None
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
        elif isinstance(message, ResultMessage):
            final = message.result
    return (final if final is not None else "\n".join(chunks)).strip()


def parse_json(text: str) -> Any:
    """Best-effort extraction of a JSON value from an LLM reply.

    Strips ```json fences, then falls back to the outermost {...} or [...] span.
    Raises ValueError if nothing parses.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = s.find(opener), s.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from model output: {text[:300]!r}")


def _verification_brief(h_dict: Dict[str, Any]) -> str:
    """One-line summary of a hypothesis's verification evidence for a prompt."""
    v = h_dict.get("verification")
    if not isinstance(v, dict):
        return "verification: none"
    if not v.get("checkable"):
        return f"verification: not checkable ({v.get('reason', 'n/a')})"
    verdict = "SUPPORTED" if v.get("claim_supported") else "REFUTED"
    return f"verification: {verdict} -- evidence={json.dumps(v.get('evidence', {}))}"


# --------------------------------------------------------------------------------------
# Role system prompts
# --------------------------------------------------------------------------------------
GENERATION_SYSTEM = """You are the Generation agent of a condensed-matter physics Co-Scientist.
Propose novel, scientifically plausible, and -- where possible -- machine-verifiable hypotheses
that address the research goal.

Method:
- Call the arxiv_search tool to ground proposals in real, recent cond-mat literature. Prefer ideas
  whose central claim could be checked numerically (e.g. tight-binding band structure, flat bands,
  band gaps, topological invariants).
- Avoid duplicating any existing hypothesis titles you are given.
- For a hypothesis that is concretely checkable, you MAY include a structured "spec" object (a JSON
  object with a "kind" field, e.g. "tight_binding", and kind-specific fields). If you are unsure,
  set "spec" to null -- never invent fields you are not confident about.

Output: return ONLY a JSON array, no prose, where each element is:
  {"title": str, "text": str, "spec": object|null}
"text" should state the hypothesis, its physical rationale, and any arXiv IDs you relied on."""

REFLECTION_SYSTEM = """You are the Reflection agent of a condensed-matter physics Co-Scientist.
Critically review a single hypothesis for novelty and feasibility, and fact-check it.

Method:
- Use arxiv_search to check whether the idea (or close prior art) already exists and whether any
  arXiv IDs the hypothesis cites actually resolve. Do not trust citations you cannot confirm.
- Judge novelty and feasibility on a HIGH / MEDIUM / LOW scale.

Output: return ONLY a JSON object, no prose:
  {"novelty_review": "HIGH|MEDIUM|LOW",
   "feasibility_review": "HIGH|MEDIUM|LOW",
   "comment": str,
   "references": [str, ...]}
"references" are arXiv IDs / DOIs / titles you actually verified as relevant."""

RANKING_SYSTEM = """You are the Ranking agent of a condensed-matter physics Co-Scientist, running a
pairwise "scientific debate" between two hypotheses to decide which is stronger.

Weighing criteria, in priority order:
1. CORRECTNESS via automated verification. Each hypothesis may carry physics-verification evidence.
   A hypothesis whose central claim is computationally SUPPORTED should be strongly preferred over an
   unverified or REFUTED one, all else equal. Novel-but-wrong loses to less-novel-but-correct.
2. Novelty and scientific significance.
3. Feasibility and soundness of the reasoning.

You may use arxiv_search or verify_hypothesis to inform your judgment, but base the decision mainly on
the evidence provided.

Output: return ONLY a JSON object, no prose:
  {"winner": "A"|"B", "reason": str}"""

EVOLUTION_SYSTEM = """You are the Evolution agent of a condensed-matter physics Co-Scientist.
Synthesize a single improved hypothesis from the top-ranked parent hypotheses -- combine their
strengths, resolve their weaknesses, and sharpen toward something machine-verifiable. This is not a
concatenation: produce one coherent, better hypothesis.

You may use arxiv_search to ground the synthesis. As in Generation, include a structured "spec"
object only if you are confident it is checkable; otherwise null.

Output: return ONLY a JSON object, no prose:
  {"title": str, "text": str, "spec": object|null}"""

META_REVIEW_SYSTEM = """You are the Meta-review agent of a condensed-matter physics Co-Scientist.
Given the ranked hypotheses and tournament outcome, synthesize a concise research overview: what
patterns emerged, the strongest directions, and concrete next steps (including which hypotheses are
worth verifying or pursuing experimentally).

Output: return ONLY a JSON object, no prose:
  {"summary": str,
   "critique": [str, ...],
   "suggested_next_steps": [str, ...]}"""


# --------------------------------------------------------------------------------------
# Role functions (called by the orchestrator)
# --------------------------------------------------------------------------------------
async def generate(
    *, goal: str, constraints: Dict[str, Any], existing_titles: List[str], n: int, model: str
) -> List[Dict[str, Any]]:
    user = (
        f"Research goal:\n{goal}\n\n"
        f"Constraints: {json.dumps(constraints) if constraints else 'none'}\n"
        f"Existing hypothesis titles (do not duplicate): {existing_titles or 'none'}\n\n"
        f"Machine-verifiable hypothesis kinds (when a hypothesis fits one, include a matching "
        f"JSON `spec` so it can be checked automatically; otherwise set spec to null):\n"
        f"{describe_verifiers()}\n\n"
        f"Propose {n} new hypotheses now."
    )
    raw = await run_role(
        system_prompt=GENERATION_SYSTEM, user_prompt=user, model=model, allowed_tools=ALL_TOOLS
    )
    data = parse_json(raw)
    if not isinstance(data, list):
        raise ValueError("Generation did not return a JSON array")
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("title") and item.get("text"):
            out.append(
                {"title": str(item["title"]), "text": str(item["text"]), "spec": item.get("spec")}
            )
    return out


async def reflect(*, hypothesis_text: str, goal: str, model: str) -> Dict[str, Any]:
    user = (
        f"Research goal:\n{goal}\n\n"
        f"Hypothesis to review:\n{hypothesis_text}\n\n"
        "Review it now."
    )
    raw = await run_role(
        system_prompt=REFLECTION_SYSTEM, user_prompt=user, model=model, allowed_tools=ALL_TOOLS
    )
    data = parse_json(raw)
    nov = str(data.get("novelty_review", "MEDIUM")).upper()
    feas = str(data.get("feasibility_review", "MEDIUM")).upper()
    refs = data.get("references", [])
    return {
        "novelty_review": nov if nov in ("HIGH", "MEDIUM", "LOW") else "MEDIUM",
        "feasibility_review": feas if feas in ("HIGH", "MEDIUM", "LOW") else "MEDIUM",
        "comment": str(data.get("comment", "")),
        "references": refs if isinstance(refs, list) else [],
    }


async def debate(*, goal: str, hypo_a: Dict[str, Any], hypo_b: Dict[str, Any], model: str) -> str:
    """Return 'A' or 'B' -- the winner of the pairwise scientific debate."""
    def block(label: str, h: Dict[str, Any]) -> str:
        return (
            f"Hypothesis {label}: {h.get('title')}\n"
            f"{h.get('text')}\n"
            f"novelty={h.get('novelty_review')} feasibility={h.get('feasibility_review')}\n"
            f"{_verification_brief(h)}"
        )

    user = (
        f"Research goal:\n{goal}\n\n"
        f"{block('A', hypo_a)}\n\n{block('B', hypo_b)}\n\n"
        "Debate and decide the winner now."
    )
    raw = await run_role(
        system_prompt=RANKING_SYSTEM, user_prompt=user, model=model, allowed_tools=ALL_TOOLS, max_turns=6
    )
    winner = str(parse_json(raw).get("winner", "A")).strip().upper()
    return "B" if winner == "B" else "A"


async def evolve(*, goal: str, parents: List[Dict[str, Any]], model: str) -> Optional[Dict[str, Any]]:
    parent_text = "\n\n".join(
        f"Parent {i + 1}: {p.get('title')}\n{p.get('text')}" for i, p in enumerate(parents)
    )
    user = (
        f"Research goal:\n{goal}\n\n"
        f"Top-ranked parent hypotheses to synthesize:\n{parent_text}\n\n"
        f"Machine-verifiable hypothesis kinds (include a matching JSON `spec` if the synthesized "
        f"hypothesis fits one; otherwise null):\n{describe_verifiers()}\n\n"
        "Produce one improved hypothesis now."
    )
    raw = await run_role(
        system_prompt=EVOLUTION_SYSTEM, user_prompt=user, model=model, allowed_tools=ALL_TOOLS
    )
    data = parse_json(raw)
    if isinstance(data, dict) and data.get("title") and data.get("text"):
        return {"title": str(data["title"]), "text": str(data["text"]), "spec": data.get("spec")}
    return None


async def meta_review(*, goal: str, ranked: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    lines = []
    for i, h in enumerate(ranked, 1):
        lines.append(
            f"{i}. [{h.get('elo_score', 0):.0f}] {h.get('title')} "
            f"(novelty={h.get('novelty_review')}, feasibility={h.get('feasibility_review')}; "
            f"{_verification_brief(h)})"
        )
    user = (
        f"Research goal:\n{goal}\n\n"
        f"Ranked hypotheses (best first):\n" + "\n".join(lines) + "\n\n"
        "Write the research overview now."
    )
    raw = await run_role(
        system_prompt=META_REVIEW_SYSTEM, user_prompt=user, model=model, allowed_tools=[]
    )
    data = parse_json(raw)
    crit = data.get("critique", [])
    steps = data.get("suggested_next_steps", [])
    return {
        "summary": str(data.get("summary", "")),
        "critique": crit if isinstance(crit, list) else [str(crit)],
        "suggested_next_steps": steps if isinstance(steps, list) else [str(steps)],
    }
