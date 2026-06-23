"""In-process MCP tools exposed to every role in the Claude Agent SDK workflow.

Two tools on one ``coscientist`` server:
  - ``arxiv_search``  -> grounds hypotheses in real cond-mat literature and lets
                          Reflection confirm that cited arXiv IDs actually resolve.
  - ``verify_hypothesis`` -> runs the (currently stubbed) automated physics check on a
                          structured hypothesis spec.

Both wrap synchronous code in ``asyncio.to_thread`` so the SDK event loop is never
blocked by a network call or (later) a numerical band-structure computation.
"""
import asyncio
import json
from typing import Any, Dict

from claude_agent_sdk import create_sdk_mcp_server, tool

from ..tools.arxiv_search import ArxivSearchTool
from ..verification import run_verification

SERVER_NAME = "coscientist"
MAX_ARXIV_RESULTS = 8

# Condensed-matter subcategories the literature search is scoped to. Broad enough to
# cover the flat-band / topology / strongly-correlated space the goals target.
CONDMAT_CATEGORIES = [
    "cond-mat.str-el",    # strongly correlated electrons
    "cond-mat.mes-hall",  # mesoscale & nanoscale (topology, edge states)
    "cond-mat.supr-con",  # superconductivity
    "cond-mat.mtrl-sci",  # materials science
    "cond-mat.quant-gas", # cold atoms / optical lattices
    "cond-mat.dis-nn",    # disordered systems & neural networks
]

# Reuse one arXiv client across calls (it holds an arxiv.Client()).
_arxiv = ArxivSearchTool()


@tool(
    "arxiv_search",
    "Search arXiv (cond-mat categories) for papers relevant to a query. Returns up to "
    f"{MAX_ARXIV_RESULTS} results as JSON with title, authors, abstract, arxiv_id, url, "
    "and published date. Use it to seed hypotheses from real recent work and to confirm "
    "that any arXiv ID you cite actually exists. Pass a focused topical query string.",
    {"query": str},
)
async def arxiv_search(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args["query"]
    try:
        papers = await asyncio.to_thread(
            _arxiv.search_papers,
            query,
            max_results=MAX_ARXIV_RESULTS,
            categories=CONDMAT_CATEGORIES,
        )
    except Exception as exc:  # keep the agent loop alive; report failure as data
        return {
            "content": [{"type": "text", "text": f"arXiv search failed: {exc}"}],
            "is_error": True,
        }
    # Trim each record to the fields agents need (keeps token usage down).
    slim = [
        {
            "title": p.get("title"),
            "authors": p.get("authors", [])[:8],
            "abstract": p.get("abstract"),
            "arxiv_id": p.get("arxiv_id"),
            "url": p.get("arxiv_url"),
            "published": p.get("published"),
        }
        for p in papers
    ]
    payload = {"query": query, "count": len(slim), "results": slim}
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}


@tool(
    "verify_hypothesis",
    "Run an automated condensed-matter physics check on a STRUCTURED hypothesis spec. "
    "The argument 'spec_json' must be a JSON object string with a 'kind' field (e.g. "
    "'tight_binding') plus kind-specific fields. Returns JSON evidence: "
    "{kind, checkable, claim_supported, evidence, reason}. A result with "
    "checkable=false means no verifier exists for that kind yet -- this is expected and "
    "is NOT an error; just note the hypothesis is not yet machine-checkable.",
    {"spec_json": str},
)
async def verify_hypothesis(args: Dict[str, Any]) -> Dict[str, Any]:
    raw = args["spec_json"]
    try:
        spec = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return {
            "content": [{"type": "text", "text": f"spec_json was not valid JSON: {exc}"}],
            "is_error": True,
        }
    # run_verification is sync (and may do real numerics later) -> offload.
    result = await asyncio.to_thread(run_verification, spec)
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}


# In-process MCP server bundling both tools.
coscientist_server = create_sdk_mcp_server(
    name=SERVER_NAME,
    version="0.1.0",
    tools=[arxiv_search, verify_hypothesis],
)

# Fully-qualified tool names for ClaudeAgentOptions.allowed_tools (mcp__{server}__{tool}).
ARXIV_TOOL = f"mcp__{SERVER_NAME}__arxiv_search"
VERIFY_TOOL = f"mcp__{SERVER_NAME}__verify_hypothesis"
ALL_TOOLS = [ARXIV_TOOL, VERIFY_TOOL]
