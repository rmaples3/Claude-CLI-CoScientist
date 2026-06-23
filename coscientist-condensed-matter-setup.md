# Building a Condensed-Matter Co-Scientist: Setup & Research Plan

*A practical guide to standing up a Co-Scientist–style multi-agent hypothesis engine on WashU RIS, with a verification-grounded angle aimed at a provable, publishable contribution in condensed matter.*

---

## 0\. The one idea that should drive every decision

The Co-Scientist paper (Gottweis et al., *Nature* 2026; preprint arXiv:2502.18864) is **model-agnostic scaffolding**. Reimplementing its six agents is no longer novel — there are already several open clones. The architecture is a commodity. Two things are *not* commodities and are where your REU contribution lives:

1. **Verification grounding.** The paper's hypotheses are validated by *wet-lab* biology, after the fact, by humans. In condensed matter you can do something the original cannot: **close the loop in-silico**. A hypothesis like "this lattice/Hamiltonian hosts a flat band" or "this material is a candidate topological insulator" can be *checked automatically* with a tight-binding or DFT computation before it's ever ranked. A co-scientist that filters and scores its own hypotheses against real physics computations is a genuine step beyond the paper.  
2. **Evaluation.** There is no good automated metric for "is this hypothesis novel *and* correct." Elo only measures relative preference between two LLM outputs. A defensible, reproducible evaluation protocol for condensed-matter hypotheses is itself a publishable result.

Everything below is organized so that you reach a working loop fast, then spend your real effort on (1) and (2).

**Strategic priorities, in order:** fork don't rebuild → get a working loop on Claude/API first → port to local vLLM on RIS for unattended runs → add condensed-matter verification tools → build an honest evaluation → write it up.

---

## 1\. Don't rebuild — fork

Pick one of these as your base. They all already implement the agent roster, the Elo tournament, and the async loop, and they all speak the **OpenAI API format** (this is what lets you swap models freely):

| Repo | Stack | Why pick it |
| :---- | :---- | :---- |
| [conradry/open-coscientist-agents](https://github.com/conradry/open-coscientist-agents) | LangGraph \+ GPT Researcher, Streamlit tournament viewer | Most complete; best if you want visualization out of the box |
| [llnl/open-ai-co-scientist](https://github.com/llnl/open-ai-co-scientist) | Gradio UI, clean modular agents | National-lab code, easiest to read and extend |
| [Kaimen-Inc/Co-Scientist](https://github.com/Kaimen-Inc/Co-Scientist) | Faithful to paper's prompts/roster | Best if you want to stay close to the published method |

**Recommendation:** start from `llnl/open-ai-co-scientist` (cleanest to extend with custom tools) or `conradry/open-coscientist-agents` (best tournament tooling). Read the Generation, Reflection, Ranking, Evolution, and Meta-review agents in whichever you pick — you'll be modifying Reflection (to call your verifier) and Ranking (to weight by verification results).

The six roles you're inheriting:

- **Generation** — proposes hypotheses from the research goal \+ literature.  
- **Reflection** — critiques/fact-checks; *this is where you'll bolt on physics verification*.  
- **Ranking** — Elo tournament via pairwise "scientific debate."  
- **Evolution** — mutates/combines top hypotheses.  
- **Proximity** — clusters similar hypotheses to keep diversity.  
- **Meta-review** — synthesizes patterns across the tournament into a research overview.

---

## 2\. The hybrid model strategy (read this before buying anything)

The paper is explicit that hypothesis quality scales with (a) the frontier model and (b) test-time compute (the tournament makes *many* calls per run). Two consequences:

- A 30B local model **will** produce visibly weaker hypotheses than a frontier model. Don't expect Gemini-3-quality science from it.  
- The tournament burns tokens. A single serious run is hundreds to thousands of LLM calls.

So split the work by what each backend is good at:

| Phase | Backend | Why |
| :---- | :---- | :---- |
| Build & debug the loop | **Claude / API** (or local model) | Get it working with the least friction |
| Unattended bulk runs, parameter sweeps, dev iteration | **Local vLLM on RIS** | Free-running, unlimited iteration, no token anxiety |
| Final/"real" hypothesis generation for the writeup | **Frontier API** (Gemini 3 / GPT-5.x / Opus 4.x), REU-funded | This is where the actual scientific value appears |

Because the system is model-agnostic, you build **once** and change a base-URL/model config flag. The local backend is mostly for cheap iteration and for the "it runs autonomously on our cluster" story; the headline results should come from a frontier model.

---

## 3\. Phase 1 — Working loop in days, not weeks (Claude-agent prototype)

Goal: prove the generate→debate→evolve loop end-to-end on a condensed-matter goal before touching RIS.

1. Fork your chosen repo, set up a Python env, point it at an API model.  
2. Write **one good research goal** in the paper's style. Example to start:  
     
   "Propose tight-binding lattice models (≤4 sites/cell, nearest- and next-nearest-neighbor hopping) that host a topologically nontrivial flat band at a tunable filling, and that could plausibly be realized in a known 2D material or optical-lattice platform. Prioritize models whose nontriviality can be checked numerically." Note how the goal is written to be **machine-verifiable** — that's deliberate and is your wedge.  
     
3. Run the tournament. Confirm you get a ranked research overview with citations.  
4. Inspect failure modes: hallucinated references, vague hypotheses, models that aren't actually checkable. These motivate Phases 3–4.

Optionally do this as a **Claude Agent SDK** workflow (subagents for each role, arXiv as an MCP tool). That's the fastest path to a working loop and gives strong reasoning for free; you then port the proven design to the local backend.

---

## 4\. Phase 2 — Local model on WashU RIS

RIS Compute1 \= WashU VPN → SSH → **IBM Spectrum LSF** scheduler → **Docker** containers, GPUs available, 5 TB free storage per faculty sponsor. The image **must live in a public Docker Hub repo** to be usable by RIS.

### 4.1 Connect

\# On WashU VPN

ssh \<wustl-key\>@compute1-client-1.ris.wustl.edu

### 4.2 Build a vLLM image and push it (public repo required)

`Dockerfile`:

FROM vllm/vllm-openai:latest

\# vLLM already exposes an OpenAI-compatible server at /v1.

\# Add anything else your agents need to call locally:

RUN pip install \--no-cache-dir arxiv pymatgen mp-api pythtb kwant sympy

docker build \-t \<dockerhub-user\>/coscientist-vllm:latest .

docker push \<dockerhub-user\>/coscientist-vllm:latest   \# repo must be PUBLIC

### 4.3 Serve a model as a GPU job

RIS GPU nodes are A100-class (\~80 GB). Sweet spot for a single user: a **30–32B reasoning model** at FP16 on 1–2 GPUs (Qwen3-32B / Qwen3-30B-A3B, a DeepSeek-R1 distill, or Gemma-3-27B), or a 70B at Q4.

**Confirm with current RIS docs:** your compute group (`-G compute-<group>`), the correct GPU queue, and the exact `gmodel` string. RIS exposes container ports via `LSF_DOCKER_PORTS` and mounts via `LSF_DOCKER_VOLUMES`. The command below is the shape; fill in your group/queue/GPU model.

Interactive (good for first bring-up):

export LSF\_DOCKER\_PORTS='8000:8000'

export LSF\_DOCKER\_VOLUMES="$HOME:$HOME /scratch1/fs1/\<group\>:/scratch1/fs1/\<group\>"

bsub \-Is \-q general-interactive \\

     \-G compute-\<group\> \\

     \-gpu "num=2" \\

     \-R 'rusage\[mem=64GB\] span\[hosts=1\]' \\

     \-a 'docker(\<dockerhub-user\>/coscientist-vllm:latest)' \\

     vllm serve Qwen/Qwen3-32B \\

        \--host 0.0.0.0 \--port 8000 \\

        \--tensor-parallel-size 2 \\

        \--max-model-len 32768 \\

        \--download-dir /scratch1/fs1/\<group\>/hf-cache

Then from the client node, tunnel to the exec node so your agents can reach the endpoint:

ssh \-N \-L 8000:\<exec-node-hostname\>:8000 \<wustl-key\>@compute1-client-1.ris.wustl.edu

\# Agents now hit http://localhost:8000/v1

(`bjobs -l <jobid>` shows the exec host; `bkill <jobid>` stops it.) For unattended runs, submit the same thing without `-Is` and add `-o serve.log`.

### 4.4 Point the agents at it

In the forked repo's config, set the OpenAI base URL to `http://localhost:8000/v1`, model to `Qwen/Qwen3-32B`, and any dummy API key vLLM accepts. No agent-code changes needed — that's the payoff of the OpenAI-compatible interface.

**Tip:** cache HF weights once to your 5 TB / scratch space (`--download-dir`) so jobs don't re-download multi-GB checkpoints each launch.

---

## 5\. Phase 3 — arXiv literature grounding

This is the single most important quality lever. In the paper, giving the Reflection agent real literature search is what suppressed "novel but implausible" hallucinations. For condensed matter, arXiv (cond-mat) is your corpus.

Layered access:

- **Live search / metadata:** the `arxiv` Python package against the arXiv API (Atom XML). Keep to \~1 request / 3 s.  
- **Bulk corpus (optional, for offline RAG):** arXiv PDFs \+ LaTeX source on Amazon S3 (requester-pays), or the Kaggle metadata dump. OAI-PMH gives metadata only.

Minimal retrieval tool the agents can call:

import arxiv

def search\_condmat(query: str, k: int \= 8):

    search \= arxiv.Search(

        query=f"cat:cond-mat.\* AND ({query})",

        max\_results=k,

        sort\_by=arxiv.SortCriterion.Relevance,

    )

    out \= \[\]

    for r in arxiv.Client().results(search):

        out.append({

            "title": r.title,

            "authors": \[a.name for a in r.authors\],

            "abstract": r.summary,

            "arxiv\_id": r.get\_short\_id(),

            "url": r.pdf\_url,

            "published": r.published.isoformat(),

        })

    return out

Wire it in two places: (1) **Generation** seeds hypotheses from real recent cond-mat work; (2) **Reflection** verifies that claimed prior art actually exists and that the hypothesis isn't already published. For stronger grounding, embed abstracts/full text into FAISS or Chroma and do RAG so the agents can cite specific passages — this also sets up the "provenance" improvement the paper itself names as future work.

---

## 6\. Phase 4 — The condensed-matter verification layer (your contribution)

This is the part that makes it *physics* and not just a literature chatbot. Add a **Verifier agent/tool** that the pipeline runs on every candidate hypothesis that is computationally checkable. Hypotheses that pass get an evidence boost in Ranking; those that provably fail get filtered or sent back to Evolution.

Concrete, lightweight tools (all pip-installable, all CPU-friendly enough to run on RIS or even locally):

- **Model Hamiltonians / topology — [PythTB](https://www.physics.rutgers.edu/pythtb/) and [Kwant](https://kwant-project.org/).** Build the tight-binding model the Generation agent proposes, then compute band structure, gaps, Berry phase / Chern number, edge states. This *directly* checks claims like "flat band," "nontrivial topology," "gap closing at this filling."  
- **Materials data — [pymatgen](https://github.com/materialsproject/pymatgen) \+ the [Materials Project API](https://next-gen.materialsproject.org/api) (`mp-api`).** Look up whether a proposed material exists, its formation energy, band structure, symmetry, magnetic ordering. Grounds "this could be realized in material X" against a real database. (pymatgen is actively maintained — 2026 releases.)  
- **Symbolic checks — SymPy.** Verify algebra in proposed effective Hamiltonians, symmetry constraints, dimensional consistency.  
- **(Stretch) DFT — a single-point check** via an interface to a DFT engine for the most promising candidates. Expensive; reserve for the final shortlist and run as its own `bsub` job.

Verifier sketch (the interface that matters; fill in per hypothesis type):

def verify\_tight\_binding(spec: dict) \-\> dict:

    """spec is a structured hypothesis: lattice vectors, orbitals, hoppings, claim.

    Returns evidence the Ranking agent can use."""

    import numpy as np

    from pythtb import tb\_model

    model \= tb\_model(spec\["dim\_k"\], spec\["dim\_r"\],

                     spec\["lat"\], spec\["orb"\])

    for h in spec\["hoppings"\]:

        model.set\_hop(h\["amp"\], h\["i"\], h\["j"\], h\["R"\])

    k \= spec.get("kmesh")  \# path or grid

    evals \= model.solve\_all(k)

    bandwidths \= evals.max(axis=1) \- evals.min(axis=1)

    return {

        "checkable": True,

        "min\_bandwidth": float(bandwidths.min()),   \# \~0 \=\> flat band claim supported

        "gaps": \_direct\_gaps(evals),

        "claim\_supported": bandwidths.min() \< spec\["flat\_tol"\],

        \# add Chern number via Berry-phase plaquette sum for topology claims

    }

Then modify **Reflection** to attach this evidence to each hypothesis, and **Ranking** to use it (e.g., a verified hypothesis wins ties / gets a prior boost in the Elo debate prompt). This is the loop the original paper *couldn't* close automatically.

---

## 7\. Where the provable, novel contribution is

Materials-science agent work already exists (MatAgent, AtomAgents, MAPPS) — mostly alloys/molecules/property prediction with LAMMPS or ML potentials. The **electronic-structure / model-Hamiltonian / topological side of condensed matter is comparatively underexplored** by these agent systems, and the original Co-Scientist barely touched physics at all. Defensible angles, roughly in increasing ambition:

1. **Verification-in-the-loop Co-Scientist for tight-binding/topological model discovery.** The novelty is the *automated in-silico verifier* gating the tournament — demonstrate that verification-weighted ranking produces more correct hypotheses than the vanilla LLM-only tournament. This is a clean, measurable claim.  
2. **An honest evaluation protocol & benchmark for condensed-matter hypotheses.** Curate a set of goals where the answer is computationally checkable, and report verified-correctness, novelty (not-already-on-arXiv), and a human-expert score from your PI/postdocs. Compare frontier vs. local model, and with/without verification. The field lacks good metrics here — this alone can be a paper.  
3. **Rediscovery test (the paper's strongest validation move).** Take a recent cond-mat result your lab knows well, give the system only pre-publication knowledge, and see if it proposes the same mechanism/model — mirroring the paper's cf-PICI "recapitulation" experiment. A successful rediscovery in condensed matter is a compelling, concrete result.

The cleanest publishable thesis: **"Automated physics verification, used as a tournament signal, measurably improves the correctness of LLM-generated condensed-matter hypotheses."** It's narrow, it's testable, and it's something the original system architecturally cannot do.

---

## 8\. Pitfalls to plan around

- **Token burn.** Tournaments are call-heavy. Cap rounds/population while developing; only scale compute for final runs. Track API spend against your REU budget.  
- **Hallucinated citations.** Don't trust any reference the LLM emits — have Reflection confirm every arXiv ID actually resolves (the `arxiv` tool above does this).  
- **"Checkable" is doing a lot of work.** Many interesting condensed-matter hypotheses are *not* cheaply verifiable. Scope your research goals toward the checkable subset first; that's a feature, not a limitation, for a first paper.  
- **Local-model ceiling.** If local outputs feel weak, that's expected — use local for iteration, frontier API for results. Don't over-invest in squeezing a 30B model.  
- **RIS specifics drift.** Confirm queue names, GPU `gmodel` strings, and your compute group against current RIS docs before submitting; the `bsub` above is the correct *shape* but those fields are site/account-specific.  
- **Novelty ≠ correctness.** Keep them as separate axes in evaluation. The whole point of the verifier is to catch novel-but-wrong.

---

## 9\. Suggested REU timeline

| Weeks | Milestone |
| :---- | :---- |
| 1 | Fork a repo; run the loop on an API model with one cond-mat goal; read the agent code. Get RIS access \+ VPN working. |
| 2 | Stand up vLLM on RIS (Phase 2); reproduce the loop against the local endpoint. Build the arXiv tool (Phase 3). |
| 3 | Implement the PythTB/Kwant \+ Materials Project verifier (Phase 4); wire it into Reflection \+ Ranking. |
| 4–5 | Define the evaluation protocol and a small benchmark of checkable goals; run with/without verification, local vs. frontier. |
| 6–7 | Rediscovery experiment on a known lab result; collect expert scores from your PI/postdocs. |
| 8+ | Analysis, ablations, writeup. The verification-vs-vanilla comparison is your headline figure. |

---

## 10\. References

- Gottweis et al., *Accelerating scientific discovery with Co-Scientist*, Nature 2026 (preprint: [arXiv:2502.18864](https://arxiv.org/abs/2502.18864)); [DeepMind blog](https://deepmind.google/blog/co-scientist-a-multi-agent-ai-partner-to-accelerate-research/)  
- Open reimplementations: [conradry/open-coscientist-agents](https://github.com/conradry/open-coscientist-agents), [llnl/open-ai-co-scientist](https://github.com/llnl/open-ai-co-scientist), [Kaimen-Inc/Co-Scientist](https://github.com/Kaimen-Inc/Co-Scientist)  
- WashU RIS: [Docker \+ Compute Service](https://washu.atlassian.net/wiki/spaces/ITKB/pages/184255610/Docker+and+the+RIS+Compute+Service), [Scientific Compute Platform](https://ris.wustl.edu/systems/scientific-compute-platform/), [community RIS guide](https://saumikn.com/blog/washu-ris-guide)  
- Models / serving: [open-weight models 2026](https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally), [self-hosted leaderboard](https://onyx.app/self-hosted-llm-leaderboard), [vLLM](https://docs.vllm.ai)  
- arXiv access: [bulk data](https://info.arxiv.org/help/bulk_data.html), [S3](https://info.arxiv.org/help/bulk_data_s3.html), [`arxiv` package](https://pypi.org/project/arxiv/)  
- Condensed-matter tooling: [pymatgen](https://github.com/materialsproject/pymatgen), [Materials Project API](https://next-gen.materialsproject.org/api), [PythTB](https://www.physics.rutgers.edu/pythtb/), [Kwant](https://kwant-project.org/)  
- Prior agent work in materials: [MatAgent](https://github.com/adibgpt/MatAgent), [goal-driven hypothesis generation (arXiv:2501.13299)](https://arxiv.org/pdf/2501.13299), [LLM scientific agents survey (arXiv:2503.24047)](https://arxiv.org/pdf/2503.24047)

