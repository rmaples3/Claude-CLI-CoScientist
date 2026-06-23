# Verification layer

The verification layer is what makes this a *physics* Co-Scientist and not just a literature
chatbot: it runs an **automated, in-silico check** on a hypothesis's central claim and feeds
the verdict into the Elo tournament (a checkably-correct hypothesis beats an equally-reviewed
but unverified rival — see `app/tournament.py`). It is **pluggable**: condensed-matter
verifiers ship built in, and other domains (e.g. biophysics) can be added as separate
pip-installed packages without touching this repo.

## The contract

A *verifier* maps a structured hypothesis `spec` (a dict with a `"kind"` discriminator) to a
**VerificationResult** dict with a fixed shape (`app/verification/base.py`):

```
kind, checkable (bool), claim_supported (bool|None), evidence (dict), reason (str), error (str|None)
```

`run_verification(spec)` (in `app/verification/registry.py`) dispatches on `spec["kind"]`,
never raises, and falls back to a "not-yet-checkable" stub for unregistered kinds. The
Generation/Evolution agents are shown `describe_verifiers()` so they emit specs that match a
registered verifier; the orchestrator computes the verdict deterministically from the spec
(not parsed from model text) so the experiment stays reproducible.

## Built-in: `tight_binding` (condensed matter)

Pure-numpy tight-binding verifier (`app/verification/condensed_matter/tight_binding.py`).
Builds the Bloch Hamiltonian H(k) from hoppings, diagonalizes on a BZ grid, and checks:

| Claim | Meaning | Method |
|---|---|---|
| `flat_band {band_index, max_bandwidth}` | band is (nearly) dispersionless | max−min of the band over the grid |
| `gapped_at_filling {filling, min_gap}` | insulating at a given filling | min direct gap between filled/empty bands |
| `chern_number {band_index, expected?}` | band topology (2D) | Fukui–Hatsugai–Suzuki lattice method |

**Spec format** (each bond listed once; the Hermitian conjugate is added automatically;
complex amplitudes as `[re, im]`; `R` is the integer cell translation; phase `exp(i 2π k·R)`,
k reduced; bands indexed 0 = lowest):

```json
{
  "kind": "tight_binding",
  "dim_k": 2,
  "onsite": [0, 0, 0],
  "hoppings": [
    {"amp": -1.0, "i": 0, "j": 1, "R": [0, 0]},
    {"amp": -1.0, "i": 0, "j": 1, "R": [-1, 0]},
    {"amp": -1.0, "i": 0, "j": 2, "R": [0, 0]},
    {"amp": -1.0, "i": 0, "j": 2, "R": [0, -1]}
  ],
  "claims": {"flat_band": {"band_index": 1, "max_bandwidth": 0.01}}
}
```

This is the **Lieb lattice** — its middle band is exactly flat at E=0. The verifier confirms
`bandwidth ≈ 0`. The test suite (`tests/test_tight_binding.py`) also pins the **QWZ model**
(Chern ±1 for |m|<2, 0 for |m|>2) and a dispersive negative control.

## Kinds of questions you can pose now

The `tight_binding` verifier makes these research goals *machine-gradeable* — i.e. the
tournament can reward correctness, not just plausibility:

- **Flat bands:** "Find a ≤4-orbital 2D lattice with a perfectly flat band" (Lieb, kagome,
  checkerboard, dice). → `flat_band`.
- **Nearly-flat Chern bands** (the fractional-Chern-insulator wedge): "Find a 2-band model
  with a Chern band whose bandwidth-to-gap ratio is small." → `flat_band` + `chern_number`.
- **Topological insulators / Chern insulators:** "Find a model gapped at half filling with a
  nonzero Chern number." → `gapped_at_filling` + `chern_number`.
- **Gap / topology engineering:** "What on-site or NNN term opens a gap / drives a band-
  inversion in this lattice?" → `gapped_at_filling`, compare `chern_number` across parameters.
- **Rediscovery:** give the system pre-publication knowledge and see if it reconstructs a
  known model (e.g. Haldane, QWZ) whose invariant the verifier then confirms.

**Not yet checkable** (good candidates for future verifiers): interaction-driven phases,
real-material realizability (DFT / Materials Project), symmetry/algebra (SymPy), transport
(Kwant). These slot in as new `kind`s without changing the loop.

## Adding a verifier — built-in or external plugin

A verifier is any object with `.kind`, `.description`, `.run(spec)->result`, and
`.spec_example` (use the `Verifier` dataclass). Two ways to register:

**1. Built-in** (in this repo): add a module under `app/verification/<domain>/`, then register
in `app/verification/registry.py::_load_builtins` (or extend a domain's `get_verifiers()`).

**2. External package** (the "separate biophysics package" path): publish a pip package that
advertises a `coscientist.verifiers` entry point returning a list of `Verifier`s. Once
`pip install`ed alongside this repo, its kinds auto-register — no core change.

```toml
# pyproject.toml of e.g. coscientist-biophysics
[project.entry-points."coscientist.verifiers"]
biophysics = "coscientist_biophysics:get_verifiers"
```

```python
# coscientist_biophysics/__init__.py
from app.verification import Verifier            # the shared contract
from .folding import verify_protein_stability

def get_verifiers():
    return [Verifier(
        kind="protein_stability",
        description="Estimate folding free energy / stability of a proposed mutant; "
                    "spec: {kind:'protein_stability', sequence, mutations, claim:{ddG_max}}.",
        run=verify_protein_stability,
        spec_example={"kind": "protein_stability", "sequence": "...", "mutations": ["A42G"],
                      "claim": {"ddG_max": -1.0}},
    )]
```

The result `dict` must follow the contract above (use `verdict()` / `unverifiable()` /
`error()` from `app.verification` to build it). That's the entire integration surface — the
tournament, roles, and loop consume every verifier identically.
