"""Reusable research goals for the condensed-matter Co-Scientist workflow."""


C2_FLATNESS_SEARCH_GOAL = """
Treat this as a constrained tight-binding model search and falsification task, not as
open-ended scientific brainstorming.

Research question:
Among translationally invariant two-dimensional tight-binding Hamiltonians with two to
four orbitals per unit cell and finite-range hopping, what is the best isolated lowest
band with Chern number C = +2 or C = -2 that can be obtained as hopping range is enlarged
from nearest-neighbor (NN), to NN+next-nearest-neighbor (NNN), to one additional harmonic
shell? In particular, determine whether any model improves materially on the known
two-band square-lattice C=2 baseline while remaining sparse and experimentally plausible.

Mandatory conventions and constraints:
1. Fix the energy scale: at least one hopping amplitude must have magnitude exactly 1.0,
   no hopping amplitude may have magnitude greater than 1.0, and every onsite energy must
   lie in [-2.0, 2.0]. Do not obtain a small bandwidth by uniformly rescaling the entire
   Hamiltonian.
2. Use 2-4 orbitals per unit cell and at most 64 distinct hopping entries. List every bond
   once; the verifier adds its Hermitian conjugate automatically.
3. The target must be the lowest band (band_index = 0), so its isolation can be checked at
   filling = 1 / n_orb.
4. Use kmesh = 64. Every returned candidate must include a complete, non-null
   kind="tight_binding" spec compatible with the supplied verifier. A proposal without a
   complete hoppings list is not a candidate and must not be returned.
5. Every spec must request all three checks:
     - chern_number for band 0 with expected +2 or -2;
     - flat_band for band 0 with max_bandwidth = 0.10;
     - gapped_at_filling at filling = 1/n_orb with min_gap = 1.0.
   These thresholds correspond to a target direct-gap-to-bandwidth ratio of at least 10
   under the fixed energy convention.
6. Before returning a candidate, call verify_hypothesis on its exact spec. Report the
   numerical bandwidth, direct gap, Chern number, and gap/bandwidth ratio in the hypothesis
   text. Do not describe an unverified parameter sweep as a positive result.
7. If a promising construction fails either numerical threshold, report it only as a
   clearly labelled negative result with the failed verifier evidence; do not weaken the
   claim thresholds to make it pass.
8. Treat these as mandatory prior-art baselines, not novel proposals: the two-band
   square-lattice C=0,+/-1,+/-2 family of Grushin et al. (arXiv:1207.4097), the arbitrary-C
   constructions of Sticlet et al. (arXiv:1201.6613) and Yang et al.
   (arXiv:1205.5792), and the C=2 dice-lattice construction of Wang and Ran
   (arXiv:1109.3435). Use arxiv_search to check for equivalent or newer models before
   claiming novelty.
9. Do not infer Chern-number addition from orbital angular momentum, layer count, or band
   hybridization. The C = +/-2 claim must come from the verifier applied to an isolated
   band.
10. Respect the finite-range flat-Chern-band no-go theorem (arXiv:1311.4956): seek a
    nearly flat band or a defensible finite-range limitation, never an exactly flat
    nonzero-Chern band.

Generate candidates that deliberately cover different mechanisms rather than cosmetic
variants: (a) the complete symmetry-allowed two-band square-lattice Fourier family,
(b) a concrete four-orbital decorated-Lieb construction, and (c) one distinct
multi-orbital or bilayer construction not equivalent to established dice, triangular, or
bilayer-checkerboard models. Prefer a verified negative result over an attractive but
underspecified story. A scientifically useful outcome may be that no candidate crosses
the stated threshold within a range class, but that conclusion must be based on concrete
failed specs and verifier evidence.
""".strip()

