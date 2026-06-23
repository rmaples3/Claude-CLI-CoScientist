"""Built-in condensed-matter verifiers.

Exposes ``get_verifiers()`` returning the Verifier objects this package provides. The
registry calls it to register the built-ins; an external plugin package mirrors this exact
shape (a zero-arg factory returning a list of Verifier) behind a ``coscientist.verifiers``
entry point.
"""
from typing import List

from ..base import Verifier
from .tight_binding import KIND as TB_KIND
from .tight_binding import verify_tight_binding

_TB_DESCRIPTION = (
    "Periodic tight-binding model check (pure numpy). Provide 'dim_k' (1 or 2), 'onsite' "
    "energies (length = number of orbitals), and 'hoppings' -- each bond listed ONCE as "
    "{amp, i, j, R}; the Hermitian conjugate is added automatically; complex amplitudes as "
    "[re, im]; R is the integer cell translation; the Bloch phase is exp(i 2*pi*k.R) with k "
    "in reduced coordinates. Supported 'claims': flat_band {band_index, max_bandwidth}; "
    "gapped_at_filling {filling, min_gap} (filling is the fraction of bands occupied); "
    "chern_number {band_index, expected} (2D only, via Fukui-Hatsugai-Suzuki). Bands are "
    "indexed 0 = lowest. Use this whenever a hypothesis names a concrete lattice + hoppings "
    "and a claim about flatness, gaps, or topology."
)

# Lieb lattice: a 3-band square-lattice model with an exact flat middle band at E=0.
_TB_EXAMPLE = {
    "kind": "tight_binding",
    "dim_k": 2,
    "lattice_vectors": [[1, 0], [0, 1]],
    "orbitals": [[0, 0], [0.5, 0], [0, 0.5]],
    "onsite": [0, 0, 0],
    "hoppings": [
        {"amp": -1.0, "i": 0, "j": 1, "R": [0, 0]},
        {"amp": -1.0, "i": 0, "j": 1, "R": [-1, 0]},
        {"amp": -1.0, "i": 0, "j": 2, "R": [0, 0]},
        {"amp": -1.0, "i": 0, "j": 2, "R": [0, -1]},
    ],
    "claims": {"flat_band": {"band_index": 1, "max_bandwidth": 0.01}},
}


def get_verifiers() -> List[Verifier]:
    return [
        Verifier(
            kind=TB_KIND,
            description=_TB_DESCRIPTION,
            run=verify_tight_binding,
            spec_example=_TB_EXAMPLE,
        )
    ]
