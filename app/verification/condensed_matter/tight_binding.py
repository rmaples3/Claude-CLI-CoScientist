"""Tight-binding verifier (condensed matter): band structure, flat bands, gaps, Chern number.

Pure numpy. Builds the Bloch Hamiltonian H(k) for a periodic tight-binding model from a
structured spec, diagonalizes it on a Brillouin-zone grid, and checks the requested claims.
No external physics package is required; numpy is imported lazily so a missing install
degrades to a graceful error rather than breaking the whole verification import.

Spec (kind="tight_binding"):
  dim_k:    1 or 2 (k-space dimension; Chern requires 2)
  onsite:   list of on-site energies, length = number of orbitals (defaults to zeros)
  orbitals: optional fractional positions (used only to infer orbital count if onsite absent)
  hoppings: list of {amp, i, j, R}. List each bond ONCE -- the Hermitian conjugate is added
            automatically. amp may be a real number, [re, im], or {"re":, "im":}. R is the
            integer cell translation (length dim_k). Bloch phase = exp(i 2*pi*k.R), k reduced.
  kmesh:    optional BZ grid size per dimension (default 24)
  claims:   any of
              flat_band:         {band_index, max_bandwidth}
              gapped_at_filling: {filling, min_gap}   # filling = fraction of bands occupied
              chern_number:      {band_index, expected?}  # 2D only
Bands are indexed 0 = lowest.
"""
import logging
from typing import Any, Dict, List

from ..base import error, unverifiable

logger = logging.getLogger("coscientist.verification")

try:
    import numpy as np
except ImportError:  # numpy is an optional runtime dep of this verifier
    np = None

KIND = "tight_binding"
DEFAULT_KMESH = 24
MAX_ORBITALS = 16
MAX_HOPPINGS = 400
MAX_KMESH = 64


def _to_complex(amp: Any) -> complex:
    if isinstance(amp, bool):
        raise ValueError(f"bad hopping amplitude: {amp!r}")
    if isinstance(amp, (int, float)):
        return complex(amp)
    if isinstance(amp, complex):
        return amp
    if isinstance(amp, (list, tuple)) and len(amp) == 2:
        return complex(float(amp[0]), float(amp[1]))
    if isinstance(amp, dict):
        return complex(float(amp.get("re", 0.0)), float(amp.get("im", 0.0)))
    raise ValueError(f"bad hopping amplitude: {amp!r}")


def _parse(spec: Dict[str, Any]):
    hoppings_raw = spec.get("hoppings")
    if not isinstance(hoppings_raw, list) or not hoppings_raw:
        raise ValueError("spec needs a non-empty 'hoppings' list")
    if len(hoppings_raw) > MAX_HOPPINGS:
        raise ValueError("too many hoppings")

    dim_k = int(spec.get("dim_k", len(hoppings_raw[0].get("R", []))))
    if dim_k not in (1, 2):
        raise ValueError("dim_k must be 1 or 2")

    onsite = spec.get("onsite")
    if onsite is None:
        orbs = spec.get("orbitals")
        if orbs:
            n_orb = len(orbs)
        else:
            n_orb = 1 + max(max(int(h["i"]), int(h["j"])) for h in hoppings_raw)
        onsite = [0.0] * n_orb
    else:
        n_orb = len(onsite)
    if not (1 <= n_orb <= MAX_ORBITALS):
        raise ValueError(f"number of orbitals out of range: {n_orb}")
    onsite = [float(x) for x in onsite]

    hoppings = []
    for h in hoppings_raw:
        i, j = int(h["i"]), int(h["j"])
        R = [float(x) for x in h["R"]]
        if len(R) != dim_k:
            raise ValueError(f"hopping R has wrong length for dim_k={dim_k}: {R}")
        if not (0 <= i < n_orb and 0 <= j < n_orb):
            raise ValueError(f"hopping index out of range: i={i} j={j} (n_orb={n_orb})")
        hoppings.append((_to_complex(h["amp"]), i, j, np.array(R)))

    kmesh = max(4, min(int(spec.get("kmesh", DEFAULT_KMESH)), MAX_KMESH))
    return dim_k, n_orb, onsite, hoppings, kmesh


def _bloch(k, n_orb, onsite, hoppings):
    """Bloch Hamiltonian at reduced k. Each bond contributes amp*e^{i2pi k.R} + h.c."""
    H = np.zeros((n_orb, n_orb), dtype=complex)
    for o in range(n_orb):
        H[o, o] += onsite[o]
    for amp, i, j, R in hoppings:
        ph = amp * np.exp(2j * np.pi * float(np.dot(k, R)))
        H[i, j] += ph
        H[j, i] += np.conjugate(ph)
    return H


def _grid_points(dim_k, nk):
    axis = np.arange(nk) / nk
    if dim_k == 1:
        return [np.array([x]) for x in axis]
    return [np.array([x, y]) for x in axis for y in axis]


def _bands(n_orb, onsite, hoppings, dim_k, nk):
    pts = _grid_points(dim_k, nk)
    evals = np.empty((len(pts), n_orb))
    for idx, k in enumerate(pts):
        evals[idx] = np.linalg.eigvalsh(_bloch(k, n_orb, onsite, hoppings))  # ascending
    return evals


def _chern(n_orb, onsite, hoppings, band, nk):
    """Chern number of `band` via the Fukui-Hatsugai-Suzuki lattice method (2D)."""
    U = np.empty((nk, nk, n_orb), dtype=complex)
    for a in range(nk):
        for b in range(nk):
            _, vecs = np.linalg.eigh(_bloch(np.array([a / nk, b / nk]), n_orb, onsite, hoppings))
            U[a, b] = vecs[:, band]

    def link(u1, u2):
        z = np.vdot(u1, u2)
        az = abs(z)
        return z / az if az > 1e-12 else 1.0 + 0j

    total = 0.0
    for a in range(nk):
        for b in range(nk):
            u00 = U[a, b]
            u10 = U[(a + 1) % nk, b]
            u01 = U[a, (b + 1) % nk]
            u11 = U[(a + 1) % nk, (b + 1) % nk]
            plaquette = link(u00, u10) * link(u10, u11) * np.conjugate(link(u01, u11)) * np.conjugate(link(u00, u01))
            total += np.angle(plaquette)
    return total / (2.0 * np.pi)


def verify_tight_binding(spec: Dict[str, Any]) -> Dict[str, Any]:
    if np is None:
        return error(KIND, "numpy is not installed; cannot run tight-binding verification")

    claims = spec.get("claims") or {}
    if not isinstance(claims, dict) or not claims:
        return unverifiable(
            KIND, "no claims to check; add a 'claims' object (flat_band / gapped_at_filling / chern_number)"
        )
    try:
        dim_k, n_orb, onsite, hoppings, nk = _parse(spec)
    except (KeyError, ValueError, TypeError) as exc:
        return error(KIND, f"invalid tight_binding spec: {exc}")

    evals = _bands(n_orb, onsite, hoppings, dim_k, nk)
    bandwidths = (evals.max(axis=0) - evals.min(axis=0)).tolist()

    evidence: Dict[str, Any] = {"n_orb": n_orb, "dim_k": dim_k, "kmesh": nk,
                                "bandwidths": [round(b, 6) for b in bandwidths]}
    verdicts: List[bool] = []
    reasons: List[str] = []

    if "flat_band" in claims:
        c = claims["flat_band"]
        bi = int(c["band_index"])
        tol = float(c.get("max_bandwidth", 0.01))
        if not (0 <= bi < n_orb):
            return error(KIND, f"flat_band.band_index {bi} out of range (n_orb={n_orb})")
        bw = float(bandwidths[bi])
        ok = bw <= tol
        verdicts.append(ok)
        evidence["flat_band"] = {"band_index": bi, "bandwidth": round(bw, 6), "max_bandwidth": tol}
        reasons.append(f"band {bi} bandwidth={bw:.4g} {'<=' if ok else '>'} {tol}")

    if "gapped_at_filling" in claims:
        c = claims["gapped_at_filling"]
        filling = float(c["filling"])
        min_gap = float(c.get("min_gap", 0.0))
        n_filled = int(round(filling * n_orb))
        if not (1 <= n_filled < n_orb):
            return error(KIND, f"filling {filling} gives n_filled={n_filled}, must be 1..{n_orb - 1}")
        gap = float(np.min(evals[:, n_filled] - evals[:, n_filled - 1]))
        ok = gap >= min_gap
        verdicts.append(ok)
        evidence["gapped_at_filling"] = {"filling": filling, "n_filled": n_filled,
                                         "direct_gap": round(gap, 6), "min_gap": min_gap}
        reasons.append(f"direct gap at filling {filling}={gap:.4g} {'>=' if ok else '<'} {min_gap}")

    if "chern_number" in claims:
        c = claims["chern_number"]
        bi = int(c["band_index"])
        if dim_k != 2:
            return error(KIND, "chern_number requires dim_k=2")
        if not (0 <= bi < n_orb):
            return error(KIND, f"chern_number.band_index {bi} out of range (n_orb={n_orb})")
        raw = _chern(n_orb, onsite, hoppings, bi, nk)
        C = int(round(raw))
        sub: Dict[str, Any] = {"band_index": bi, "chern": C, "chern_raw": round(float(raw), 4)}
        if "expected" in c:
            exp = int(c["expected"])
            ok = (C == exp)
            sub["expected"] = exp
            verdicts.append(ok)
            reasons.append(f"Chern(band {bi})={C} {'==' if ok else '!='} expected {exp}")
        else:
            reasons.append(f"Chern(band {bi})={C} (informational)")
        evidence["chern_number"] = sub

    overall = all(verdicts) if verdicts else None
    return {
        "kind": KIND,
        "checkable": True,
        "claim_supported": overall,
        "evidence": evidence,
        "reason": "; ".join(reasons) if reasons else "computed band structure",
        "error": None,
    }
