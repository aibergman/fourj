"""q-mesh inference and q-space symmetrization."""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from .structure import CrystalStructure
from .symmetry import SymmetryAnalyzer


def simpson_weights(n: int) -> np.ndarray:
    if n < 2:
        return np.ones(n)
    if n % 2 == 0:
        w = np.ones(n)
        w[1:-1] = 2.0
        return w
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w


def infer_nested_mesh(q: np.ndarray, tol: float) -> tuple[tuple[int, int, int], np.ndarray]:
    n = len(q)
    if n < 8:
        raise ValueError("Need at least 8 q points to infer a 3D mesh")

    d3 = q[1] - q[0]
    n3 = 1
    for idx in range(1, n):
        if np.linalg.norm((q[idx] - q[0]) - idx * d3) <= tol:
            n3 = idx + 1
        else:
            break
    if n3 < 2 or n % n3 != 0:
        raise ValueError("Could not infer fastest q-mesh dimension")

    d2 = q[n3] - q[0]
    nblocks = n // n3
    n2 = 1
    for j in range(1, nblocks):
        idx = j * n3
        if np.linalg.norm((q[idx] - q[0]) - j * d2) <= tol:
            n2 = j + 1
        else:
            break
    if n2 < 2 or nblocks % n2 != 0:
        raise ValueError("Could not infer middle q-mesh dimension")

    n1 = nblocks // n2
    if n1 < 2:
        raise ValueError("Could not infer slowest q-mesh dimension")

    d1 = q[n2 * n3] - q[0]
    for i, j, k in itertools.product(range(n1), range(n2), range(n3)):
        idx = (i * n2 + j) * n3 + k
        expected = q[0] + i * d1 + j * d2 + k * d3
        if np.linalg.norm(q[idx] - expected) > tol:
            raise ValueError("q points are not a regular nested mesh")
    return (n1, n2, n3), np.vstack([d1, d2, d3])


def canonical_q_key(q: np.ndarray, decimals: int) -> tuple[float, float, float]:
    wrapped = np.mod(q, 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=10.0 ** -decimals)] = 0.0
    return tuple(np.round(wrapped, decimals))


class QMeshSymmetrizer:
    """Build symmetry-completed q samples and integration weights.

    Args:
        structure: Crystal structure.
        tol: Numerical tolerance for mesh inference.
        decimals: Rounding precision for grouping equivalent q-points.
        symprec: spglib symmetry tolerance.
    """

    def __init__(self, structure: CrystalStructure, tol: float = 1e-8, decimals: int = 10, symprec: float = 1e-5) -> None:
        self.structure = structure
        self.tol = tol
        self.decimals = decimals
        self.symprec = symprec

    def build_samples(self, q: np.ndarray, energy: np.ndarray, symmetry: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Return symmetrized q-points, energies, weights, and mode label."""
        if symmetry == "none":
            weights = np.ones(len(q), dtype=float)
            return q, energy, weights / weights.sum(), "none"

        dims, steps = infer_nested_mesh(q, self.tol)
        rotations = [np.eye(3, dtype=int)]
        mode_used = "time-reversal"
        if symmetry in {"lattice", "spglib"}:
            rotations, mode_used = SymmetryAnalyzer(self.structure, self.tol, self.symprec).reciprocal_rotations()

        one_d_weights = [simpson_weights(n) for n in dims]
        grouped: dict[tuple[float, float, float], list[tuple[float, float]]] = defaultdict(list)
        for i, j, k in itertools.product(range(dims[0]), range(dims[1]), range(dims[2])):
            idx = (i * dims[1] + j) * dims[2] + k
            base_q = q[0] + i * steps[0] + j * steps[1] + k * steps[2]
            if np.linalg.norm(base_q - q[idx]) > self.tol:
                raise AssertionError("internal q-mesh indexing error")
            base_w = one_d_weights[0][i] * one_d_weights[1][j] * one_d_weights[2][k]
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                signed_q = q[0] + signs[0] * i * steps[0] + signs[1] * j * steps[1] + signs[2] * k * steps[2]
                for rotation in rotations:
                    grouped[canonical_q_key(rotation @ signed_q, self.decimals)].append((energy[idx], base_w))

        q_out, e_out, w_out = [], [], []
        for key, values in grouped.items():
            local_w = sum(w for _, w in values)
            q_out.append(key)
            e_out.append(sum(val * w for val, w in values) / local_w)
            w_out.append(local_w)
        weights = np.asarray(w_out, dtype=float)
        return np.asarray(q_out, dtype=float), np.asarray(e_out, dtype=float), weights / weights.sum(), mode_used
