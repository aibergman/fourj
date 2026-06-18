"""Crystal symmetry helpers for q-space averaging and real-space shells."""

from __future__ import annotations

import itertools

import numpy as np

from .structure import CrystalStructure


class SymmetryAnalyzer:
    """Find direct and reciprocal point-group rotations.

    spglib is used when available. If spglib cannot provide rotations, a small
    internal integer-matrix search is used as a fallback.

    Args:
        structure: Crystal structure.
        tol: Numerical tolerance for internal checks.
        symprec: spglib symmetry tolerance.
    """

    def __init__(self, structure: CrystalStructure, tol: float = 1e-8, symprec: float = 1e-5) -> None:
        self.structure = structure
        self.tol = tol
        self.symprec = symprec

    def spglib_direct_rotations(self) -> list[np.ndarray]:
        """Return direct-space rotations from spglib, or an empty list."""
        try:
            import spglib  # type: ignore
        except Exception:
            return []
        dataset = spglib.get_symmetry(self.structure.spglib_cell, symprec=self.symprec)
        return [np.asarray(rotation, dtype=int) for rotation in dataset["rotations"]]

    def reciprocal_rotations(self) -> tuple[list[np.ndarray], str]:
        """Return reciprocal-space rotations and a source description."""
        direct_rotations, source = self.direct_rotations()
        rotations = [np.linalg.inv(rotation).T.astype(int) for rotation in direct_rotations]
        return rotations, source.replace("direct", "reciprocal")

    def direct_rotations(self) -> tuple[list[np.ndarray], str]:
        """Return direct-space rotations and a source description."""
        rotations = self.spglib_direct_rotations()
        if rotations:
            return rotations, f"spglib ({len(rotations)} direct rotations)"

        rotations = []
        metric = self.structure.lattice_bohr @ self.structure.lattice_bohr.T
        for values in itertools.product((-1, 0, 1), repeat=9):
            rotation = np.asarray(values, dtype=int).reshape(3, 3)
            det = round(float(np.linalg.det(rotation)))
            if abs(det) != 1:
                continue
            if not np.allclose(rotation.T @ metric @ rotation, metric, atol=self.tol, rtol=0.0):
                continue
            if not self._maps_basis_onto_itself(rotation):
                continue
            rotations.append(rotation)
        if rotations:
            return rotations, f"internal lattice+basis ({len(rotations)} direct rotations)"
        return [np.eye(3, dtype=int)], "identity only"

    def _maps_basis_onto_itself(self, rotation: np.ndarray) -> bool:
        positions_by_species: dict[int, list[np.ndarray]] = {}
        for z, pos in zip(self.structure.species_numbers, self.structure.positions):
            positions_by_species.setdefault(int(z), []).append(np.mod(pos, 1.0))

        for z, pos in zip(self.structure.species_numbers, self.structure.positions):
            transformed = np.mod(rotation @ pos, 1.0)
            for candidate in positions_by_species[int(z)]:
                delta = np.mod(transformed - candidate + 0.5, 1.0) - 0.5
                if np.linalg.norm(delta) <= self.tol:
                    break
            else:
                return False
        return True

    def vector_orbit(self, vector: np.ndarray, rotations: list[np.ndarray]) -> set[tuple[int, int, int]]:
        """Return the symmetry orbit of a vector, including its inverse."""
        orbit = set()
        for rotation in rotations:
            rotated = rotation @ vector
            orbit.add(tuple(int(x) for x in rotated))
            orbit.add(tuple(int(x) for x in -rotated))
        return orbit
