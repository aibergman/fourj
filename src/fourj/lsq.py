"""Linear least-squares fitting of finite exchange shells."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .structure import CrystalStructure
from .symmetry import SymmetryAnalyzer


def vector_distance_angstrom(vector: np.ndarray, structure: CrystalStructure) -> float:
    return float(np.linalg.norm(vector @ structure.lattice_angstrom))


@dataclass(frozen=True)
class ExchangeShell:
    """Symmetry-equivalent real-space vectors sharing one fitted coefficient.

    Attributes:
        index: One-based shell index ordered by distance.
        vectors: Integer lattice vectors in the shell.
        distance_angstrom: Mean shell distance in Angstrom.
    """

    index: int
    vectors: list[tuple[int, int, int]]
    distance_angstrom: float

    @property
    def size(self) -> int:
        return len(self.vectors)


@dataclass(frozen=True)
class LSQFitResult:
    """Result of a finite-shell least-squares exchange fit."""

    shells: list[ExchangeShell]
    coefficients_mry: np.ndarray
    fitted_mry: np.ndarray
    singular_values: np.ndarray
    rmse_mry: float
    max_abs_error_mry: float
    rotation_source: str

    def spectrum(self, q: np.ndarray) -> np.ndarray:
        """Evaluate fitted `J(0)-J(q)` at arbitrary q-points in mRy."""
        spectrum = np.zeros(len(q), dtype=float)
        for shell, coeff in zip(self.shells, self.coefficients_mry):
            vectors = np.asarray(shell.vectors, dtype=float)
            phase = 2.0 * np.pi * (q @ vectors.T)
            spectrum += coeff * np.sum(1.0 - np.cos(phase), axis=1)
        return spectrum


class ShellBuilder:
    """Build real-space exchange shells from symmetry operations.

    Args:
        structure: Crystal structure.
        tol: Numerical tolerance for internal symmetry checks.
        symprec: spglib symmetry tolerance.
    """

    def __init__(self, structure: CrystalStructure, tol: float = 1e-8, symprec: float = 1e-5) -> None:
        self.structure = structure
        self.symmetry = SymmetryAnalyzer(structure, tol, symprec)

    def build(self, vectors: np.ndarray, max_shells: int | None = None, rmax: float | None = None, distance_tol: float = 1e-6) -> tuple[list[ExchangeShell], str]:
        """Group vectors into symmetry shells.

        Args:
            vectors: Integer real-space vectors.
            max_shells: Optional maximum number of shortest shells.
            rmax: Optional distance cutoff in Angstrom.
            distance_tol: Tolerance for cutoff selection.

        Returns:
            Shells and a short description of the symmetry source.
        """
        rotations, source = self.symmetry.direct_rotations()
        vector_set = {tuple(int(x) for x in vector) for vector in vectors if np.any(vector)}
        if rmax is not None:
            vector_set = {
                vector for vector in vector_set
                if vector_distance_angstrom(np.asarray(vector, dtype=int), self.structure) <= rmax + distance_tol
            }

        raw_shells = []
        remaining = set(vector_set)
        while remaining:
            seed = min(remaining, key=lambda item: (vector_distance_angstrom(np.asarray(item, dtype=int), self.structure), item))
            orbit = self.symmetry.vector_orbit(np.asarray(seed, dtype=int), rotations)
            shell = sorted(remaining.intersection(orbit)) or [seed]
            for item in shell:
                remaining.discard(item)
            raw_shells.append(shell)

        raw_shells.sort(key=lambda shell: (np.mean([vector_distance_angstrom(np.asarray(v, dtype=int), self.structure) for v in shell]), shell[0]))
        if max_shells is not None:
            raw_shells = raw_shells[:max_shells]
        shells = [
            ExchangeShell(
                index=i,
                vectors=shell,
                distance_angstrom=float(np.mean([vector_distance_angstrom(np.asarray(v, dtype=int), self.structure) for v in shell])),
            )
            for i, shell in enumerate(raw_shells, 1)
        ]
        return shells, source


class LSQExchangeFitter:
    """Linear least-squares fitter for shell coefficients."""

    def fit(self, q: np.ndarray, target_mry: np.ndarray, shells: list[ExchangeShell], rotation_source: str = "") -> LSQFitResult:
        """Fit shell coefficients to `J(0)-J(q)`.

        Args:
            q: q-points in fractional reciprocal coordinates.
            target_mry: Target `J(0)-J(q)` values in mRy.
            shells: Symmetry shells defining the basis functions.
            rotation_source: Description saved in the result metadata.

        Returns:
            Least-squares fit result.
        """
        if not shells:
            raise ValueError("No shells available for LSQ fit")
        design = np.zeros((len(q), len(shells)), dtype=float)
        for column, shell in enumerate(shells):
            vectors = np.asarray(shell.vectors, dtype=float)
            phase = 2.0 * np.pi * (q @ vectors.T)
            design[:, column] = np.sum(1.0 - np.cos(phase), axis=1)
        coeffs, _residuals, _rank, singular_values = np.linalg.lstsq(design, target_mry, rcond=None)
        fitted = design @ coeffs
        residual = fitted - target_mry
        return LSQFitResult(
            shells=shells,
            coefficients_mry=coeffs,
            fitted_mry=fitted,
            singular_values=singular_values,
            rmse_mry=float(np.sqrt(np.mean(residual**2))),
            max_abs_error_mry=float(np.max(np.abs(residual))),
            rotation_source=rotation_source,
        )


class LSQWriter:
    """Write LSQ shell coefficients and q-resolved residuals."""

    def __init__(self, output_prefix: Path) -> None:
        self.output_prefix = output_prefix

    def write(self, result: LSQFitResult, q: np.ndarray, target_mry: np.ndarray) -> tuple[Path, Path]:
        """Write LSQ output files and return their paths."""
        shell_path = self.output_prefix.with_suffix(".lsq_shells.dat")
        fit_path = self.output_prefix.with_suffix(".lsq_fit_q.dat")
        with shell_path.open("w") as handle:
            handle.write("# LSQ fit of J(0)-J(q) to symmetry-grouped real-space shells\n")
            handle.write(f"# rotation_source {result.rotation_source}\n")
            handle.write(f"# rmse_mRy {result.rmse_mry:.12g}\n")
            handle.write(f"# max_abs_error_mRy {result.max_abs_error_mry:.12g}\n")
            handle.write("# singular_values " + " ".join(f"{value:.12g}" for value in result.singular_values) + "\n")
            handle.write("# shell size distance_A J_shell_mRy vectors_R1_R2_R3\n")
            for shell, coeff in zip(result.shells, result.coefficients_mry):
                vectors_text = ";".join(f"{r[0]},{r[1]},{r[2]}" for r in shell.vectors)
                handle.write(f"{shell.index:5d} {shell.size:5d} {shell.distance_angstrom:14.8f} {coeff:16.8f} {vectors_text}\n")
        with fit_path.open("w") as handle:
            handle.write("# q1 q2 q3 target_mRy fitted_mRy residual_mRy\n")
            for qq, target, fitted in zip(q, target_mry, result.fitted_mry):
                handle.write(f"{qq[0]:14.8f} {qq[1]:14.8f} {qq[2]:14.8f} {target:16.8f} {fitted:16.8f} {fitted - target:16.8f}\n")
        return shell_path, fit_path
