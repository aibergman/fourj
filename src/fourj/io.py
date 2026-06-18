"""Input and output helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .structure import CrystalStructure, strip_comment


class EnergyTable:
    """FourJ q-point and energy table.

    Args:
        q: q-points in fractional reciprocal coordinates.
        energy_hartree: Total energies in Hartree.
    """

    def __init__(self, q: np.ndarray, energy_hartree: np.ndarray) -> None:
        self.q = q
        self.energy_hartree = energy_hartree

    @classmethod
    def from_file(cls, path: Path, q_cols=(0, 1, 2), e_col=3) -> "EnergyTable":
        """Read an `energy_vs_q.dat`-style table."""
        rows = []
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            text = strip_comment(line)
            if not text:
                continue
            fields = text.split()
            needed = max(max(q_cols), e_col)
            if len(fields) <= needed:
                raise ValueError(f"Line {lineno} in {path} has too few columns")
            rows.append(([float(fields[i]) for i in q_cols], float(fields[e_col])))
        if not rows:
            raise ValueError(f"No q/E rows found in {path}")
        return cls(
            q=np.asarray([row[0] for row in rows], dtype=float),
            energy_hartree=np.asarray([row[1] for row in rows], dtype=float),
        )


def unique_int_rows(rows: np.ndarray) -> np.ndarray:
    seen = set()
    out = []
    for row in rows:
        key = tuple(int(x) for x in row)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return np.asarray(out, dtype=int)


class VectorSet:
    """Set of integer real-space lattice vectors."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = unique_int_rows(np.asarray(vectors, dtype=int))

    @classmethod
    def from_jfile(cls, path: Path) -> "VectorSet":
        """Read vectors from a `jfile`-like file using columns 3-5."""
        vectors = []
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            text = strip_comment(line)
            if not text:
                continue
            fields = text.split()
            if len(fields) < 5:
                raise ValueError(f"Line {lineno} in {path} has too few columns")
            vectors.append([int(round(float(x))) for x in fields[2:5]])
        if not vectors:
            raise ValueError(f"No R vectors found in {path}")
        return cls(np.asarray(vectors, dtype=int))

    @classmethod
    def from_radius(cls, structure: CrystalStructure, rmax: float | None) -> "VectorSet":
        """Generate integer lattice vectors inside a real-space cutoff."""
        import itertools
        import math

        if rmax is None:
            from .constants import BOHR_TO_ANGSTROM

            rmax = 3.0 * BOHR_TO_ANGSTROM * float(np.linalg.norm(structure.lattice_bohr, axis=1).max())
        lengths = np.linalg.norm(structure.lattice_angstrom, axis=1)
        bounds = [int(math.ceil(rmax / max(length, 1e-12))) + 1 for length in lengths]
        vectors = []
        for r in itertools.product(*(range(-b, b + 1) for b in bounds)):
            cart = np.asarray(r, dtype=float) @ structure.lattice_angstrom
            if float(np.linalg.norm(cart)) <= rmax and any(r):
                vectors.append(r)
        if not vectors:
            raise ValueError("Generated no R vectors; increase --rmax")
        return cls(np.asarray(vectors, dtype=int))


class ResultWriter:
    """Write full-transform output tables."""

    def __init__(self, output_prefix: Path) -> None:
        self.output_prefix = output_prefix

    def write_exchange_outputs(
        self,
        vectors: np.ndarray,
        jij: np.ndarray,
        structure: CrystalStructure,
        q: np.ndarray,
        jq_delta: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[Path, Path, Path]:
        """Write `J_ij`, distance, and `J(0)-J(q)` output tables."""
        lattice = structure.lattice_angstrom
        rows = []
        for r, j in zip(vectors, jij):
            cart = r @ lattice
            rows.append([1, 1, int(r[0]), int(r[1]), int(r[2]), float(np.linalg.norm(cart)), float(j.real), float(j.imag)])
        rows.sort(key=lambda row: (row[5], row[2], row[3], row[4]))

        jfile_path = self.output_prefix.with_suffix(".jij.dat")
        with jfile_path.open("w") as handle:
            handle.write("# i j R1 R2 R3 distance_A J_real_mRy J_imag_mRy\n")
            for row in rows:
                handle.write(
                    f"{row[0]:4d} {row[1]:4d} {row[2]:5d} {row[3]:5d} {row[4]:5d} "
                    f"{row[5]:14.8f} {row[6]:16.8f} {row[7]:16.8e}\n"
                )

        dist_path = self.output_prefix.with_suffix(".Jij_vs_distance.dat")
        with dist_path.open("w") as handle:
            handle.write("# distance_A J_real_mRy J_imag_mRy R1 R2 R3\n")
            for row in rows:
                handle.write(
                    f"{row[5]:14.8f} {row[6]:16.8f} {row[7]:16.8e} "
                    f"{row[2]:5d} {row[3]:5d} {row[4]:5d}\n"
                )

        jq_path = self.output_prefix.with_suffix(".Jq_delta.dat")
        with jq_path.open("w") as handle:
            handle.write("# q1 q2 q3 J0_minus_Jq_mRy normalized_weight\n")
            for qq, jj, ww in zip(q, jq_delta, weights):
                handle.write(f"{qq[0]:14.8f} {qq[1]:14.8f} {qq[2]:14.8f} {jj:16.8f} {ww:16.8e}\n")
        return jfile_path, dist_path, jq_path
