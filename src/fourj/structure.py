"""Crystal structure representation and Elk input parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import BOHR_TO_ANGSTROM


@dataclass(frozen=True)
class CrystalStructure:
    """Crystal structure as parsed from an Elk input file.

    Attributes:
        scale: Elk lattice scale in Bohr.
        avec: Lattice vectors before applying `scale`, row-wise.
        species: Species names from the Elk `atoms` block.
        positions: Fractional atomic coordinates.
        species_numbers: Integer species labels suitable for spglib.
    """

    scale: float
    avec: np.ndarray
    species: list[str]
    positions: np.ndarray
    species_numbers: np.ndarray

    @property
    def lattice_bohr(self) -> np.ndarray:
        return self.scale * self.avec

    @property
    def lattice_angstrom(self) -> np.ndarray:
        return BOHR_TO_ANGSTROM * self.lattice_bohr

    @property
    def spglib_cell(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.lattice_angstrom, self.positions, self.species_numbers


def strip_comment(line: str) -> str:
    return line.split(":", 1)[0].split("!", 1)[0].strip()


def _read_numeric_block(lines: list[str], start: int, nrows: int) -> np.ndarray:
    rows = []
    for idx in range(start, start + nrows):
        text = strip_comment(lines[idx])
        if not text:
            raise ValueError(f"Expected numeric row at line {idx + 1}")
        rows.append([float(x) for x in text.split()[:3]])
    return np.asarray(rows, dtype=float)


class ElkInputParser:
    """Parse the subset of Elk input needed for fourj analysis."""

    def parse(self, path: Path) -> CrystalStructure:
        """Parse `scale`, `avec`, and `atoms` from an Elk file.

        Args:
            path: Elk input or temporary file.

        Returns:
            Parsed crystal structure.
        """
        lines = path.read_text().splitlines()
        scale = None
        avec = None
        species: list[str] = []
        positions: list[list[float]] = []
        species_numbers: list[int] = []

        i = 0
        while i < len(lines):
            key = strip_comment(lines[i]).lower()
            if key == "scale":
                i += 1
                while i < len(lines) and not strip_comment(lines[i]):
                    i += 1
                scale = float(strip_comment(lines[i]).split()[0])
            elif key == "avec":
                avec = _read_numeric_block(lines, i + 1, 3)
                i += 3
            elif key == "atoms":
                i += 1
                while i < len(lines) and not strip_comment(lines[i]):
                    i += 1
                nspecies = int(strip_comment(lines[i]).split()[0])
                for ispecies in range(1, nspecies + 1):
                    i += 1
                    while i < len(lines) and not strip_comment(lines[i]):
                        i += 1
                    species.append(Path(strip_comment(lines[i]).strip("'\"")).stem)
                    i += 1
                    while i < len(lines) and not strip_comment(lines[i]):
                        i += 1
                    natoms = int(strip_comment(lines[i]).split()[0])
                    for _ in range(natoms):
                        i += 1
                        while i < len(lines) and not strip_comment(lines[i]):
                            i += 1
                        positions.append([float(x) for x in strip_comment(lines[i]).split()[:3]])
                        species_numbers.append(ispecies)
            i += 1

        if scale is None:
            raise ValueError(f"Could not find 'scale' in {path}")
        if avec is None:
            raise ValueError(f"Could not find 'avec' in {path}")
        if not positions:
            raise ValueError(f"Could not find 'atoms' block in {path}")

        return CrystalStructure(
            scale=scale,
            avec=avec,
            species=species,
            positions=np.asarray(positions, dtype=float),
            species_numbers=np.asarray(species_numbers, dtype=int),
        )
