"""FourJ transforms between E(q), J(q), and J_ij."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import HARTREE_TO_MRY, MRY_TO_MEV


@dataclass(frozen=True)
class ExchangeTransformResult:
    """Container for inverse-transform outputs.

    Attributes:
        vectors: Integer real-space lattice vectors.
        jij_mry: Pair exchange constants in mRy.
        jq_delta_mry: Sampled :math:`J(0)-J(q)` values in mRy.
        q: Symmetrized q-points in fractional reciprocal coordinates.
        energy_hartree: Symmetrized energies in Hartree.
        weights: Normalized q-integration weights.
    """

    vectors: np.ndarray
    jij_mry: np.ndarray
    jq_delta_mry: np.ndarray
    q: np.ndarray
    energy_hartree: np.ndarray
    weights: np.ndarray


class FrozenMagnonTransformer:
    """Convert one-sublattice fourj energies to exchange quantities.

    Args:
        theta_degrees: Spin-spiral cone angle in degrees.
        e0_mode: Reference energy mode. `q0` uses the q-point nearest Gamma;
            `min` uses the minimum energy in the supplied table.
    """

    def __init__(self, theta_degrees: float = 90.0, e0_mode: str = "q0") -> None:
        self.theta_degrees = theta_degrees
        self.e0_mode = e0_mode

    @property
    def sin2(self) -> float:
        value = math.sin(math.radians(self.theta_degrees)) ** 2
        if value <= 1e-14:
            raise ValueError("theta gives sin(theta)^2 too close to zero")
        return value

    def reference_energy(self, q: np.ndarray, energy: np.ndarray) -> float:
        """Return the reference energy used in `E(q)-E0`."""
        if self.e0_mode == "q0":
            distances = np.linalg.norm(np.mod(q + 0.5, 1.0) - 0.5, axis=1)
            return float(energy[int(np.argmin(distances))])
        if self.e0_mode == "min":
            return float(np.min(energy))
        raise ValueError(f"Unknown e0 mode: {self.e0_mode}")

    def inverse_transform(self, q: np.ndarray, energy: np.ndarray, weights: np.ndarray, vectors: np.ndarray) -> ExchangeTransformResult:
        """Evaluate the inverse Fourier transform from `E(q)` to `J_ij`.

        Args:
            q: q-points in fractional reciprocal coordinates.
            energy: Energies in Hartree.
            weights: Normalized q-integration weights.
            vectors: Integer real-space vectors.

        Returns:
            Exchange transform result with `J_ij` in mRy.
        """
        j0_minus_jq_hartree = (energy - self.reference_energy(q, energy)) / self.sin2
        phase = np.exp(-2j * np.pi * (q @ vectors.T))
        jij = -(weights[:, None] * j0_minus_jq_hartree[:, None] * phase).sum(axis=0) * HARTREE_TO_MRY
        return ExchangeTransformResult(vectors, jij, j0_minus_jq_hartree * HARTREE_TO_MRY, q, energy, weights)

    def dft_spectrum_mry(self, q: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """Return `[E(q)-E0]/sin(theta)^2` in mRy."""
        return (energy - self.reference_energy(q, energy)) / self.sin2 * HARTREE_TO_MRY


class ExchangeSpectrum:
    """Evaluate and scale spectra reconstructed from real-space exchange."""

    @staticmethod
    def from_jij(q: np.ndarray, vectors: np.ndarray, jij_mry: np.ndarray) -> np.ndarray:
        """Evaluate `J(0)-J(q)` from pair interactions in mRy."""
        phase = np.exp(2j * np.pi * (q @ vectors.T))
        return np.sum(jij_mry[None, :] * (1.0 - phase), axis=1).real

    @staticmethod
    def scale(spectrum_mry: np.ndarray, lswt_moment: float | None) -> tuple[np.ndarray, str, str]:
        """Optionally apply the `4/M` LSWT prefactor and convert to meV."""
        if lswt_moment is None:
            return spectrum_mry, r"$J(0)-J(q)$ from extracted $J_{ij}$ (mRy)", "exchange_mRy"
        if lswt_moment <= 0.0:
            raise ValueError("--lswt-moment must be positive")
        scaled = (4.0 / lswt_moment) * spectrum_mry * MRY_TO_MEV
        return scaled, rf"$4[J(0)-J(q)]/{lswt_moment:g}$ (meV)", "lswt_meV"


def symmetrize_real_space(vectors: np.ndarray, jij: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enforce the pair relation :math:`J(R)=J(-R)^*`.

    Args:
        vectors: Integer real-space vectors.
        jij: Complex exchange values aligned with `vectors`.

    Returns:
        Symmetrized vectors and exchange values.
    """
    values = {tuple(row): val for row, val in zip(vectors, jij)}
    grouped = {}
    for row, val in values.items():
        inv = tuple(-x for x in row)
        vals = [val]
        if inv in values:
            vals.append(np.conjugate(values[inv]))
        grouped[min(row, inv)] = np.mean(vals)

    out_vectors, out_jij = [], []
    for key in sorted(grouped):
        val = grouped[key]
        for row, value in ((key, val), (tuple(-x for x in key), np.conjugate(val))):
            if row in values and row not in out_vectors:
                out_vectors.append(row)
                out_jij.append(value)
    return np.asarray(out_vectors, dtype=int), np.asarray(out_jij, dtype=complex)
