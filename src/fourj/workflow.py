"""High-level class-based FourJ workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import EnergyTable, ResultWriter, VectorSet
from .lsq import LSQExchangeFitter, LSQFitResult, LSQWriter, ShellBuilder
from .mesh import QMeshSymmetrizer
from .structure import CrystalStructure, ElkInputParser
from .transforms import ExchangeTransformResult, FrozenMagnonTransformer, symmetrize_real_space
from .visualization import SeekPathPlotter


@dataclass(frozen=True)
class WorkflowConfig:
    """Configuration for a complete FourJ analysis.

    Attributes:
        energy_path: Path to the `energy_vs_q.dat` table.
        elk_path: Path to an Elk input or temporary file containing `scale`,
            `avec`, and `atoms` blocks.
        vectors_path: Optional `jfile`-like list of real-space vectors.
        rmax: Radius in Angstrom used when generating vectors instead of
            reading them from a file.
        theta: Spin-spiral cone angle in degrees.
        symmetry: q-space symmetry mode: `none`, `time-reversal`, `lattice`,
            or `spglib`.
        e0: Reference-energy mode, either `q0` or `min`.
        output_prefix: Prefix used for generated output files.
        tol: Numerical tolerance used for mesh and internal symmetry checks.
        round_decimals: Number of decimals used when grouping q-points.
        symprec: Symmetry tolerance passed to spglib and Seekpath.
        realspace_pair_symmetry: If true, enforce
            :math:`J(R)=J(-R)^*` after the inverse transform.
    """

    energy_path: Path = Path("energy_vs_q.dat")
    elk_path: Path = Path("elk.tmp")
    vectors_path: Path | None = None
    rmax: float | None = None
    theta: float = 90.0
    symmetry: str = "time-reversal"
    e0: str = "q0"
    output_prefix: Path = Path("fourj")
    tol: float = 1e-8
    round_decimals: int = 10
    symprec: float = 1e-5
    realspace_pair_symmetry: bool = True


class FrozenMagnonWorkflow:
    """High-level orchestration for parsing, transforms, fitting, and plotting.

    The workflow keeps the individual computational steps available as methods
    so scripts and notebooks can run only the parts they need. The CLI is a thin
    layer over this class.

    Args:
        config: Workflow configuration.
    """

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config
        self.structure: CrystalStructure | None = None
        self.energy_table: EnergyTable | None = None
        self.transform_result: ExchangeTransformResult | None = None
        self.symmetry_used: str | None = None
        self.lsq_result: LSQFitResult | None = None

    def load(self) -> None:
        """Read the Elk structure and frozen-magnon energy table."""
        self.structure = ElkInputParser().parse(self.config.elk_path)
        self.energy_table = EnergyTable.from_file(self.config.energy_path)

    def run_transform(self) -> ExchangeTransformResult:
        """Run q-space symmetrization and inverse Fourier transform.

        Returns:
            The full transform result, including real-space `J_ij`, sampled
            `J(0)-J(q)`, q-points, and integration weights.
        """
        if self.structure is None or self.energy_table is None:
            self.load()
        assert self.structure is not None
        assert self.energy_table is not None

        q, energy, weights, symmetry_used = QMeshSymmetrizer(
            self.structure,
            tol=self.config.tol,
            decimals=self.config.round_decimals,
            symprec=self.config.symprec,
        ).build_samples(self.energy_table.q, self.energy_table.energy_hartree, self.config.symmetry)
        self.symmetry_used = symmetry_used

        vectors_path = self.config.vectors_path
        if vectors_path is None and Path("jfile").exists():
            vectors_path = Path("jfile")
        vectors = VectorSet.from_jfile(vectors_path).vectors if vectors_path is not None else VectorSet.from_radius(self.structure, self.config.rmax).vectors

        result = FrozenMagnonTransformer(self.config.theta, self.config.e0).inverse_transform(q, energy, weights, vectors)
        if self.config.realspace_pair_symmetry:
            vectors_sym, jij_sym = symmetrize_real_space(result.vectors, result.jij_mry)
            result = ExchangeTransformResult(vectors_sym, jij_sym, result.jq_delta_mry, result.q, result.energy_hartree, result.weights)
        self.transform_result = result
        return result

    def write_transform_outputs(self) -> tuple[Path, Path, Path]:
        """Write `J_ij`, `J_ij` versus distance, and `J(0)-J(q)` tables."""
        if self.transform_result is None:
            self.run_transform()
        assert self.structure is not None
        assert self.transform_result is not None
        return ResultWriter(self.config.output_prefix).write_exchange_outputs(
            self.transform_result.vectors,
            self.transform_result.jij_mry,
            self.structure,
            self.transform_result.q,
            self.transform_result.jq_delta_mry,
            self.transform_result.weights,
        )

    def fit_lsq(self, max_shells: int | None = None, rmax: float | None = None, distance_tol: float = 1e-6) -> LSQFitResult:
        """Fit finite symmetry shells to the transformed `J(0)-J(q)` data.

        Args:
            max_shells: Maximum number of shortest shells to include.
            rmax: Optional real-space cutoff in Angstrom.
            distance_tol: Tolerance in Angstrom for radius-based selection.

        Returns:
            Least-squares fit result.
        """
        if self.transform_result is None:
            self.run_transform()
        assert self.structure is not None
        assert self.transform_result is not None
        shells, rotation_source = ShellBuilder(self.structure, self.config.tol, self.config.symprec).build(
            self.transform_result.vectors,
            max_shells=max_shells,
            rmax=rmax,
            distance_tol=distance_tol,
        )
        self.lsq_result = LSQExchangeFitter().fit(
            self.transform_result.q,
            self.transform_result.jq_delta_mry,
            shells,
            rotation_source,
        )
        return self.lsq_result

    def write_lsq_outputs(self) -> tuple[Path, Path]:
        """Write shell coefficients and q-resolved LSQ residuals."""
        if self.lsq_result is None:
            raise ValueError("Run fit_lsq before writing LSQ outputs")
        assert self.transform_result is not None
        return LSQWriter(self.config.output_prefix).write(
            self.lsq_result,
            self.transform_result.q,
            self.transform_result.jq_delta_mry,
        )

    def plot_seekpath(
        self,
        plot_kind: str = "magnon",
        plot_lswt: bool = False,
        lswt_moment: float | None = None,
        dense_path: bool = False,
        path_points: int = 202,
        tol: float = 1e-7,
        min_points: int = 2,
    ) -> tuple[Path, Path, list[str]]:
        """Plot DFT, full-FT, and optional LSQ spectra along Seekpath segments.

        Args:
            plot_kind: `magnon` for rescaled fourj energies or `energy`
                for raw energy differences.
            plot_lswt: Overlay the spectrum reconstructed from full `J_ij`.
            lswt_moment: Optional magnetic moment for `4[J(0)-J(q)]/M` meV
                scaling.
            dense_path: Evaluate reconstructed spectra on a dense Seekpath mesh.
            path_points: Points per Seekpath segment for dense curves.
            tol: Tolerance for matching existing DFT q-points to path segments.
            min_points: Minimum DFT points required to draw a segment.

        Returns:
            Tuple with plot path, tabulated path-data path, and skipped segment
            names.
        """
        if self.transform_result is None:
            self.run_transform()
        if self.energy_table is None or self.structure is None or self.transform_result is None:
            raise RuntimeError("Workflow has not loaded inputs")
        return SeekPathPlotter(self.structure, self.config.output_prefix, self.config.symprec).plot(
            self.energy_table.q,
            self.energy_table.energy_hartree,
            FrozenMagnonTransformer(self.config.theta, self.config.e0),
            plot_kind=plot_kind,
            vectors=self.transform_result.vectors,
            jij_mry=self.transform_result.jij_mry,
            lsq_result=self.lsq_result,
            plot_lswt=plot_lswt,
            lswt_moment=lswt_moment,
            dense_path=dense_path,
            path_points=path_points,
            tol=tol,
            min_points=min_points,
        )

    def max_imag_jij(self) -> float:
        """Return the maximum absolute imaginary component of fitted `J_ij`."""
        if self.transform_result is None or len(self.transform_result.jij_mry) == 0:
            return 0.0
        return float(np.max(np.abs(self.transform_result.jij_mry.imag)))
