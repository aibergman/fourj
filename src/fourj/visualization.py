"""Visualization utilities for fourj spectra."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from .constants import HARTREE_TO_MRY, MRY_TO_MEV
from .lsq import LSQFitResult
from .structure import CrystalStructure
from .transforms import ExchangeSpectrum, FrozenMagnonTransformer


def reciprocal_lattice_rows(lattice_rows: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.linalg.inv(lattice_rows).T


def seekpath_labels(label: str) -> str:
    replacements = {
        "GAMMA": r"$\Gamma$",
        "SIGMA": r"$\Sigma$",
        "DELTA": r"$\Delta$",
        "LAMBDA": r"$\Lambda$",
    }
    return replacements.get(label, label.replace("_", r"\_"))


class SeekPath:
    """Wrapper around Seekpath for original-cell high-symmetry paths."""

    def __init__(self, structure: CrystalStructure, symprec: float = 1e-5) -> None:
        self.structure = structure
        self.symprec = symprec

    def get(self) -> tuple[dict[str, np.ndarray], list[tuple[str, str]], str]:
        """Return point coordinates, path segments, and Bravais label."""
        try:
            import seekpath  # type: ignore
        except Exception as exc:
            raise RuntimeError("Plotting along symmetry paths requires seekpath in the active Python environment") from exc
        path_data = seekpath.get_path_orig_cell(self.structure.spglib_cell, symprec=self.symprec)
        points = {label: np.asarray(coords, dtype=float) for label, coords in path_data["point_coords"].items()}
        bravais = str(path_data.get("bravais_lattice_extended", path_data.get("bravais_lattice", "unknown")))
        return points, list(path_data["path"]), bravais


def point_on_segment(q: np.ndarray, start: np.ndarray, end: np.ndarray, tol: float) -> tuple[bool, float]:
    direction = end - start
    denom = float(np.dot(direction, direction))
    if denom <= tol * tol:
        return np.linalg.norm(q - start) <= tol, 0.0
    t = float(np.dot(q - start, direction) / denom)
    if t < -tol or t > 1.0 + tol:
        return False, t
    return np.linalg.norm(q - (start + t * direction)) <= tol, min(1.0, max(0.0, t))


def find_existing_points_on_segment(q: np.ndarray, values: np.ndarray, start: np.ndarray, end: np.ndarray, tol: float) -> list[tuple[float, np.ndarray, float]]:
    points = []
    seen = set()
    for qq, value in zip(q, values):
        on_segment, t = point_on_segment(qq, start, end, tol)
        if not on_segment:
            continue
        key = tuple(np.round(qq, 10))
        if key in seen:
            continue
        seen.add(key)
        points.append((t, qq, float(value)))
    points.sort(key=lambda item: item[0])
    return points


def make_seekpath_line(start: np.ndarray, end: np.ndarray, npoints: int) -> list[tuple[float, np.ndarray]]:
    if npoints < 2:
        raise ValueError("--lswt-path-points must be at least 2")
    return [(float(t), start + float(t) * (end - start)) for t in np.linspace(0.0, 1.0, npoints)]


class SeekPathPlotter:
    """Matplotlib plots for DFT, full-FT, and LSQ spectra along Seekpath lines."""

    def __init__(self, structure: CrystalStructure, output_prefix: Path, symprec: float = 1e-5) -> None:
        self.structure = structure
        self.output_prefix = output_prefix
        self.symprec = symprec

    def plot(
        self,
        q_raw: np.ndarray,
        energy_raw_hartree: np.ndarray,
        transformer: FrozenMagnonTransformer,
        plot_kind: str = "magnon",
        vectors: np.ndarray | None = None,
        jij_mry: np.ndarray | None = None,
        lsq_result: LSQFitResult | None = None,
        plot_lswt: bool = False,
        lswt_moment: float | None = None,
        dense_path: bool = False,
        path_points: int = 202,
        tol: float = 1e-7,
        min_points: int = 2,
    ) -> tuple[Path, Path, list[str]]:
        """Create a Seekpath spectrum plot.

        Args:
            q_raw: Raw DFT q-points from the energy table.
            energy_raw_hartree: Raw DFT energies in Hartree.
            transformer: FourJ transformer defining theta and E0.
            plot_kind: `magnon` or `energy`.
            vectors: Real-space vectors for full-FT spectrum reconstruction.
            jij_mry: Exchange constants for full-FT spectrum reconstruction.
            lsq_result: Optional LSQ shell fit to overlay.
            plot_lswt: Overlay the full-FT reconstructed spectrum.
            lswt_moment: Optional moment for `4/M` meV scaling.
            dense_path: Evaluate reconstructed curves on a dense Seekpath mesh.
            path_points: Points per dense path segment.
            tol: q-point matching tolerance for raw DFT points.
            min_points: Minimum raw DFT points required for a segment.

        Returns:
            Plot path, sparse path-data path, and skipped segment names.
        """
        sin2 = math.sin(math.radians(transformer.theta_degrees)) ** 2
        if sin2 <= 1e-14:
            raise ValueError("theta gives sin(theta)^2 too close to zero")

        if plot_kind == "energy":
            e0 = transformer.reference_energy(q_raw, energy_raw_hartree)
            plot_values = (energy_raw_hartree - e0) * HARTREE_TO_MRY
            ylabel = r"$E(q)-E_0$ (mRy)"
            dft_column = "dft_value_mRy"
        elif plot_kind == "magnon":
            plot_values = transformer.dft_spectrum_mry(q_raw, energy_raw_hartree)
            ylabel = r"$(E(q)-E_0)//\sin^2\theta$ (mRy)"
            dft_column = "dft_magnon_mRy"
        else:
            raise ValueError(f"Unknown plot kind: {plot_kind}")

        if plot_lswt:
            if vectors is None or jij_mry is None:
                raise ValueError("LSWT/exchange overlay requires extracted J_ij values")
            if plot_kind != "magnon":
                raise ValueError("--plot-lswt is only meaningful with --plot-kind magnon")
            if lswt_moment is not None:
                if lswt_moment <= 0.0:
                    raise ValueError("--lswt-moment must be positive")
                plot_values = (4.0 / lswt_moment) * plot_values * MRY_TO_MEV
                ylabel = rf"$4[J(0)-J(q)]/{lswt_moment:g}$ (meV)"
                dft_column = "dft_lswt_meV"
            _lswt_values, lswt_label, lswt_column = ExchangeSpectrum.scale(
                ExchangeSpectrum.from_jij(q_raw, vectors, jij_mry),
                lswt_moment,
            )
        else:
            lswt_label = None
            lswt_column = None
        if lsq_result is not None and plot_kind != "magnon":
            raise ValueError("LSQ plot overlay is only meaningful with --plot-kind magnon")

        points, path_segments, bravais = SeekPath(self.structure, self.symprec).get()
        recip = reciprocal_lattice_rows(self.structure.lattice_angstrom)
        x_offset = 0.0
        tick_positions = []
        tick_labels = []
        rows = []
        plotted_segments = []
        skipped_segments = []

        mpl_config_dir = Path("/private/tmp/matplotlib")
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for start_label, end_label in path_segments:
            start = points[start_label]
            end = points[end_label]
            segment_points = find_existing_points_on_segment(q_raw, plot_values, start, end, tol)
            if len(segment_points) < min_points:
                skipped_segments.append(f"{start_label}-{end_label}")
                continue

            segment_length = float(np.linalg.norm((end - start) @ recip))
            x = np.asarray([x_offset + t * segment_length for t, _, _ in segment_points], dtype=float)
            y = np.asarray([value for _, _, value in segment_points], dtype=float)
            dft_line = ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.2, alpha=0.5, label="DFT frozen magnon")

            y_secondary = None
            if plot_lswt:
                q_existing = np.asarray([qq for _, qq, _ in segment_points], dtype=float)
                y_secondary = ExchangeSpectrum.scale(ExchangeSpectrum.from_jij(q_existing, vectors, jij_mry), lswt_moment)[0]
                if dense_path:
                    line = make_seekpath_line(start, end, path_points)
                    q_line = np.asarray([qq for _, qq in line], dtype=float)
                    y_line = ExchangeSpectrum.scale(ExchangeSpectrum.from_jij(q_line, vectors, jij_mry), lswt_moment)[0]
                    x_line = np.asarray([x_offset + t * segment_length for t, _ in line], dtype=float)
                else:
                    y_line = y_secondary
                    x_line = x
                ax.plot(x_line, y_line, marker=None if dense_path else "s", markersize=3.0, linewidth=1.5 if dense_path else 1.1, linestyle="--", color=dft_line[0].get_color(), label=lswt_label)

            if lsq_result is not None:
                q_existing = np.asarray([qq for _, qq, _ in segment_points], dtype=float)
                y_lsq_existing = ExchangeSpectrum.scale(lsq_result.spectrum(q_existing), lswt_moment if plot_lswt else None)[0]
                if y_secondary is None:
                    y_secondary = y_lsq_existing
                if dense_path:
                    line = make_seekpath_line(start, end, path_points)
                    q_line = np.asarray([qq for _, qq in line], dtype=float)
                    y_lsq = ExchangeSpectrum.scale(lsq_result.spectrum(q_line), lswt_moment if plot_lswt else None)[0]
                    x_lsq = np.asarray([x_offset + t * segment_length for t, _ in line], dtype=float)
                else:
                    y_lsq = y_lsq_existing
                    x_lsq = x
                ax.plot(x_lsq, y_lsq, marker=None, linewidth=1.4, linestyle=":", color=dft_line[0].get_color(), label=f"LSQ shell fit ({len(lsq_result.shells)} shells)")

            tick_positions.append(x_offset)
            tick_labels.append(seekpath_labels(start_label))
            tick_positions.append(x_offset + segment_length)
            tick_labels.append(seekpath_labels(end_label))
            plotted_segments.append(f"{start_label}-{end_label}")

            if y_secondary is None:
                for xx, (t, qq, value) in zip(x, segment_points):
                    rows.append((start_label, end_label, xx, t, qq[0], qq[1], qq[2], value, None))
            else:
                for xx, (t, qq, value), secondary_value in zip(x, segment_points, y_secondary):
                    rows.append((start_label, end_label, xx, t, qq[0], qq[1], qq[2], value, secondary_value))
            x_offset += segment_length

        if not plotted_segments:
            raise ValueError("No Seekpath segment had enough existing q points")

        for xpos in tick_positions:
            ax.axvline(xpos, color="0.85", linewidth=0.8, zorder=0)
        ax.set_xlim(min(tick_positions), max(tick_positions))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"Wave-vector distance ($\mathrm{\AA}^{-1}$)")
        ax.set_title(f"Seekpath {bravais}; existing input q-points only")
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        if plot_lswt or lsq_result is not None:
            handles, labels = ax.get_legend_handles_labels()
            unique = {}
            for handle, label in zip(handles, labels):
                unique.setdefault(label, handle)
            ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8)
        fig.tight_layout()

        plot_path = self.output_prefix.with_suffix(f".seekpath_{plot_kind}.png")
        data_path = self.output_prefix.with_suffix(f".seekpath_{plot_kind}.dat")
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        with data_path.open("w") as handle:
            if lswt_column is None:
                handle.write(f"# start_label end_label path_distance_1_per_A segment_fraction q1 q2 q3 {dft_column}\n")
            else:
                handle.write("# start_label end_label path_distance_1_per_A segment_fraction q1 q2 q3 " f"{dft_column} {lswt_column}\n")
            handle.write("# plotted_segments " + " ".join(plotted_segments) + "\n")
            if skipped_segments:
                handle.write("# skipped_segments " + " ".join(skipped_segments) + "\n")
            for row in rows:
                if row[8] is None:
                    handle.write(f"{row[0]:>8s} {row[1]:>8s} {row[2]:16.8f} {row[3]:12.8f} {row[4]:12.8f} {row[5]:12.8f} {row[6]:12.8f} {row[7]:16.8f}\n")
                else:
                    handle.write(f"{row[0]:>8s} {row[1]:>8s} {row[2]:16.8f} {row[3]:12.8f} {row[4]:12.8f} {row[5]:12.8f} {row[6]:12.8f} {row[7]:16.8f} {row[8]:16.8f}\n")

        if plot_lswt and dense_path:
            dense_path_file = self.output_prefix.with_suffix(f".seekpath_{plot_kind}_lswt_dense.dat")
            with dense_path_file.open("w") as handle:
                handle.write("# start_label end_label path_distance_1_per_A segment_fraction q1 q2 q3 " f"{lswt_column}\n")
                x_offset = 0.0
                for start_label, end_label in path_segments:
                    if f"{start_label}-{end_label}" in skipped_segments:
                        continue
                    start = points[start_label]
                    end = points[end_label]
                    segment_length = float(np.linalg.norm((end - start) @ recip))
                    line = make_seekpath_line(start, end, path_points)
                    q_line = np.asarray([qq for _, qq in line], dtype=float)
                    values = ExchangeSpectrum.scale(ExchangeSpectrum.from_jij(q_line, vectors, jij_mry), lswt_moment)[0]
                    for (t, qq), value in zip(line, values):
                        handle.write(f"{start_label:>8s} {end_label:>8s} {x_offset + t * segment_length:16.8f} {t:12.8f} {qq[0]:12.8f} {qq[1]:12.8f} {qq[2]:12.8f} {value:16.8f}\n")
                    x_offset += segment_length
        return plot_path, data_path, skipped_segments
