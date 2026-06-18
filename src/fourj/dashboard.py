"""Interactive Plotly/Dash dashboard for FourJ analyses."""

from __future__ import annotations

import argparse
import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import HARTREE_TO_MRY, MRY_TO_MEV
from .transforms import ExchangeSpectrum, FrozenMagnonTransformer
from .visualization import SeekPath, find_existing_points_on_segment, make_seekpath_line, reciprocal_lattice_rows
from .workflow import FrozenMagnonWorkflow, WorkflowConfig


@dataclass(frozen=True)
class UploadedText:
    """Decoded Dash upload content."""

    filename: str
    text: str


@dataclass(frozen=True)
class PathAnalysisConfig:
    """Initial dashboard analysis based on local file paths."""

    elk_path: Path
    energy_path: Path
    vectors_path: Path | None = None
    theta: float = 90.0
    symmetry: str = "spglib"
    e0: str = "q0"
    rmax: float | None = None
    fit_lsq: bool = False
    fit_shells: int | None = None
    moment: float | None = None
    dense_points: int = 400


def _decode_upload(contents: str | None, filename: str | None, label: str) -> UploadedText:
    if not contents:
        raise ValueError(f"Upload {label} before running the analysis")
    if not filename:
        filename = label
    try:
        _metadata, encoded = contents.split(",", 1)
        data = base64.b64decode(encoded)
    except Exception as exc:  # pragma: no cover - defensive Dash input validation
        raise ValueError(f"Could not decode uploaded {label}") from exc
    return UploadedText(filename=filename, text=data.decode("utf-8"))


def _empty_figure(message: str) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"size": 15})
    fig.update_layout(template="plotly_white", autosize=True, title={"font": {"size": 15}}, margin={"l": 8, "r": 8, "t": 22, "b": 4})
    return fig


def _cartesian_reciprocal_vectors(real_lattice_rows: np.ndarray) -> np.ndarray:
    """Return Cartesian reciprocal vectors as rows, matching bz_viz.py."""
    a1, a2, a3 = np.asarray(real_lattice_rows, dtype=float)
    volume = float(np.dot(a1, np.cross(a2, a3)))
    if abs(volume) <= 1e-14:
        raise ValueError("Cannot build reciprocal lattice from a singular real-space lattice")
    return np.asarray(
        [
            2.0 * np.pi * np.cross(a2, a3) / volume,
            2.0 * np.pi * np.cross(a3, a1) / volume,
            2.0 * np.pi * np.cross(a1, a2) / volume,
        ],
        dtype=float,
    )


def _fractional_q_to_cartesian(q: np.ndarray, reciprocal_rows: np.ndarray) -> np.ndarray:
    """Convert reciprocal-lattice coordinates to Cartesian k in 1/A."""
    return np.asarray(q, dtype=float) @ reciprocal_rows


def _structure_metadata(workflow: FrozenMagnonWorkflow) -> dict[str, str]:
    """Return best-effort symmetry and structure labels for display."""
    assert workflow.structure is not None
    metadata: dict[str, str] = {}
    try:
        _points, _segments, bravais = SeekPath(workflow.structure, workflow.config.symprec).get()
        metadata["Bravais"] = bravais
    except Exception:
        pass
    try:
        import spglib  # type: ignore

        dataset = spglib.get_symmetry_dataset(workflow.structure.spglib_cell, symprec=workflow.config.symprec)
        if dataset is not None:
            number = getattr(dataset, "number", None) if not isinstance(dataset, dict) else dataset.get("number")
            international = getattr(dataset, "international", None) if not isinstance(dataset, dict) else dataset.get("international")
            hall = getattr(dataset, "hall", None) if not isinstance(dataset, dict) else dataset.get("hall")
            if international and number:
                metadata["Space group"] = f"{international} ({number})"
            elif international:
                metadata["Space group"] = str(international)
            if hall:
                metadata["Hall"] = str(hall)
    except Exception:
        pass
    return metadata


def _first_bz_geometry(reciprocal: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    """Return first Brillouin-zone vertices, true edges, and cell vertex ids."""
    try:
        from scipy.spatial import Voronoi  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("The BZ dashboard view requires scipy") from exc

    grid_range = np.arange(-2, 3)
    grid = np.asarray(np.meshgrid(grid_range, grid_range, grid_range, indexing="ij")).reshape(3, -1).T
    points = _fractional_q_to_cartesian(grid, reciprocal)
    voronoi = Voronoi(points)
    origin_index = int(np.argmin(np.linalg.norm(points, axis=1)))

    edges: set[tuple[int, int]] = set()
    cell_vertices: set[int] = set()
    for ridge_points, ridge_vertices in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        if origin_index not in ridge_points or -1 in ridge_vertices:
            continue
        cell_vertices.update(int(v) for v in ridge_vertices)
        for idx, vertex in enumerate(ridge_vertices):
            nxt = ridge_vertices[(idx + 1) % len(ridge_vertices)]
            edges.add(tuple(sorted((int(vertex), int(nxt)))))
    return np.asarray(voronoi.vertices, dtype=float), sorted(edges), sorted(cell_vertices)


def _fold_fractional_to_first_bz(q: np.ndarray, reciprocal: np.ndarray) -> np.ndarray:
    """Map fractional reciprocal q-points to nearest first-BZ images."""
    shifts = np.asarray(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T
    wrapped = np.mod(q + 0.5, 1.0) - 0.5
    folded = []
    for point in wrapped:
        candidates = _fractional_q_to_cartesian(point[None, :] + shifts, reciprocal)
        folded.append(candidates[int(np.argmin(np.linalg.norm(candidates, axis=1)))])
    return np.asarray(folded, dtype=float)


def _seekpath_cartesian_traces(workflow: FrozenMagnonWorkflow, reciprocal: np.ndarray, dense_points: int = 80) -> tuple[list[Any], list[tuple[str, np.ndarray]]]:
    """Build 3D traces for the original-cell Seekpath line in Cartesian k-space."""
    import plotly.graph_objects as go

    assert workflow.structure is not None
    points, segments, _bravais = SeekPath(workflow.structure, workflow.config.symprec).get()
    traces: list[Any] = []
    label_points: dict[str, np.ndarray] = {}
    for segment_index, (start_label, end_label) in enumerate(segments):
        start = points[start_label]
        end = points[end_label]
        line = make_seekpath_line(start, end, max(2, dense_points))
        q_line = np.asarray([q for _, q in line], dtype=float)
        k_line = _fractional_q_to_cartesian(q_line, reciprocal)
        traces.append(
            go.Scatter3d(
                x=k_line[:, 0],
                y=k_line[:, 1],
                z=k_line[:, 2],
                mode="lines",
                line={"color": "#ef4444", "width": 7},
                hovertemplate=f"{start_label}-{end_label}<extra>Seekpath</extra>",
                name="Seekpath line" if segment_index == 0 else None,
                showlegend=segment_index == 0,
            )
        )
        label_points[start_label] = _fractional_q_to_cartesian(start[None, :], reciprocal)[0]
        label_points[end_label] = _fractional_q_to_cartesian(end[None, :], reciprocal)[0]
    return traces, sorted(label_points.items())


def _plot_reciprocal_cell(workflow: FrozenMagnonWorkflow) -> Any:
    import plotly.graph_objects as go

    assert workflow.structure is not None
    assert workflow.energy_table is not None
    reciprocal = _cartesian_reciprocal_vectors(workflow.structure.lattice_angstrom)
    bz_vertices, edges, cell_vertex_indices = _first_bz_geometry(reciprocal)
    q_cart = _fold_fractional_to_first_bz(workflow.energy_table.q, reciprocal)
    transformer = FrozenMagnonTransformer(workflow.config.theta, workflow.config.e0)
    e0 = transformer.reference_energy(workflow.energy_table.q, workflow.energy_table.energy_hartree)
    energy_mry = (workflow.energy_table.energy_hartree - e0) * HARTREE_TO_MRY

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for a, b in edges:
        p1 = bz_vertices[a]
        p2 = bz_vertices[b]
        edge_x.extend([float(p1[0]), float(p2[0]), None])
        edge_y.extend([float(p1[1]), float(p2[1]), None])
        edge_z.extend([float(p1[2]), float(p2[2]), None])

    cell_vertices = bz_vertices[cell_vertex_indices]
    seekpath_traces, labels = _seekpath_cartesian_traces(workflow, reciprocal)
    label_text = [label.replace("GAMMA", r"\Gamma") for label, _coords in labels]
    label_coords = np.asarray([coords for _label, coords in labels], dtype=float) if labels else np.empty((0, 3))

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=cell_vertices[:, 0],
            y=cell_vertices[:, 1],
            z=cell_vertices[:, 2],
            alphahull=0,
            color="#67e8f9",
            opacity=0.18,
            name="first BZ volume",
            hoverinfo="skip",
            showscale=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line={"color": "#111827", "width": 4},
            name="first BZ edges",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=q_cart[:, 0],
            y=q_cart[:, 1],
            z=q_cart[:, 2],
            mode="markers",
            marker={"size": 4, "color": energy_mry, "colorscale": "Viridis", "opacity": 0.82, "colorbar": {"title": "E-E0 (mRy)", "len": 0.62}},
            customdata=np.column_stack([workflow.energy_table.q, energy_mry]),
            hovertemplate="q=(%{customdata[0]:.5f}, %{customdata[1]:.5f}, %{customdata[2]:.5f})<br>k=(%{x:.4f}, %{y:.4f}, %{z:.4f}) 1/A<br>E-E0=%{customdata[3]:.6f} mRy<extra>input q</extra>",
            name="input q-points colored by energy",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker={"size": 5, "color": "#dc2626"},
            text=["Gamma"],
            textposition="top center",
            name="Gamma",
        )
    )
    for trace in seekpath_traces:
        fig.add_trace(trace)
    if len(label_coords):
        fig.add_trace(
            go.Scatter3d(
                x=label_coords[:, 0],
                y=label_coords[:, 1],
                z=label_coords[:, 2],
                mode="markers+text",
                marker={"size": 4, "color": "#ef4444"},
                text=label_text,
                textposition="top center",
                hovertemplate="%{text}<br>k=(%{x:.4f}, %{y:.4f}, %{z:.4f}) 1/A<extra>Seekpath point</extra>",
                name="Seekpath labels",
                showlegend=False,
            )
        )

    fig.update_layout(
        title={"text": "First Brillouin Zone, q-Points, and Seekpath Line", "font": {"size": 16}},
        template="plotly_white",
        autosize=True,
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        scene={
            "xaxis": {"title": "kx (1/A)", "showbackground": False},
            "yaxis": {"title": "ky (1/A)", "showbackground": False},
            "zaxis": {"title": "kz (1/A)", "showbackground": False},
            "aspectmode": "data",
            "camera": {"projection": {"type": "orthographic"}},
        },
        legend={"orientation": "h", "y": 0.99, "x": 0.01, "xanchor": "left", "yanchor": "top", "bgcolor": "rgba(255,255,255,0.70)", "font": {"size": 10}},
    )
    return fig


def _plot_input_energy(workflow: FrozenMagnonWorkflow) -> Any:
    import plotly.graph_objects as go

    assert workflow.energy_table is not None
    transformer = FrozenMagnonTransformer(workflow.config.theta, workflow.config.e0)
    e0 = transformer.reference_energy(workflow.energy_table.q, workflow.energy_table.energy_hartree)
    y = (workflow.energy_table.energy_hartree - e0) * HARTREE_TO_MRY
    custom = np.column_stack([workflow.energy_table.q, workflow.energy_table.energy_hartree])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(y)),
            y=y,
            mode="lines+markers",
            line={"color": "#2563eb", "width": 1.7},
            marker={"size": 5},
            customdata=custom,
            hovertemplate="index=%{x}<br>q=(%{customdata[0]:.5f}, %{customdata[1]:.5f}, %{customdata[2]:.5f})<br>E=%{customdata[3]:.10f} Ha<br>Delta E=%{y:.6f} mRy<extra></extra>",
            name="DFT input",
        )
    )
    fig.update_layout(
        title={"text": "Input E(q) vs q-Index", "font": {"size": 16}},
        template="plotly_white",
        autosize=True,
        margin={"l": 44, "r": 8, "t": 24, "b": 28},
        xaxis_title="Input q-index",
        yaxis_title=r"$E(\mathbf{q})-E0$ (mRy)",
    )
    return fig


def _path_axis(points: dict[str, np.ndarray], segments: list[tuple[str, str]], reciprocal: np.ndarray) -> tuple[dict[tuple[str, str], float], list[float], list[str]]:
    offset = 0.0
    starts: dict[tuple[str, str], float] = {}
    ticks: list[float] = []
    labels: list[str] = []
    for start_label, end_label in segments:
        start = points[start_label]
        end = points[end_label]
        length = float(np.linalg.norm((end - start) @ reciprocal))
        starts[(start_label, end_label)] = offset
        ticks.extend([offset, offset + length])
        labels.extend([start_label.replace("GAMMA", "Gamma"), end_label.replace("GAMMA", "Gamma")])
        offset += length
    return starts, ticks, labels


def _plot_seekpath(workflow: FrozenMagnonWorkflow, show_ft: bool, show_lsq: bool, dense_points: int, moment: float | None) -> Any:
    import plotly.graph_objects as go

    if workflow.transform_result is None:
        return _empty_figure("Run the transform first")
    assert workflow.structure is not None
    assert workflow.energy_table is not None

    transformer = FrozenMagnonTransformer(workflow.config.theta, workflow.config.e0)
    dft_y = transformer.dft_spectrum_mry(workflow.energy_table.q, workflow.energy_table.energy_hartree)
    ylabel = r"$[E(\mathbf{q})-E0]/\sin^2(\theta)$ (mRy)"
    if moment is not None:
        dft_y = (4.0 / moment) * dft_y * MRY_TO_MEV
        ylabel = f"$4[J(0)-J(\mathbf{{q}})]/{moment:g}$ (meV)"

    points, segments, bravais = SeekPath(workflow.structure, workflow.config.symprec).get()
    reciprocal = reciprocal_lattice_rows(workflow.structure.lattice_angstrom)
    starts, ticks, labels = _path_axis(points, segments, reciprocal)
    fig = go.Figure()
    plotted = 0

    for start_label, end_label in segments:
        start = points[start_label]
        end = points[end_label]
        length = float(np.linalg.norm((end - start) @ reciprocal))
        offset = starts[(start_label, end_label)]
        segment_points = find_existing_points_on_segment(workflow.energy_table.q, dft_y, start, end, 1e-7)
        if len(segment_points) >= 2:
            x = np.asarray([offset + t * length for t, _, _ in segment_points])
            y = np.asarray([value for _, _, value in segment_points])
            q_existing = np.asarray([q for _, q, _ in segment_points])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    line={"color": "rgba(37,99,235,0.50)", "width": 2},
                    marker={"size": 5, "color": "rgba(37,99,235,0.60)"},
                    customdata=q_existing,
                    hovertemplate="q=(%{customdata[0]:.5f}, %{customdata[1]:.5f}, %{customdata[2]:.5f})<br>%{y:.6f}<extra>DFT</extra>",
                    name="DFT frozen magnon" if plotted == 0 else None,
                    showlegend=plotted == 0,
                )
            )
            plotted += 1

        q_line = np.asarray([q for _, q in make_seekpath_line(start, end, dense_points)], dtype=float)
        x_line = np.asarray([offset + t * length for t, _ in make_seekpath_line(start, end, dense_points)], dtype=float)
        if show_ft:
            ft_y = ExchangeSpectrum.from_jij(q_line, workflow.transform_result.vectors, workflow.transform_result.jij_mry)
            ft_y, ft_label, _ = ExchangeSpectrum.scale(ft_y, moment)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=ft_y,
                    mode="lines",
                    line={"color": "#dc2626", "width": 2},
                    hovertemplate="%{y:.6f}<extra>Full FT</extra>",
                    name=ft_label,
                    showlegend=(start_label, end_label) == segments[0],
                )
            )
        if show_lsq and workflow.lsq_result is not None:
            lsq_y = workflow.lsq_result.spectrum(q_line)
            lsq_y, _, _ = ExchangeSpectrum.scale(lsq_y, moment)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=lsq_y,
                    mode="lines",
                    line={"color": "#16a34a", "width": 2, "dash": "dot"},
                    hovertemplate="%{y:.6f}<extra>LSQ</extra>",
                    name=f"LSQ ({len(workflow.lsq_result.shells)} shells)",
                    showlegend=(start_label, end_label) == segments[0],
                )
            )

    if plotted == 0 and not show_ft and not (show_lsq and workflow.lsq_result is not None):
        return _empty_figure("No Seekpath segment contains enough existing DFT q-points")

    for tick in ticks:
        fig.add_vline(x=tick, line_width=1, line_color="#e5e7eb")
    fig.update_layout(
        title={"text": f"Energy / magnon spectrum in original Cell ({bravais})", "font": {"size": 16}},
        template="plotly_white",
        autosize=True,
        margin={"l": 48, "r": 8, "t": 26, "b": 28},
        xaxis={"title": "Wave-vector distance (1/A)", "tickmode": "array", "tickvals": ticks, "ticktext": labels},
        yaxis_title=ylabel,
        legend={"orientation": "h", "y": 0.99, "x": 0.01, "xanchor": "left", "yanchor": "top", "bgcolor": "rgba(255,255,255,0.76)", "font": {"size": 10}},
    )
    return fig


def _plot_jij(workflow: FrozenMagnonWorkflow, show_lsq: bool, scale_r2: bool = False) -> Any:
    import plotly.graph_objects as go

    if workflow.transform_result is None or workflow.structure is None:
        return _empty_figure("Run the transform first")
    result = workflow.transform_result
    distances = np.linalg.norm(result.vectors @ workflow.structure.lattice_angstrom, axis=1)
    scale = distances**2 if scale_r2 else np.ones_like(distances)
    y_values = result.jij_mry.real * scale
    y_label = r"$J(R) \cdot R^2$ (mRy A$^2$)" if scale_r2 else r"$J(R)$ (mRy)"
    custom = np.column_stack([result.vectors, result.jij_mry.real, distances])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distances,
            y=y_values,
            mode="markers",
            marker={"size": 7, "color": "#4f46e5", "opacity": 0.82},
            customdata=custom,
            hovertemplate="R=(%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>distance=%{customdata[4]:.4f} A<br>J=%{customdata[3]:.6f} mRy<br>plotted=%{y:.6f}<extra>Full FT</extra>",
            name="Full FT J(R)",
        )
    )
    if show_lsq and workflow.lsq_result is not None:
        fig.add_trace(
            go.Scatter(
                x=[shell.distance_angstrom for shell in workflow.lsq_result.shells],
                y=workflow.lsq_result.coefficients_mry * (np.asarray([shell.distance_angstrom for shell in workflow.lsq_result.shells]) ** 2 if scale_r2 else 1.0),
                mode="markers+lines",
                marker={"size": 9, "symbol": "diamond", "color": "#16a34a"},
                line={"dash": "dot", "color": "#16a34a"},
                hovertemplate="shell distance=%{x:.4f} A<br>plotted=%{y:.6f}<extra>LSQ</extra>",
                name="LSQ shells",
            )
        )
    fig.add_hline(y=0.0, line_width=1, line_color="#d1d5db")
    fig.update_layout(
        title={"text": "Real-Space Exchange J(R)" + (" * R^2" if scale_r2 else ""), "font": {"size": 16}},
        template="plotly_white",
        autosize=True,
        margin={"l": 48, "r": 8, "t": 26, "b": 28},
        xaxis_title="Distance (A)",
        yaxis_title=y_label,
        legend={"orientation": "h", "y": 0.99, "x": 0.01, "xanchor": "left", "yanchor": "top", "bgcolor": "rgba(255,255,255,0.76)", "font": {"size": 10}},
    )
    return fig


def _run_path_analysis(config: PathAnalysisConfig) -> FrozenMagnonWorkflow:
    """Run an analysis from local file paths for CLI-launched dashboard sessions."""
    workflow = FrozenMagnonWorkflow(
        WorkflowConfig(
            energy_path=config.energy_path,
            elk_path=config.elk_path,
            vectors_path=config.vectors_path,
            rmax=config.rmax,
            theta=config.theta,
            symmetry=config.symmetry,
            e0=config.e0,
            output_prefix=Path.cwd() / "fourj_dashboard",
        )
    )
    workflow.run_transform()
    if config.fit_lsq:
        workflow.fit_lsq(max_shells=config.fit_shells)
    return workflow


def _run_uploaded_analysis(
    elk_contents: str | None,
    elk_filename: str | None,
    energy_contents: str | None,
    energy_filename: str | None,
    jfile_contents: str | None,
    jfile_filename: str | None,
    theta: float,
    symmetry: str,
    e0: str,
    rmax: float | None,
    fit_lsq: list[str] | None,
    fit_shells: int | None,
) -> FrozenMagnonWorkflow:
    elk = _decode_upload(elk_contents, elk_filename, "elk input")
    energy = _decode_upload(energy_contents, energy_filename, "energy_vs_q.dat")
    jfile = _decode_upload(jfile_contents, jfile_filename, "jfile") if jfile_contents else None

    tmp = tempfile.TemporaryDirectory(prefix="fourj-dashboard-")
    tmp_path = Path(tmp.name)
    elk_path = tmp_path / (elk.filename or "elk.tmp")
    energy_path = tmp_path / (energy.filename or "energy_vs_q.dat")
    elk_path.write_text(elk.text)
    energy_path.write_text(energy.text)
    vectors_path = None
    if jfile is not None:
        vectors_path = tmp_path / (jfile.filename or "jfile")
        vectors_path.write_text(jfile.text)

    workflow = FrozenMagnonWorkflow(
        WorkflowConfig(
            energy_path=energy_path,
            elk_path=elk_path,
            vectors_path=vectors_path,
            rmax=rmax,
            theta=float(theta),
            symmetry=symmetry,
            e0=e0,
            output_prefix=tmp_path / "fourj_dashboard",
        )
    )
    workflow._dashboard_tmp = tmp  # type: ignore[attr-defined]
    workflow.run_transform()
    if fit_lsq and "fit" in fit_lsq:
        workflow.fit_lsq(max_shells=fit_shells)
    return workflow


def _status_panel(workflow: FrozenMagnonWorkflow) -> list[Any]:
    from dash import html

    assert workflow.energy_table is not None
    assert workflow.transform_result is not None
    rows = [
        ("Raw q-points", len(workflow.energy_table.q)),
        ("Symmetrized q samples", len(workflow.transform_result.q)),
        ("R vectors", len(workflow.transform_result.vectors)),
        ("R source", workflow.vector_source or "unknown"),
        ("Symmetry mode", workflow.symmetry_used or workflow.config.symmetry),
    ]
    rows.extend(_structure_metadata(workflow).items())
    rows.append(("Max |Im J|", f"{workflow.max_imag_jij():.3e} mRy"))
    if workflow.lsq_result is not None:
        rows.extend(
            [
                ("LSQ shells", len(workflow.lsq_result.shells)),
                ("LSQ RMSE", f"{workflow.lsq_result.rmse_mry:.4g} mRy"),
                ("LSQ max error", f"{workflow.lsq_result.max_abs_error_mry:.4g} mRy"),
            ]
        )
    return [html.Div([html.Span(label), html.Strong(str(value))], className="metric-row") for label, value in rows]


def create_app(initial_config: PathAnalysisConfig | None = None) -> Any:
    """Create the FourJ Dash application."""
    try:
        from dash import Dash, Input, Output, State, dcc, html, no_update
    except Exception as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError("Install dashboard dependencies with `pip install -e .[dashboard]`") from exc

    app = Dash(__name__, title="FourJ Dashboard")
    initial_workflow = None
    initial_error = None
    if initial_config is not None:
        try:
            initial_workflow = _run_path_analysis(initial_config)
        except Exception as exc:  # pragma: no cover - startup validation
            initial_error = str(exc)
    upload_style = {
        "border": "1px dashed #94a3b8",
        "borderRadius": "6px",
        "padding": "12px",
        "textAlign": "center",
        "cursor": "pointer",
        "background": "#f8fafc",
    }
    if initial_workflow is None:
        initial_bz = _empty_figure("Upload inputs and run analysis" if initial_error is None else "Initial analysis failed")
        initial_energy = _empty_figure("Upload inputs and run analysis" if initial_error is None else "Initial analysis failed")
        initial_path = _empty_figure("Upload inputs and run analysis" if initial_error is None else "Initial analysis failed")
        initial_jij = _empty_figure("Upload inputs and run analysis" if initial_error is None else "Initial analysis failed")
        initial_status: Any = [] if initial_error is None else __import__("dash").html.Div(initial_error, className="error")
    else:
        initial_bz = _plot_reciprocal_cell(initial_workflow)
        initial_energy = _plot_input_energy(initial_workflow)
        initial_path = _plot_seekpath(initial_workflow, True, initial_workflow.lsq_result is not None, initial_config.dense_points if initial_config else 400, initial_config.moment if initial_config else None)
        initial_jij = _plot_jij(initial_workflow, initial_workflow.lsq_result is not None, False)
        initial_status = _status_panel(initial_workflow)

    app.layout = html.Div(
        [
            dcc.Store(id="initial-config", data={
                "elk_path": str(initial_config.elk_path) if initial_config else None,
                "energy_path": str(initial_config.energy_path) if initial_config else None,
                "vectors_path": str(initial_config.vectors_path) if initial_config and initial_config.vectors_path else None,
            }),
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(src=app.get_asset_url("fourj.png"), className="logo") if Path("assets/fourj.png").exists() else html.Div("FourJ", className="wordmark"),
                            html.Div([html.H1("FourJ Dashboard"), html.P("Interactive frozen-magnon exchange analysis from Elk spin-spiral data.")]),
                        ],
                        className="brand",
                    )
                ],
                className="topbar",
            ),
            html.Div(
                [
                    html.Main(
                        [
                            html.Section(dcc.Graph(id="bz-figure", figure=initial_bz, config={"displaylogo": False, "responsive": True}, style={"height": "100%", "width": "100%"}), className="panel"),
                            html.Section(dcc.Graph(id="energy-figure", figure=initial_energy, config={"displaylogo": False, "responsive": True}, style={"height": "100%", "width": "100%"}), className="panel"),
                            html.Section(dcc.Graph(id="path-figure", figure=initial_path, config={"displaylogo": False, "responsive": True}, style={"height": "100%", "width": "100%"}), className="panel"),
                            html.Section(dcc.Graph(id="jij-figure", figure=initial_jij, config={"displaylogo": False, "responsive": True}, style={"height": "100%", "width": "100%"}), className="panel"),
                        ],
                        className="grid",
                    ),
                    html.Aside(
                        [
                            html.H2("Inputs"),
                            dcc.Upload(id="elk-upload", children=html.Div(["Drop or select ", html.Strong("elk.in / elk.tmp")]), style=upload_style),
                            html.Div(initial_config.elk_path.name if initial_config else "", id="elk-name", className="file-name"),
                            dcc.Upload(id="energy-upload", children=html.Div(["Drop or select ", html.Strong("energy_vs_q.dat")]), style=upload_style),
                            html.Div(initial_config.energy_path.name if initial_config else "", id="energy-name", className="file-name"),
                            dcc.Upload(id="jfile-upload", children=html.Div(["Optional ", html.Strong("jfile"), " vectors"]), style=upload_style),
                            html.Div(initial_config.vectors_path.name if initial_config and initial_config.vectors_path else "", id="jfile-name", className="file-name"),
                            html.H2("Controls"),
                            html.Label("Theta (degrees)"),
                            dcc.Input(id="theta", type="number", value=initial_config.theta if initial_config else 90.0, step=1.0, className="control"),
                            html.Label("Symmetry"),
                            dcc.Dropdown(id="symmetry", options=[{"label": x, "value": x} for x in ["none", "time-reversal", "lattice", "spglib"]], value=initial_config.symmetry if initial_config else "spglib", clearable=False),
                            html.Label("E0 reference"),
                            dcc.Dropdown(id="e0", options=[{"label": x, "value": x} for x in ["q0", "min"]], value=initial_config.e0 if initial_config else "q0", clearable=False),
                            html.Label("R cutoff if no jfile (A)"),
                            dcc.Input(id="rmax", type="number", value=initial_config.rmax if initial_config else None, placeholder="auto: half inferred mesh max", step=0.5, className="control"),
                            dcc.Checklist(id="fit-lsq", options=[{"label": "Fit LSQ shells", "value": "fit"}], value=["fit"] if initial_config and initial_config.fit_lsq else [], className="checklist"),
                            html.Label("Number of LSQ shells"),
                            dcc.Input(id="fit-shells", type="number", value=initial_config.fit_shells if initial_config and initial_config.fit_shells else 2, min=1, step=1, className="control"),
                            dcc.Checklist(
                                id="overlays",
                                options=[{"label": "Show FT spectrum", "value": "ft"}, {"label": "Show LSQ spectrum", "value": "lsq"}],
                                value=["ft", "lsq"],
                                className="checklist",
                            ),
                            dcc.Checklist(
                                id="jij-options",
                                options=[{"label": "Plot J(R) * R^2", "value": "r2"}],
                                value=[],
                                className="checklist",
                            ),
                            html.Label("LSWT moment M (mu_B)"),
                            dcc.Input(id="moment", type="number", value=initial_config.moment if initial_config else None, placeholder="optional", step=0.1, className="control"),
                            html.Label("Dense path points/segment"),
                            dcc.Input(id="dense-points", type="number", value=initial_config.dense_points if initial_config else 400, min=2, step=1, className="control"),
                            html.Button("Run Analysis", id="run-button", n_clicks=0, className="run-button"),
                            html.Div(initial_status, id="status", className="status"),
                        ],
                        className="sidebar",
                    ),
                ],
                className="content",
            ),
        ],
        className="app-shell",
    )

    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #eef2f7; color: #111827; }
                .topbar { padding: 10px 16px; background: #0f172a; color: white; border-bottom: 1px solid #1e293b; }
                .brand { display: flex; align-items: center; gap: 10px; }
                .brand h1 { margin: 0; font-size: 20px; letter-spacing: 0; }
                .brand p { margin: 1px 0 0; color: #cbd5e1; font-size: 12px; }
                .logo { width: 40px; height: 40px; object-fit: contain; }
                .wordmark { font-size: 22px; font-weight: 800; color: #7dd3fc; }
                .content { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 10px; padding: 10px; }
                .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-auto-rows: 40vh; gap: 10px; align-items: start; }
                .panel { background: white; border: 1px solid #d9e2ec; border-radius: 8px; height: 40vh; min-height: 260px; max-height: 360px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); overflow: hidden; }
                .panel .dash-graph, .panel .js-plotly-plot, .panel .plot-container, .panel .svg-container { height: 100% !important; width: 100% !important; }
                .sidebar { background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); align-self: start; position: sticky; top: 12px; }
                .sidebar h2 { font-size: 14px; margin: 2px 0 8px; color: #0f172a; }
                .sidebar label { display: block; margin: 8px 0 4px; font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase; }
                .control, .Select-control { width: 100%; box-sizing: border-box; }
                .control { height: 32px; padding: 5px 8px; border: 1px solid #cbd5e1; border-radius: 5px; font-size: 13px; }
                .checklist { margin: 8px 0; font-size: 13px; }
                .file-name { min-height: 15px; margin: 3px 0 7px; color: #475569; font-size: 11px; }
                .run-button { width: 100%; height: 36px; margin-top: 10px; border: 0; border-radius: 6px; background: #2563eb; color: white; font-size: 14px; font-weight: 700; cursor: pointer; }
                .run-button:hover { background: #1d4ed8; }
                .status { margin-top: 10px; font-size: 12px; color: #334155; }
                .metric-row { display: flex; justify-content: space-between; gap: 10px; padding: 4px 0; border-bottom: 1px solid #e2e8f0; }
                .metric-row strong { color: #0f172a; text-align: right; }
                .error { color: #b91c1c; background: #fef2f2; border: 1px solid #fecaca; padding: 10px; border-radius: 6px; }
                @media (max-width: 1200px) { .content { grid-template-columns: 1fr; } .sidebar { position: static; } }
                @media (max-width: 860px) { .grid { grid-template-columns: 1fr; grid-auto-rows: 38vh; } .panel { height: 38vh; min-height: 300px; max-height: none; } }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>{%config%}{%scripts%}{%renderer%}</footer>
        </body>
    </html>
    """

    @app.callback(Output("elk-name", "children"), Input("elk-upload", "filename"))
    def _elk_name(filename: str | None) -> str:
        return filename or (initial_config.elk_path.name if initial_config else "")

    @app.callback(Output("energy-name", "children"), Input("energy-upload", "filename"))
    def _energy_name(filename: str | None) -> str:
        return filename or (initial_config.energy_path.name if initial_config else "")

    @app.callback(Output("jfile-name", "children"), Input("jfile-upload", "filename"))
    def _jfile_name(filename: str | None) -> str:
        return filename or (initial_config.vectors_path.name if initial_config and initial_config.vectors_path else "")

    @app.callback(
        Output("bz-figure", "figure"),
        Output("energy-figure", "figure"),
        Output("path-figure", "figure"),
        Output("jij-figure", "figure"),
        Output("status", "children"),
        Input("run-button", "n_clicks"),
        State("elk-upload", "contents"),
        State("elk-upload", "filename"),
        State("energy-upload", "contents"),
        State("energy-upload", "filename"),
        State("jfile-upload", "contents"),
        State("jfile-upload", "filename"),
        State("theta", "value"),
        State("symmetry", "value"),
        State("e0", "value"),
        State("rmax", "value"),
        State("fit-lsq", "value"),
        State("fit-shells", "value"),
        State("overlays", "value"),
        State("jij-options", "value"),
        State("moment", "value"),
        State("dense-points", "value"),
        State("initial-config", "data"),
        prevent_initial_call=True,
    )
    def _run(
        n_clicks: int,
        elk_contents: str | None,
        elk_filename: str | None,
        energy_contents: str | None,
        energy_filename: str | None,
        jfile_contents: str | None,
        jfile_filename: str | None,
        theta: float,
        symmetry: str,
        e0: str,
        rmax: float | None,
        fit_lsq: list[str] | None,
        fit_shells: int | None,
        overlays: list[str] | None,
        jij_options: list[str] | None,
        moment: float | None,
        dense_points: int | None,
        initial_data: dict[str, str | None] | None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update
        try:
            if elk_contents or energy_contents:
                workflow = _run_uploaded_analysis(
                    elk_contents,
                    elk_filename,
                    energy_contents,
                    energy_filename,
                    jfile_contents,
                    jfile_filename,
                    theta,
                    symmetry,
                    e0,
                    rmax,
                    fit_lsq,
                    fit_shells,
                )
            elif initial_data and initial_data.get("elk_path") and initial_data.get("energy_path"):
                workflow = _run_path_analysis(
                    PathAnalysisConfig(
                        elk_path=Path(str(initial_data["elk_path"])),
                        energy_path=Path(str(initial_data["energy_path"])),
                        vectors_path=Path(str(initial_data["vectors_path"])) if initial_data.get("vectors_path") else None,
                        theta=float(theta),
                        symmetry=symmetry,
                        e0=e0,
                        rmax=rmax,
                        fit_lsq=bool(fit_lsq and "fit" in fit_lsq),
                        fit_shells=fit_shells,
                        moment=float(moment) if moment else None,
                        dense_points=int(dense_points or 400),
                    )
                )
            else:
                workflow = _run_uploaded_analysis(
                    elk_contents,
                    elk_filename,
                    energy_contents,
                    energy_filename,
                    jfile_contents,
                    jfile_filename,
                    theta,
                    symmetry,
                    e0,
                    rmax,
                    fit_lsq,
                    fit_shells,
                )
            overlays = overlays or []
            return (
                _plot_reciprocal_cell(workflow),
                _plot_input_energy(workflow),
                _plot_seekpath(workflow, "ft" in overlays, "lsq" in overlays, int(dense_points or 400), float(moment) if moment else None),
                _plot_jij(workflow, "lsq" in overlays, bool(jij_options and "r2" in jij_options)),
                _status_panel(workflow),
            )
        except Exception as exc:
            from dash import html

            error = html.Div(str(exc), className="error")
            return _empty_figure("Analysis failed"), _empty_figure("Analysis failed"), _empty_figure("Analysis failed"), _empty_figure("Analysis failed"), error

    return app


def main(argv: list[str] | None = None) -> int:
    """Run the FourJ dashboard server."""
    parser = argparse.ArgumentParser(description="Run the FourJ Plotly dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--energy", type=Path, default=None, help="Optional initial energy_vs_q.dat path")
    parser.add_argument("--elk", type=Path, default=None, help="Optional initial Elk input/tmp path")
    parser.add_argument("--vectors", type=Path, default=None, help="Optional initial jfile-like R-vector path")
    parser.add_argument("--rmax", type=float, default=None, help="Optional initial R cutoff in Angstrom")
    parser.add_argument("--theta", type=float, default=90.0)
    parser.add_argument("--symmetry", choices=("none", "time-reversal", "lattice", "spglib"), default="spglib")
    parser.add_argument("--e0", choices=("q0", "min"), default="q0")
    parser.add_argument("--fit-lsq", action="store_true")
    parser.add_argument("--fit-num-shells", type=int, default=None)
    parser.add_argument("--lswt-moment", type=float, default=None)
    parser.add_argument("--dense-points", type=int, default=400)
    args = parser.parse_args(argv)
    initial_config = None
    if args.energy is not None or args.elk is not None:
        if args.energy is None or args.elk is None:
            raise SystemExit("--energy and --elk must be given together for initial dashboard analysis")
        initial_config = PathAnalysisConfig(
            elk_path=args.elk,
            energy_path=args.energy,
            vectors_path=args.vectors,
            theta=args.theta,
            symmetry=args.symmetry,
            e0=args.e0,
            rmax=args.rmax,
            fit_lsq=args.fit_lsq,
            fit_shells=args.fit_num_shells,
            moment=args.lswt_moment,
            dense_points=args.dense_points,
        )
    app = create_app(initial_config)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
