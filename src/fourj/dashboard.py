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
from .io import EnergyTable
from .transforms import ExchangeSpectrum, FrozenMagnonTransformer
from .visualization import SeekPath, find_existing_points_on_segment, make_seekpath_line, reciprocal_lattice_rows
from .workflow import FrozenMagnonWorkflow, WorkflowConfig


@dataclass(frozen=True)
class UploadedText:
    """Decoded Dash upload content."""

    filename: str
    text: str


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
    fig.update_layout(template="plotly_white", margin={"l": 24, "r": 24, "t": 40, "b": 24})
    return fig


def _plot_reciprocal_cell(workflow: FrozenMagnonWorkflow) -> Any:
    import plotly.graph_objects as go

    assert workflow.structure is not None
    assert workflow.energy_table is not None
    reciprocal = reciprocal_lattice_rows(workflow.structure.lattice_angstrom)
    corners_frac = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=float,
    ) - 0.5
    corners = corners_frac @ reciprocal
    q_cart = (np.mod(workflow.energy_table.q + 0.5, 1.0) - 0.5) @ reciprocal
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=corners[:, 0],
            y=corners[:, 1],
            z=corners[:, 2],
            i=[0, 0, 4, 4, 0, 2, 1, 3, 0, 1, 2, 3],
            j=[1, 2, 5, 6, 1, 3, 5, 7, 2, 3, 6, 7],
            k=[2, 3, 6, 7, 4, 6, 4, 5, 4, 5, 4, 5],
            color="#7f8c8d",
            opacity=0.16,
            name="reciprocal cell",
            hoverinfo="skip",
            showscale=False,
        )
    )
    for a, b in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[corners[a, 0], corners[b, 0]],
                y=[corners[a, 1], corners[b, 1]],
                z=[corners[a, 2], corners[b, 2]],
                mode="lines",
                line={"color": "#5f6c72", "width": 3},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=q_cart[:, 0],
            y=q_cart[:, 1],
            z=q_cart[:, 2],
            mode="markers",
            marker={"size": 4, "color": np.arange(len(q_cart)), "colorscale": "Viridis", "opacity": 0.9},
            customdata=workflow.energy_table.q,
            hovertemplate="q=(%{customdata[0]:.5f}, %{customdata[1]:.5f}, %{customdata[2]:.5f})<extra></extra>",
            name="input q-points",
        )
    )
    fig.update_layout(
        title="Reciprocal Cell and Input q-Points",
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        scene={
            "xaxis_title": "kx (1/A)",
            "yaxis_title": "ky (1/A)",
            "zaxis_title": "kz (1/A)",
            "aspectmode": "data",
        },
        legend={"orientation": "h", "y": 1.02, "x": 0},
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
        title="Input E(q) vs q-Index",
        template="plotly_white",
        margin={"l": 56, "r": 24, "t": 44, "b": 48},
        xaxis_title="Input q-index",
        yaxis_title="E(q)-E0 (mRy)",
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
        labels.extend([start_label.replace("GAMMA", "Γ"), end_label.replace("GAMMA", "Γ")])
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
    ylabel = "[E(q)-E0]/sin²θ (mRy)"
    if moment is not None:
        dft_y = (4.0 / moment) * dft_y * MRY_TO_MEV
        ylabel = f"4[J(0)-J(q)]/{moment:g} (meV)"

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
        title=f"Seekpath Spectrum ({bravais})",
        template="plotly_white",
        margin={"l": 60, "r": 24, "t": 44, "b": 54},
        xaxis={"title": "Wave-vector distance (1/A)", "tickmode": "array", "tickvals": ticks, "ticktext": labels},
        yaxis_title=ylabel,
        legend={"orientation": "h", "y": 1.10, "x": 0},
    )
    return fig


def _plot_jij(workflow: FrozenMagnonWorkflow, show_lsq: bool) -> Any:
    import plotly.graph_objects as go

    if workflow.transform_result is None or workflow.structure is None:
        return _empty_figure("Run the transform first")
    result = workflow.transform_result
    distances = np.linalg.norm(result.vectors @ workflow.structure.lattice_angstrom, axis=1)
    custom = result.vectors
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=distances,
            y=result.jij_mry.real,
            mode="markers",
            marker={"size": 7, "color": "#4f46e5", "opacity": 0.82},
            customdata=custom,
            hovertemplate="R=(%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<br>distance=%{x:.4f} A<br>J=%{y:.6f} mRy<extra>Full FT</extra>",
            name="Full FT J(R)",
        )
    )
    if show_lsq and workflow.lsq_result is not None:
        fig.add_trace(
            go.Scatter(
                x=[shell.distance_angstrom for shell in workflow.lsq_result.shells],
                y=workflow.lsq_result.coefficients_mry,
                mode="markers+lines",
                marker={"size": 9, "symbol": "diamond", "color": "#16a34a"},
                line={"dash": "dot", "color": "#16a34a"},
                hovertemplate="shell distance=%{x:.4f} A<br>J_shell=%{y:.6f} mRy<extra>LSQ</extra>",
                name="LSQ shells",
            )
        )
    fig.add_hline(y=0.0, line_width=1, line_color="#d1d5db")
    fig.update_layout(
        title="Real-Space Exchange J(R)",
        template="plotly_white",
        margin={"l": 60, "r": 24, "t": 44, "b": 54},
        xaxis_title="Distance (A)",
        yaxis_title="J(R) (mRy)",
        legend={"orientation": "h", "y": 1.10, "x": 0},
    )
    return fig


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
        ("Symmetry", workflow.symmetry_used or workflow.config.symmetry),
        ("Max |Im J|", f"{workflow.max_imag_jij():.3e} mRy"),
    ]
    if workflow.lsq_result is not None:
        rows.extend(
            [
                ("LSQ shells", len(workflow.lsq_result.shells)),
                ("LSQ RMSE", f"{workflow.lsq_result.rmse_mry:.4g} mRy"),
                ("LSQ max error", f"{workflow.lsq_result.max_abs_error_mry:.4g} mRy"),
            ]
        )
    return [html.Div([html.Span(label), html.Strong(str(value))], className="metric-row") for label, value in rows]


def create_app() -> Any:
    """Create the FourJ Dash application."""
    try:
        from dash import Dash, Input, Output, State, dcc, html, no_update
    except Exception as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError("Install dashboard dependencies with `pip install -e .[dashboard]`") from exc

    app = Dash(__name__, title="FourJ Dashboard")
    upload_style = {
        "border": "1px dashed #94a3b8",
        "borderRadius": "6px",
        "padding": "12px",
        "textAlign": "center",
        "cursor": "pointer",
        "background": "#f8fafc",
    }
    app.layout = html.Div(
        [
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
                            html.Section(dcc.Graph(id="bz-figure", figure=_empty_figure("Upload inputs and run analysis"), config={"displaylogo": False}), className="panel"),
                            html.Section(dcc.Graph(id="energy-figure", figure=_empty_figure("Upload inputs and run analysis"), config={"displaylogo": False}), className="panel"),
                            html.Section(dcc.Graph(id="path-figure", figure=_empty_figure("Upload inputs and run analysis"), config={"displaylogo": False}), className="panel"),
                            html.Section(dcc.Graph(id="jij-figure", figure=_empty_figure("Upload inputs and run analysis"), config={"displaylogo": False}), className="panel"),
                        ],
                        className="grid",
                    ),
                    html.Aside(
                        [
                            html.H2("Inputs"),
                            dcc.Upload(id="elk-upload", children=html.Div(["Drop or select ", html.Strong("elk.in / elk.tmp")]), style=upload_style),
                            html.Div(id="elk-name", className="file-name"),
                            dcc.Upload(id="energy-upload", children=html.Div(["Drop or select ", html.Strong("energy_vs_q.dat")]), style=upload_style),
                            html.Div(id="energy-name", className="file-name"),
                            dcc.Upload(id="jfile-upload", children=html.Div(["Optional ", html.Strong("jfile"), " vectors"]), style=upload_style),
                            html.Div(id="jfile-name", className="file-name"),
                            html.H2("Controls"),
                            html.Label("Theta (degrees)"),
                            dcc.Input(id="theta", type="number", value=90.0, step=1.0, className="control"),
                            html.Label("Symmetry"),
                            dcc.Dropdown(id="symmetry", options=[{"label": x, "value": x} for x in ["none", "time-reversal", "lattice", "spglib"]], value="spglib", clearable=False),
                            html.Label("E0 reference"),
                            dcc.Dropdown(id="e0", options=[{"label": x, "value": x} for x in ["q0", "min"]], value="q0", clearable=False),
                            html.Label("R cutoff if no jfile (A)"),
                            dcc.Input(id="rmax", type="number", value=None, placeholder="auto", step=0.5, className="control"),
                            dcc.Checklist(id="fit-lsq", options=[{"label": "Fit LSQ shells", "value": "fit"}], value=[], className="checklist"),
                            html.Label("Number of LSQ shells"),
                            dcc.Input(id="fit-shells", type="number", value=2, min=1, step=1, className="control"),
                            dcc.Checklist(
                                id="overlays",
                                options=[{"label": "Show FT spectrum", "value": "ft"}, {"label": "Show LSQ spectrum", "value": "lsq"}],
                                value=["ft", "lsq"],
                                className="checklist",
                            ),
                            html.Label("LSWT moment M (mu_B)"),
                            dcc.Input(id="moment", type="number", value=None, placeholder="optional", step=0.1, className="control"),
                            html.Label("Dense path points/segment"),
                            dcc.Input(id="dense-points", type="number", value=404, min=2, step=10, className="control"),
                            html.Button("Run Analysis", id="run-button", n_clicks=0, className="run-button"),
                            html.Div(id="status", className="status"),
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
                .topbar { padding: 14px 18px; background: #0f172a; color: white; border-bottom: 1px solid #1e293b; }
                .brand { display: flex; align-items: center; gap: 14px; }
                .brand h1 { margin: 0; font-size: 22px; letter-spacing: 0; }
                .brand p { margin: 2px 0 0; color: #cbd5e1; font-size: 13px; }
                .logo { width: 48px; height: 48px; object-fit: contain; }
                .wordmark { font-size: 22px; font-weight: 800; color: #7dd3fc; }
                .content { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 16px; padding: 16px; }
                .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
                .panel { background: white; border: 1px solid #d9e2ec; border-radius: 8px; min-height: 420px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); overflow: hidden; }
                .panel .dash-graph { height: 420px; }
                .sidebar { background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); align-self: start; position: sticky; top: 16px; }
                .sidebar h2 { font-size: 15px; margin: 4px 0 12px; color: #0f172a; }
                .sidebar label { display: block; margin: 12px 0 5px; font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; }
                .control, .Select-control { width: 100%; box-sizing: border-box; }
                .control { height: 36px; padding: 6px 8px; border: 1px solid #cbd5e1; border-radius: 5px; font-size: 14px; }
                .checklist { margin: 12px 0; font-size: 14px; }
                .file-name { min-height: 18px; margin: 4px 0 10px; color: #475569; font-size: 12px; }
                .run-button { width: 100%; height: 40px; margin-top: 14px; border: 0; border-radius: 6px; background: #2563eb; color: white; font-size: 14px; font-weight: 700; cursor: pointer; }
                .run-button:hover { background: #1d4ed8; }
                .status { margin-top: 14px; font-size: 13px; color: #334155; }
                .metric-row { display: flex; justify-content: space-between; gap: 12px; padding: 6px 0; border-bottom: 1px solid #e2e8f0; }
                .metric-row strong { color: #0f172a; text-align: right; }
                .error { color: #b91c1c; background: #fef2f2; border: 1px solid #fecaca; padding: 10px; border-radius: 6px; }
                @media (max-width: 1200px) { .content { grid-template-columns: 1fr; } .sidebar { position: static; } }
                @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } .panel, .panel .dash-graph { min-height: 360px; height: 360px; } }
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
        return filename or ""

    @app.callback(Output("energy-name", "children"), Input("energy-upload", "filename"))
    def _energy_name(filename: str | None) -> str:
        return filename or ""

    @app.callback(Output("jfile-name", "children"), Input("jfile-upload", "filename"))
    def _jfile_name(filename: str | None) -> str:
        return filename or ""

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
        State("moment", "value"),
        State("dense-points", "value"),
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
        moment: float | None,
        dense_points: int | None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update
        try:
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
                _plot_seekpath(workflow, "ft" in overlays, "lsq" in overlays, int(dense_points or 404), float(moment) if moment else None),
                _plot_jij(workflow, "lsq" in overlays),
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
    args = parser.parse_args(argv)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
