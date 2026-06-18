"""Command-line interface for the FourJ package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .workflow import FrozenMagnonWorkflow, WorkflowConfig


def structure_metadata(workflow: FrozenMagnonWorkflow) -> list[tuple[str, str]]:
    """Return best-effort structure labels for CLI output."""
    rows: list[tuple[str, str]] = []
    if workflow.structure is None:
        return rows
    try:
        from .visualization import SeekPath

        _points, _segments, bravais = SeekPath(workflow.structure, workflow.config.symprec).get()
        rows.append(("Bravais", bravais))
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
                rows.append(("Space group", f"{international} ({number})"))
            elif international:
                rows.append(("Space group", str(international)))
            if hall:
                rows.append(("Hall", str(hall)))
    except Exception:
        pass
    return rows


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FourJ E(q) -> J(q) -> J_ij transform for Elk spin spirals.")
    parser.add_argument("--energy", default="energy_vs_q.dat", type=Path, help="Input E(q) table")
    parser.add_argument("--elk", default="elk.tmp", type=Path, help="Elk input/tmp file with scale, avec, atoms")
    parser.add_argument("--vectors", type=Path, default=None, help="Optional jfile-like R-vector list; columns 3-5 are integer direct-lattice R. Defaults to ./jfile if present; otherwise infer from q mesh unless --rmax is set.")
    parser.add_argument("--rmax", type=float, default=None, help="Generate all integer direct-lattice R vectors up to this distance in Angstrom when no vector file is given")
    parser.add_argument("--theta", type=float, default=90.0, help="Cone angle in degrees")
    parser.add_argument("--symmetry", choices=("none", "time-reversal", "lattice", "spglib"), default="time-reversal", help="q-space symmetry averaging/completion mode")
    parser.add_argument("--e0", choices=("q0", "min"), default="q0", help="Reference energy for E(q)-E0")
    parser.add_argument("--output-prefix", default="fourj", type=Path)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--round-decimals", type=int, default=10)
    parser.add_argument("--symprec", type=float, default=1e-5, help="spglib symmetry tolerance")
    parser.add_argument("--no-realspace-pair-symmetry", action="store_true", help="Do not enforce J(R)=conj[J(-R)]")
    parser.add_argument("--gui", action="store_true", help="Launch the Plotly dashboard preloaded with these CLI input paths and settings.")
    parser.add_argument("--gui-host", default="127.0.0.1", help="Host for --gui dashboard server")
    parser.add_argument("--gui-port", default=8050, type=int, help="Port for --gui dashboard server")
    parser.add_argument("--gui-debug", action="store_true", help="Run --gui dashboard in Dash debug mode")

    parser.add_argument("--plot-path", action="store_true", help="Plot an E(q)-derived spectrum along Seekpath symmetry segments.")
    parser.add_argument("--plot-kind", choices=("magnon", "energy"), default="magnon")
    parser.add_argument("--plot-tol", type=float, default=1e-7)
    parser.add_argument("--plot-min-points", type=int, default=2)
    parser.add_argument("--plot-lswt", action="store_true", help="Overlay J(0)-J(q) reconstructed from extracted J_ij.")
    parser.add_argument("--lswt-moment", type=float, default=None, help="If set, plot 4[J(0)-J(q)]/M in meV.")
    parser.add_argument("--lswt-dense-path", action="store_true", help="Evaluate LSWT/exchange overlay on a dense Seekpath line.")
    parser.add_argument("--lswt-path-points", type=int, default=202)

    parser.add_argument("--fit-lsq", action="store_true", help="Fit J(0)-J(q) to symmetry-grouped real-space exchange shells.")
    parser.add_argument("--fit-num-shells", type=int, default=None, help="Use only this many shortest shells, e.g. 2 for NN+NNN.")
    parser.add_argument("--fit-rmax", type=float, default=None)
    parser.add_argument("--fit-distance-tol", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.gui:
        from .dashboard import PathAnalysisConfig, create_app

        app = create_app(
            PathAnalysisConfig(
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
                dense_points=args.lswt_path_points,
            )
        )
        app.run(host=args.gui_host, port=args.gui_port, debug=args.gui_debug)
        return 0

    workflow = FrozenMagnonWorkflow(
        WorkflowConfig(
            energy_path=args.energy,
            elk_path=args.elk,
            vectors_path=args.vectors,
            rmax=args.rmax,
            theta=args.theta,
            symmetry=args.symmetry,
            e0=args.e0,
            output_prefix=args.output_prefix,
            tol=args.tol,
            round_decimals=args.round_decimals,
            symprec=args.symprec,
            realspace_pair_symmetry=not args.no_realspace_pair_symmetry,
        )
    )
    result = workflow.run_transform()
    jij_path, dist_path, jq_path = workflow.write_transform_outputs()

    print(f"Read {len(workflow.energy_table.q)} raw q points from {args.energy}")
    print(f"Using {len(result.q)} symmetry-averaged q samples; symmetry mode: {workflow.symmetry_used}")
    print(f"Read/generated {len(result.vectors)} R vectors")
    print(f"R vectors are integer direct-lattice coordinates; source: {workflow.vector_source}")
    for label, value in structure_metadata(workflow):
        print(f"{label}: {value}")
    print(f"Wrote {jij_path}")
    print(f"Wrote {dist_path}")
    print(f"Wrote {jq_path}")
    print(f"Max |Im J_ij| after selected symmetrization: {workflow.max_imag_jij():.6e} mRy")

    if args.fit_lsq:
        lsq_result = workflow.fit_lsq(args.fit_num_shells, args.fit_rmax, args.fit_distance_tol)
        shell_path, fit_path = workflow.write_lsq_outputs()
        print(f"LSQ fit used {len(lsq_result.shells)} symmetry shells; rotation mode: {lsq_result.rotation_source}")
        print(f"LSQ RMSE: {lsq_result.rmse_mry:.6f} mRy; max |error|: {lsq_result.max_abs_error_mry:.6f} mRy")
        print(f"Wrote {shell_path}")
        print(f"Wrote {fit_path}")

    if args.plot_path:
        plot_path, data_path, skipped_segments = workflow.plot_seekpath(
            plot_kind=args.plot_kind,
            plot_lswt=args.plot_lswt,
            lswt_moment=args.lswt_moment,
            dense_path=args.lswt_dense_path,
            path_points=args.lswt_path_points,
            tol=args.plot_tol,
            min_points=args.plot_min_points,
        )
        print(f"Wrote {plot_path}")
        print(f"Wrote {data_path}")
        if args.plot_lswt and args.lswt_dense_path:
            print(f"Wrote {args.output_prefix.with_suffix(f'.seekpath_{args.plot_kind}_lswt_dense.dat')}")
        if skipped_segments:
            print("Skipped Seekpath segments without enough existing q points: " + ", ".join(skipped_segments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
