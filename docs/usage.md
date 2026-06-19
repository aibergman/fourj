# Usage

Run from a calculation directory containing:

- `energy_vs_q.dat`
- `elk.tmp`
- optionally `jfile`

Install in editable mode before running the examples:

```bash
pip install -e /path/to/FourJ[dashboard]
```

## Full Fourier Transform

```bash
fourj \
  --energy energy_vs_q.dat \
  --elk elk.tmp \
  --vectors jfile \
  --theta 90 \
  --symmetry spglib \
  --output-prefix fourj
```


## Real-Space Vector Selection

`J(R)` vectors are integer direct-lattice coordinates. In output tables these
appear as `R1 R2 R3`, meaning

```text
R_cart = R1*a1 + R2*a2 + R3*a3
```

where `a1`, `a2`, and `a3` are the direct lattice vectors parsed from the Elk
input. Selection priority is:

1. explicit `--vectors` file, or `./jfile` if present;
2. `--rmax`, which generates all integer direct-lattice translations within the
   real-space cutoff in Angstrom;
3. otherwise, a centered integer `R` grid inferred from the nested q-mesh
   dimensions, limited to half of the maximum inferred real-space distance.
   This keeps the default transform away from the longest mesh-boundary
   vectors while still requiring no manual cutoff.

## Seekpath Plot with Dense FT Spectrum

```bash
fourj \
  --energy energy_vs_q.dat \
  --elk elk.tmp \
  --vectors jfile \
  --theta 90 \
  --symmetry spglib \
  --plot-path \
  --plot-lswt \
  --lswt-dense-path
```

## Two-Shell LSQ Fit

```bash
fourj \
  --energy energy_vs_q.dat \
  --elk elk.tmp \
  --vectors jfile \
  --theta 90 \
  --symmetry spglib \
  --fit-lsq \
  --fit-num-shells 2 \
  --plot-path \
  --plot-lswt \
  --lswt-dense-path
```


## Interactive Dashboard

Install the optional dashboard dependencies:

```bash
pip install -e /path/to/FourJ[dashboard]
```

Then run:

```bash
fourj-dashboard
```

The dashboard opens at `http://127.0.0.1:8050`. Upload an Elk input or
`elk.tmp`, upload `energy_vs_q.dat`, and optionally upload a `jfile`. The app
runs the same `FrozenMagnonWorkflow` as the CLI and shows the reciprocal
q-point cloud, full input `E(q)`, Seekpath DFT/FT/LSQ comparisons, and
real-space `J(R)`. It can also download an UppASD-style exchange file with
columns `iatom jatom r_x r_y r_z Jij |rij|`; `Jij` is in mRy and `|rij|` is in
Angstrom. The dashboard q-point markers are colored by
`E(q)-E0` in mRy, and the status panel reports available Bravais and
space-group metadata from Seekpath/spglib.

You can also launch it preloaded from CLI file paths and settings:

```bash
fourj --energy energy_vs_q.dat --elk elk.tmp --symmetry spglib --gui
```

## Programmatic API

```python
from pathlib import Path
from fourj import FrozenMagnonWorkflow, WorkflowConfig

workflow = FrozenMagnonWorkflow(
    WorkflowConfig(
        energy_path=Path("energy_vs_q.dat"),
        elk_path=Path("elk.tmp"),
        vectors_path=Path("jfile"),
        theta=90.0,
        symmetry="spglib",
    )
)

result = workflow.run_transform()
workflow.write_transform_outputs()
lsq = workflow.fit_lsq(max_shells=2)
workflow.write_lsq_outputs()
```


## Hosting

The repository includes `app.py` and a `Dockerfile` for hosting the dashboard.
For Hugging Face Spaces, create a Docker Space and push the repository; the
container listens on port `7860`. Generic Python hosts can run
`gunicorn app:server`, while command-based hosts can run `fourj-dashboard` with
`HOST=0.0.0.0` and their provided `PORT`.
