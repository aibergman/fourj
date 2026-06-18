# Usage

Run from a calculation directory containing:

- `energy_vs_q.dat`
- `elk.tmp`
- optionally `jfile`

Install in editable mode with `pip install -e /path/to/FourJ`, or replace
`fourj` in the examples below with `python /path/to/FourJ/fourj.py`.

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
real-space `J(R)`.

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
