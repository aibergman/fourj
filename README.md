# FourJ

![FourJ logo](fourj.png)

`fourj` extracts one-sublattice Heisenberg exchange parameters from Elk
spin-spiral frozen-magnon calculations.

The workflow reads `energy_vs_q.dat` and `elk.tmp`, converts `E(q)` to
`J(0)-J(q)`, inverse Fourier transforms to real-space `J_ij`, optionally fits
finite symmetry shells by least squares, and plots DFT/FT/LSQ spectra along
Seekpath high-symmetry paths.

## Quick Start

From a calculation directory containing `energy_vs_q.dat`, `elk.tmp`, and
`jfile`:

After `pip install -e /path/to/FourJ`, use the console command below.
For local, non-installed use, replace `fourj` with `python /path/to/FourJ/fourj.py`.

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

For a two-shell LSQ comparison:

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
