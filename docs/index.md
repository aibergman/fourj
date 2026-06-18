# FourJ

`fourj` is a small Python package for analyzing one-sublattice
frozen-magnon spin-spiral calculations from Elk.

It supports:

- parsing `energy_vs_q.dat` and Elk lattice/basis data from `elk.tmp`;
- reciprocal-space symmetry averaging with `spglib`;
- full inverse Fourier transforms to `J_ij`;
- finite-shell LSQ fitting of `J(0)-J(q)`;
- Matplotlib Seekpath plots comparing DFT, full-FT, and LSQ spectra.

```{toctree}
:maxdepth: 2

usage
theory
api
```
