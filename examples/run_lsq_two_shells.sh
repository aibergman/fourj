#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/Users/andersb/envs/p311/bin/python}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${PYTHON}" "${ROOT}/fourj.py" \
  --energy energy_vs_q.dat \
  --elk elk.tmp \
  --vectors jfile \
  --theta 90 \
  --symmetry spglib \
  --output-prefix fourj_example_lsq2 \
  --fit-lsq \
  --fit-num-shells 2 \
  --plot-path \
  --plot-lswt \
  --lswt-dense-path
