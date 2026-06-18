Theory and Conventions
======================

FourJ implements a frozen-magnon route from calculated spin-spiral energies
:math:`E(\mathbf{q})` to isotropic real-space exchange parameters
:math:`J(\mathbf{R})`.  The current implementation is intentionally scalar and
single-sublattice: it assumes that the magnetic degrees of freedom can be mapped
onto one magnetic atom per primitive cell.  Non-magnetic atoms can be present in
the structure and are used for symmetry, but they do not add independent magnon
branches.

The conventions follow the frozen-magnon mapping described in Kuebler's text on
itinerant magnetism [Kuebler2000]_, with closely related background from the
magnetic force-theorem and adiabatic spin-dynamics literature
[Liechtenstein1987]_ [Halilov1998]_ [Sandratskii1998]_ [Katsnelson2004]_.

Model Hamiltonian
-----------------

The exchange model used internally is the classical isotropic Heisenberg form

.. math::

   H = - \sum_{ij} J_{ij}\, \mathbf{e}_i \cdot \mathbf{e}_j,

where :math:`\mathbf{e}_i` is a unit vector along the local magnetic moment.  For
a periodic crystal with one magnetic sublattice, the pair interaction can be
written as :math:`J(\mathbf{R})`, where :math:`\mathbf{R}` is an integer direct
lattice translation from one magnetic atom to another equivalent magnetic atom.

The lattice Fourier transform convention is

.. math::

   J(\mathbf{q}) = \sum_{\mathbf{R}} J(\mathbf{R})
   \exp\left(i 2\pi\,\mathbf{q}\cdot\mathbf{R}\right),

with :math:`\mathbf{q}` expressed in fractional reciprocal coordinates.  With
this convention, :math:`\mathbf{q}\cdot\mathbf{R}` is dimensionless.

Frozen-magnon energy relation
-----------------------------

A flat spin spiral with cone angle :math:`\theta` can be written as

.. math::

   \mathbf{e}_{\mathbf{R}} =
   \left(
   \sin\theta\cos(2\pi\mathbf{q}\cdot\mathbf{R}),
   \sin\theta\sin(2\pi\mathbf{q}\cdot\mathbf{R}),
   \cos\theta
   \right).

For the scalar one-sublattice model, the spin-spiral energy difference is
proportional to :math:`J(0)-J(\mathbf{q})`:

.. math::

   E(\mathbf{q}) - E(0)
   = \sin^2\theta\,\left[J(0)-J(\mathbf{q})\right].

Thus FourJ converts the input energies to the reciprocal-space exchange
quantity

.. math::

   \Delta J(\mathbf{q}) = J(0)-J(\mathbf{q})
   = \frac{E(\mathbf{q})-E(0)}{\sin^2\theta}.

Elk total energies are read in Hartree.  FourJ converts the exchange scale to
mRy using

.. math::

   1\ \mathrm{Hartree} = 2000\ \mathrm{mRy}.

The sign convention above follows the ferromagnetic Heisenberg convention in
which positive :math:`J` favors parallel alignment.  Since the frozen-magnon
input gives only energy differences, the absolute constant :math:`J(0)` is fixed
through the chosen transform or fitting convention rather than independently
known.

Full Fourier-transform workflow
-------------------------------

For a complete sampled reciprocal mesh, FourJ performs a discrete inverse
transform from the symmetrized :math:`\Delta J(\mathbf{q})` values to real-space
interactions.  In the current convention this is equivalent to

.. math::

   J(\mathbf{R}) = - \sum_{\mathbf{q}} w_{\mathbf{q}}\,
   \Delta J(\mathbf{q})
   \exp\left(-i 2\pi\,\mathbf{q}\cdot\mathbf{R}\right),

for :math:`\mathbf{R}\neq 0`, with normalized reciprocal weights
:math:`w_{\mathbf{q}}`.  The minus sign appears because the input quantity is
:math:`J(0)-J(\mathbf{q})`, not :math:`J(\mathbf{q})` itself.

The extracted interaction set can be transformed back to an adiabatic spectrum
on arbitrary path points:

.. math::

   \Delta J_{\mathrm{FT}}(\mathbf{q})
   = \sum_{\mathbf{R}} J(\mathbf{R})
   \left[1 - \exp\left(i2\pi\,\mathbf{q}\cdot\mathbf{R}\right)\right].

For a real isotropic model the plotted branch uses the real part; the imaginary
part is a diagnostic for incomplete sampling or residual symmetry error.

Symmetry handling
-----------------

The code uses the crystal symmetry operations from ``spglib`` to work with
symmetry-equivalent reciprocal vectors and real-space translations.  In the full
Fourier-transform workflow, the main enforced symmetries are:

* reciprocal-space averaging of equivalent :math:`\mathbf{q}` points before the
  transform;
* real-space grouping of equivalent :math:`\mathbf{R}` vectors after the
  transform;
* the Hermitian pair relation :math:`J(\mathbf{R}) = J(-\mathbf{R})^*`, which
  becomes :math:`J(\mathbf{R}) = J(-\mathbf{R})` for the final real scalar
  exchange model.

These operations reduce numerical noise and make the result compatible with the
space-group symmetry of the parsed structure.  They do not turn a
multi-sublattice magnetic problem into a single-sublattice one; that limitation
is part of the model assumption.

Least-squares shell fitting
---------------------------

As a complementary approach, FourJ can fit a truncated shell model directly to
the input frozen-magnon energies.  Symmetry-equivalent real-space vectors are
grouped into shells :math:`s`, and each shell contributes the basis function

.. math::

   B_s(\mathbf{q}) = \sum_{\mathbf{R}\in s}
   \left[1 - \cos\left(2\pi\,\mathbf{q}\cdot\mathbf{R}\right)\right].

The fitted model is then

.. math::

   \Delta J(\mathbf{q}) \approx \sum_s J_s B_s(\mathbf{q}),

which is a standard linear least-squares problem once the shells have been
selected.  This is useful when testing whether nearest-neighbor or
next-nearest-neighbor interactions already explain the DFT frozen-magnon curve.
It is also a useful check on the full Fourier transform, because both models can
be plotted against the same input :math:`E(\mathbf{q})` data.

Adiabatic magnon scale
----------------------

For comparison plots FourJ can rescale :math:`\Delta J(\mathbf{q})` to an
adiabatic magnon-like energy.  In the one-sublattice classical convention used
here,

.. math::

   \omega(\mathbf{q}) = \frac{4}{M}\left[J(0)-J(\mathbf{q})\right],

where :math:`M` is the magnetic moment in Bohr magnetons.  The result is then
converted from mRy to meV when requested.  Users should treat this as a
single-branch adiabatic comparison scale.  A true multi-sublattice LSWT treatment
would require an exchange matrix :math:`J_{\alpha\beta}(\mathbf{R})` and would
produce multiple magnon branches.

Relation to other exchange formalisms
-------------------------------------

Frozen magnons and real-space magnetic force-theorem approaches are closely
related but not identical workflows.  The Liechtenstein-Katsnelson-Antropov-
Gubanov method extracts pair interactions from infinitesimal rotations of a
reference magnetic state [Liechtenstein1987]_, while the frozen-magnon route used
here maps total energy differences for finite spin spirals onto the same scalar
Heisenberg language [Kuebler2000]_ [Halilov1998]_.  Agreement between the two
approaches is expected when the adiabatic mapping is valid, the chosen reference
state is appropriate, and the magnetic system is well described by pairwise
isotropic exchange.

References
----------

.. [Kuebler2000] J. Kuebler, *Theory of Itinerant Electron Magnetism*, Oxford
   University Press, 2000.

.. [Liechtenstein1987] A. I. Liechtenstein, M. I. Katsnelson, V. P. Antropov,
   and V. A. Gubanov, "Local spin density functional approach to the theory of
   exchange interactions in ferromagnetic metals and alloys", *Journal of
   Magnetism and Magnetic Materials* **67**, 65--74 (1987).

.. [Halilov1998] S. V. Halilov, H. Eschrig, A. Y. Perlov, and P. M. Oppeneer,
   "Adiabatic spin dynamics from spin-density-functional theory: Application to
   Fe, Co, and Ni", *Physical Review B* **58**, 293 (1998).

.. [Sandratskii1998] L. M. Sandratskii, "Noncollinear magnetism in itinerant-
   electron systems: theory and applications", *Advances in Physics* **47**,
   91--160 (1998).

.. [Katsnelson2004] M. I. Katsnelson and A. I. Liechtenstein, "Magnetic
   susceptibility, exchange interactions and spin-wave spectra in the local spin
   density approximation", arXiv:cond-mat/0406488 (2004).
