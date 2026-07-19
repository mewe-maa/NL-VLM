# nl-vlm

Nonlinear vortex-lattice method (NL-VLM) for rotor / propeller aerodynamics.

Validated on the DJI 9443 rotor in hover: **thrust ≈ 1.999 N at 5400 rpm**
(≈5% below the experimental value). See `validation/dji9443_validation.ipynb`.

## Package layout

```
src/nl_vlm/
├── rotor/         blade geometry, meshing, multi-rotor vehicle assembly
├── solvers/       the NL-VLM solver (vlm.py) + experimental variants
├── environment/   wind field (KD-tree lookup, Jacobian/Hessian approximation)
├── dynamics/      rotor azimuthal rotation + whole-vehicle 6-DOF dynamics
├── wing/          fixed-wing modules (experimental)
└── control/       flight-dynamics / control (experimental)
```

## Install

```bash
pip install -e .          # runtime
pip install -e ".[dev]"   # + pytest for the tests
```

## Usage

Edit the config block at the top of `main.py`, then:

```bash
python main.py
```

Or from Python — the validated pipeline is `PropGeom → PropMesh → Vehicle → VLM`:

```python
from nl_vlm import PropGeom, PropMesh, Vehicle, VLM
# build geometry -> panel mesh -> vehicle assembly -> solve
```

`validation/dji9443_validation.ipynb` shows the full DJI hover case with the
azimuth time-stepping loop and run-level progress reporting.

## Validation & mesh convergence

- `validation/` — DJI 9443, APC 10x7, Caradonna–Tung validation notebooks.
- `studies/mesh_convergence.py` — grid-convergence study (span / chord / diagonal
  refinement + Richardson extrapolation). The KJ force uses a strip-averaged
  induced velocity so it is chordwise grid-convergent; the diagonal-refined
  thrust extrapolates to ≈ 1.976 N (2nd order, GCI 0.14%).

## Tests

```bash
pytest
```

`tests/test_dji_validation.py` guards the DJI hover thrust against regressions.

## Data

Small geometry / airfoil / polar inputs live in `data/`. Large flow-field data
(`.vtk` snapshots, resampled fields) are **not** tracked in git — supply them
separately.
