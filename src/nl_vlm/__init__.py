"""nl_vlm - Nonlinear vortex-lattice method for rotor/propeller aerodynamics.

Convenience re-exports of the validated core pipeline::

    from nl_vlm import PropGeom, PropMesh, Vehicle, VLM, WindField
"""
from nl_vlm.rotor.prop_geom import PropGeom
from nl_vlm.rotor.prop_mesh import PropMesh
from nl_vlm.rotor.vehicle import Vehicle
from nl_vlm.dynamics.rotor import rotate_vehicle_mesh
from nl_vlm.solvers.vlm import VLM   # active solver: BEMT momentum-closure variant
from nl_vlm.environment.wind import WindField

__all__ = ["PropGeom", "PropMesh", "Vehicle", "rotate_vehicle_mesh", "VLM", "WindField"]
__version__ = "0.1.0"
