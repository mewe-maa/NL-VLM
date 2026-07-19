"""Dynamics: time-evolution / motion of the rotor and the whole vehicle.

- ``rotor``   : rotor azimuthal rotation (kinematic transform of the bound mesh)
"""
from nl_vlm.dynamics.rotor import rotate_vehicle_mesh

__all__ = ["rotate_vehicle_mesh"]
