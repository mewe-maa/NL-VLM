"""Rigid azimuthal rotation of a rotor mesh, for quasi-steady azimuth sweeps.

The steady VLM solver holds the blade at one azimuth. To sweep the rotor through
a revolution we rotate the *bound* mesh about each rotor's hub z-axis by an angle
psi, re-solve, and record the loading. This module provides that rotation as a
pure function: it returns a new mesh and never mutates the input.
"""
import copy
import numpy as np


def _rz(psi):
    """3x3 rotation matrix about the +z axis by ``psi`` radians."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def rotate_vehicle_mesh(vehicle_mesh, psi):
    """Return a copy of ``vehicle_mesh`` spun about each rotor's hub z-axis.

    Parameters
    ----------
    vehicle_mesh : dict
        Mesh produced by :meth:`nl_vlm.rotor.vehicle.Vehicle.generate_vehicle`.
    psi : float or dict
        Azimuth angle in radians. A float rotates every rotor by the same
        angle; a ``{propeller_key: angle}`` dict rotates each rotor
        independently. Use signed angles so CW (negative-omega) rotors turn the
        correct way, e.g. ``psi = spin_direction * omega_mag * dt``.

    Returns
    -------
    dict
        A new, rotated mesh. Point data (control points, panel corners, vortex
        ring vertices) is rotated about the rotor hub; direction data (normals,
        tangential vectors) is rotated in place. Scalars (twist, chord, r_R,
        gamma) and the hub position are unchanged.

    Notes
    -----
    Because a z-rotation preserves dot products and the axial (z) direction, an
    axisymmetric hover solution is invariant under this rotation: thrust (Fz)
    and axial torque (Mz) are unchanged at every azimuth. That invariant is the
    correctness test for this function.
    """
    rotated = copy.deepcopy(vehicle_mesh)

    for prop_key, prop_data in rotated.items():
        angle = psi[prop_key] if isinstance(psi, dict) else psi
        if angle == 0.0:
            continue
        R = _rz(angle)
        hub = np.asarray(prop_data['Hub Position'], dtype=float)

        for blade_data in prop_data['Blades'].values():
            # --- point data: rotate about the hub (translate, rotate, translate back)
            for cp_index, cp in blade_data['Control Points'].items():
                blade_data['Control Points'][cp_index] = hub + R @ (np.asarray(cp, dtype=float) - hub)

            for panel_index, panel in blade_data['Panels'].items():
                blade_data['Panels'][panel_index] = [
                    hub + R @ (np.asarray(v, dtype=float) - hub) for v in panel
                ]

            for ring_index, ring in blade_data['Vortex Rings'].items():
                ring['Vertices'] = [
                    hub + R @ (np.asarray(v, dtype=float) - hub) for v in ring['Vertices']
                ]

            # --- direction data: rotate the vectors only (no translation)
            for n_index, normal in blade_data['Normals'].items():
                blade_data['Normals'][n_index] = R @ np.asarray(normal, dtype=float)

            for t_index, tv in blade_data['Tangential Vectors'].items():
                tv['Tangential i'] = R @ np.asarray(tv['Tangential i'], dtype=float)
                tv['Tangential j'] = R @ np.asarray(tv['Tangential j'], dtype=float)

    return rotated
