"""Offline preprocessing that produces solver inputs.

The rotor solve consumes 2D polars (``PropGeom(polar_path=...)``); this subpackage
produces them. It is the producer half of that schema, so it is versioned with the
consumer rather than living outside the package.

Requires an XFOIL executable, which is NOT installed with nl_vlm -- XFOIL is a
separate GPL-2 program by Drela & Youngren. Pass its path as ``xfoil_exe``.
"""
from nl_vlm.preprocess.xfoil import generate_polars

__all__ = ["generate_polars"]
