"""Mesh convergence study for the DJI 9443 hover case (5400 rpm).

Refines the panel mesh and watches the thrust settle to a grid-independent
value along three refinement paths (span, chord, diagonal). Each path is
extrapolated to zero grid spacing with a power-law fit

    T(h) = T_inf + C * h**p ,     h ~ 1 / sqrt(number of panels)

and the converged value T_inf is drawn on the plot as a dashed asymptote, so
the data is seen approaching its limit.


"""
import os

import numpy as np
import matplotlib.pyplot as plt

from nl_vlm import PropGeom, PropMesh, Vehicle, VLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# fixed operating point: DJI 9443, hover, 5400 rpm
RHO = 1.07175
RPM = 5400
R_TIP = 0.12
R_HUB = 0.00624
NUM_BLADES = 2
OMEGA = {"Propeller_1": np.array([0.0, 0.0, RPM * 2 * np.pi / 60])}


def thrust_at(span, chord):
    """Solve DJI hover at the given mesh resolution -> (thrust_N, n_panels)."""
    blade = os.path.join(DATA, "blades", "dji_9443")
    geom = PropGeom(
        airfoil_distribution_file=os.path.join(blade, "DJI9443_airfoildist.csv"),
        chorddist_file=os.path.join(blade, "DJI9443_chorddist.csv"),
        pitchdist_file=os.path.join(blade, "DJI9443_pitchdist.csv"),
        sweepdist_file=os.path.join(blade, "DJI9443_sweepdist.csv"),
        heightdist_file=os.path.join(blade, "DJI9443_heightdist.csv"),
        airfoil_path=os.path.join(DATA, "airfoils", "dji_9443"),
        polar_path=os.path.join(DATA, "polars", "dji_9443"),
        R_tip=R_TIP, R_hub=R_HUB, num_blades=NUM_BLADES,
    )
    mesh = PropMesh(geom, span_resolution=span, chord_resolution=chord)
    vm = Vehicle(mesh, hub_positions=None, spin_directions=[1]).generate_vehicle()
    fm = VLM(vm, mesh, verbose=False).calculate_total_forces_and_moments(
        propeller_mesh=vm, dt=0.0, rho=RHO, time_step=1,
        body_velocity=np.zeros(3), freestream=np.zeros(3), omega=OMEGA,
        wind_field=None, com_position=np.zeros(3), euler_angles=np.zeros(3),
    )
    T = fm["Propeller_1"]["force"][2]
    n_panels = (span - 1) * (chord - 1) * NUM_BLADES
    return T, n_panels


def run_study(name, cases):
    """cases: list of (span, chord). Print a table; return (panels[], thrust[])."""
    panels, thrust, prev = [], [], None
    for span, chord in cases:
        T, npan = thrust_at(span, chord)
        dpct = "-" if prev is None else f"{100 * (T - prev) / prev:+.2f}%"
        panels.append(npan); thrust.append(T); prev = T
    return panels, thrust


def richardson(cases3):
    """3-grid, ratio-2 Richardson extrapolation to zero grid spacing.

    ``cases3`` = [(coarse), (medium), (fine)] (span, chord), with the grid
    spacing halving each level (panel count x4). Returns (T_inf, order p), or
    (None, None) if the three values are non-monotonic (not meaningful).
    """
    T = [thrust_at(s, c)[0] for (s, c) in cases3]
    e_fine = T[1] - T[2]      # medium - fine
    e_coarse = T[0] - T[1]    # coarse - medium
    if e_fine == 0 or (e_coarse / e_fine) <= 0:
        return None, None
    p = np.log(e_coarse / e_fine) / np.log(2.0)
    t_inf = T[2] + (T[2] - T[1]) / (2.0 ** p - 1)
    return t_inf, p


if __name__ == "__main__":
    # extend well past the "knee" so the curves visibly flatten
    span_cases = [(s, 7) for s in (7, 11, 15, 19, 23, 27, 33, 41, 49)]        # chord = 7
    chord_cases = [(15, c) for c in (3, 5, 7, 9, 11, 15, 21, 27)]            # span = 15
    diag_cases = [(7, 3), (11, 5), (15, 7), (19, 9), (23, 11), (29, 15)]      # refine both

    # clean 3-grid, ratio-2 grids for the defensible Richardson asymptote
    rich_grids = {
        "Span study (chord = 7)":  [(9, 7),  (17, 7),  (33, 7)],
        "Chord study (span = 15)": [(15, 5), (15, 9),  (15, 17)],
        "Diagonal refinement":     [(9, 5),  (17, 9),  (33, 17)],
    }

    studies = []
    for name, cases in [("Span study (chord = 7)", span_cases),
                        ("Chord study (span = 15)", chord_cases),
                        ("Diagonal refinement", diag_cases)]:
        panels, thrust = run_study(name, cases)
        t_inf, p = richardson(rich_grids[name])
        reliable = t_inf is not None and abs(p) >= 0.5
        if reliable:
            print(f"  -> converged T(inf) = {t_inf:.4f} N   (Richardson r=2, order p = {p:.2f})")
        else:
            porder = "n/a" if p is None else round(p, 2)
            print(f"  -> no reliable asymptote (slow convergence; order p ~ {porder})")
        studies.append((panels, thrust, name, t_inf if reliable else None))

    # ---- plot ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 26,
        "axes.labelsize": 26,
        "axes.titlesize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "axes.linewidth": 1.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    for ax, (panels, thrust, title, t_inf) in zip(axes, studies):
        ax.plot(panels, thrust, 'o-', color=plt.cm.viridis(0.4),
                markersize=8, linewidth=2.0)
        if t_inf is not None:
            ax.axhline(t_inf, ls='--', color='0.35', linewidth=1.5)
            ax.text(0.96, 0.10, rf"$T_\infty$ = {t_inf:.3f} N", transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=20)
        else:
            ax.text(0.96, 0.10, "slow convergence", transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=18, style='italic')
        ax.set_xlabel("number of panels")
        ax.set_ylabel(r"thrust  $F_z$  (N)")
        ax.set_title(title)
        ax.tick_params(axis='both', which='major', length=6, width=1.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_convergence.png")
    plt.savefig(out, dpi=300)
    plt.show()
