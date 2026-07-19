"""Human-facing progress / status reporting.

All console output lives here, kept out of the solver so the solver stays pure
physics. Two reporters:

* :class:`SolverReporter` — per-solve: banner + viscous-convergence bar + result.
  Held by :class:`~nl_vlm.solvers.vlm.VLM`; disabled when ``verbose=False``.
* :class:`SimReporter` — run-level for a time-stepping simulation: the banner
  once, a single progress bar over the *time steps*, and a mean-force summary.
"""
import numpy as np


def render_banner(propeller_mesh, omega, rho, freestream,
                  span_res=None, chord_res=None, width=56):
    """Print the 'NL-VLM analysis starting' configuration banner."""
    n_rotors = len(propeller_mesh)
    first = next(iter(propeller_mesh.values()))
    n_blades = len(first['Blades'])
    n_panels = sum(len(bd['Control Points'])
                   for pd in propeller_mesh.values()
                   for bd in pd['Blades'].values())
    omega_z = float(np.asarray(next(iter(omega.values())))[2])
    rpm = abs(omega_z) * 60.0 / (2.0 * np.pi)

    print("=" * width)
    print("  NL-VLM  |  analysis starting")
    print("=" * width)
    print(f"  Rotors        : {n_rotors}")
    print(f"  Blades/rotor  : {n_blades}")
    if span_res and chord_res:
        print(f"  Resolution    : {span_res} span x {chord_res} chord  ({n_panels} panels)")
    else:
        print(f"  Panels (total): {n_panels}")
    print(f"  Rotor speed   : {rpm:.0f} rpm  ({abs(omega_z):.1f} rad/s)")
    print(f"  Air density   : {rho:.4f} kg/m^3")
    print(f"  Freestream    : {np.asarray(freestream)} m/s")
    print("-" * width)


def _bar(fraction, width=24):
    """Text progress bar for a fraction in [0, 1]."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(width * fraction))
    return "#" * filled + "-" * (width - filled)


class SolverReporter:
    """Per-solve progress to stdout. Use ``enabled=False`` for a silent run."""

    def __init__(self, enabled=True, width=56, bar_width=24):
        self.enabled = enabled
        self.width = width
        self.bar_width = bar_width
        self._delta0 = None
        self._progress = 0.0
        self._log_tol = 0.0

    def announce(self, propeller_mesh, omega, rho, freestream,
                 span_res=None, chord_res=None):
        """Print the config banner for this solve."""
        if self.enabled:
            render_banner(propeller_mesh, omega, rho, freestream,
                          span_res, chord_res, width=self.width)

    def start_solve(self, tolerance):
        """Reset progress state at the start of a per-rotor solve."""
        self._delta0 = None
        self._progress = 0.0
        self._log_tol = np.log10(tolerance)

    def progress(self, label, iteration, delta, every=5):
        """Refresh the in-place viscous-convergence bar (log-scale residual)."""
        if not self.enabled:
            return
        if self._delta0 is None and delta > 0:
            self._delta0 = delta
        if self._delta0 and delta > 0:
            frac = (np.log10(self._delta0) - np.log10(delta)) / (np.log10(self._delta0) - self._log_tol)
            self._progress = max(self._progress, min(1.0, float(frac)))
        if iteration % every == 0:
            print(f"\r  solving {label:<12} [{_bar(self._progress, self.bar_width)}] "
                  f"{100 * self._progress:5.1f}%   iter {iteration:4d}   res {delta:.1e}",
                  end="", flush=True)

    def finish_solve(self, label, iteration, delta, converged, max_iterations):
        """Close out the convergence line for a rotor solve."""
        if not self.enabled:
            return
        if converged:
            print(f"\r  solving {label:<12} [{_bar(1.0, self.bar_width)}] 100.0%   "
                  f"lift coupling converged in {iteration + 1} iters   res {delta:.1e}      ")
        else:
            print(f"\r  solving {label:<12} [{_bar(self._progress, self.bar_width)}] "
                  f"{100 * self._progress:5.1f}%   lift coupling NOT converged after {max_iterations} "
                  f"iters (res {delta:.1e})")

    def summary(self, forces_and_moments):
        """Print per-rotor thrust and torque once the solve is complete."""
        if not self.enabled:
            return
        print("-" * self.width)
        print("  VLM solved")
        for pk, fm in forces_and_moments.items():
            f = fm['force']
            m = fm['moment']
            print(f"    {pk}:  thrust Fz = {f[2]:+.4f} N    torque Mz = {m[2]:+.4e} N*m")
        print("=" * self.width)


class SimReporter:
    """Run-level reporting for a time-stepping rotor simulation.

    Usage per run:  :meth:`announce` once, then :meth:`step` after each solved
    time step, then :meth:`summary` with the mean force/moment. The per-step VLM
    solver should run with ``verbose=False`` so only this run-level output shows.
    The single progress bar tracks the *time steps* and reaches 100% when the
    last step finishes.
    """

    def __init__(self, total_steps, enabled=True, width=56, bar_width=24):
        self.total = total_steps
        self.enabled = enabled
        self.width = width
        self.bar_width = bar_width

    def announce(self, propeller_mesh, omega, rho, freestream,
                 span_res=None, chord_res=None):
        """Print the config banner ONCE at the start of the run."""
        if self.enabled:
            render_banner(propeller_mesh, omega, rho, freestream,
                          span_res, chord_res, width=self.width)

    def step(self, done, label="Propeller_1"):
        """Advance the time-step progress bar in place ('done' = steps completed)."""
        if not self.enabled:
            return
        frac = done / self.total
        print(f"\r  solving {label:<12} [{_bar(frac, self.bar_width)}] "
              f"{100 * frac:5.1f}%   time step {done}/{self.total}",
              end="", flush=True)

    def summary(self, mean_force, mean_moment, label="Propeller_1"):
        """Print the mean thrust / torque over all time steps."""
        if not self.enabled:
            return
        print()  # finish the progress-bar line
        print("-" * self.width)
        print(f"  VLM solved  (mean over {self.total} time steps)")
        print(f"    {label}:  mean thrust Fz = {mean_force[2]:+.4f} N    "
              f"mean torque Mz = {mean_moment[2]:+.4e} N*m")
        print("=" * self.width)
