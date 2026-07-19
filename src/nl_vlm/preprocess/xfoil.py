"""Generate 2D airfoil polars with XFOIL -- the input `PropGeom(polar_path=...)` reads.

For each r/R station of the blade:
  1. local velocity  V = sqrt((omega*r)^2 + V_inf^2),  V_inf = J * n * D
  2. local Re = rho*V*c/mu  and  Mach = V/a
  3. XFOIL viscous alpha sweep, outward from 0 in both directions
  4. write <out_dir>/polar_rR<tag>.csv
  5. repoint the airfoildist's 'Aero file' column at those files
"""

import os
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def local_conditions(r_R, chord_R, R_tip, rpm, J, rho, mu, a_sound, re_floor):
    """Local (V, Re, Mach) at one radial station.

    Re is floored because at the root r -> 0, so V -> 0 and XFOIL's viscous march
    has nothing to solve; those stations carry almost no load anyway.
    """
    n = rpm / 60.0
    V_inf = J * n * (2.0 * R_tip)
    omega = rpm * 2.0 * np.pi / 60.0

    V = np.hypot(omega * r_R * R_tip, V_inf)
    Re = max(rho * V * chord_R * R_tip / mu, re_floor)
    Mach = V / a_sound
    Mach = 0
    return V, Re, Mach


def run_xfoil(xfoil_exe, work_dir, dat_name, polar_txt_name, Re, Mach,
              alpha_i, alpha_f, alpha_step, n_iter, Ncrit):
    """Run one viscous alpha sweep at this Re/Mach. True if a polar appeared.

    Sweeps outward from alpha=0 both ways, re-initializing the boundary layer
    between, so the viscous march stays converged.

    XFOIL runs IN `work_dir` and is handed bare filenames, never paths. Two reasons:
    it resolves them against its own cwd (not the caller's, so a relative path from
    the caller would silently point somewhere else), and its Fortran filename buffer
    is short enough that a long absolute path can be truncated.
    """
    input_file = os.path.join(work_dir, '_xfoil_input.in')
    log_file = os.path.join(work_dir, '_xfoil_log.txt')
    polar_txt_path = os.path.join(work_dir, polar_txt_name)

    if os.path.exists(polar_txt_path):
        os.remove(polar_txt_path)

    with open(input_file, 'w') as f:
        f.write(f"LOAD {dat_name}\n")
        f.write(f"{os.path.splitext(dat_name)[0]}\n")
        f.write("PANE\n")
        f.write("OPER\n")
        f.write(f"VISC {Re:.0f}\n")
        f.write(f"MACH {Mach:.4f}\n")
        f.write(f"ITER {n_iter}\n")
        f.write("VPAR\n")
        f.write(f"N {Ncrit}\n\n")             # exit VPAR
        f.write("PACC\n")
        f.write(f"{polar_txt_name}\n\n")      # no dump file
        f.write(f"ASEQ 0 {alpha_f} {alpha_step}\n")        # 0 -> alpha_f
        f.write("INIT\n")                                  # reset BL
        f.write(f"ASEQ 0 {alpha_i} {-abs(alpha_step)}\n")  # 0 -> alpha_i
        f.write("\nQUIT\n")

    # XFOIL hangs on some sections; the timeout keeps one bad station from stalling
    # the whole sweep.
    try:
        with open(input_file, 'r') as fin, open(log_file, 'w') as flog:
            subprocess.run([xfoil_exe], stdin=fin, stdout=flog, stderr=flog,
                           cwd=work_dir, timeout=60)
    except subprocess.TimeoutExpired:
        print("    ! timed out (60 s) -- keeping whatever converged")

    ok = os.path.exists(polar_txt_path) and os.path.getsize(polar_txt_path) > 0

    for tmp in (input_file, log_file):
        if os.path.exists(tmp):
            os.remove(tmp)

    return ok


def parse_xfoil_polar(polar_txt_path):
    """XFOIL polar file (12-line header) -> DataFrame."""
    raw = np.loadtxt(polar_txt_path, skiprows=12)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return pd.DataFrame(raw, columns=['Alpha', 'Cl', 'Cd', 'Cdp', 'Cm', 'Top_Xtr', 'Bot_Xtr'])


def csv_to_dat(csv_path, dat_path):
    """Airfoil contour CSV (x/c, y/c) -> XFOIL .dat."""
    df = pd.read_csv(csv_path)
    with open(dat_path, 'w') as f:
        f.write(f"{os.path.splitext(os.path.basename(csv_path))[0]}\n")
        for _, row in df.iterrows():
            f.write(f"  {row['x/c']:.6f}  {row['y/c']:.6f}\n")


def generate_polars(airfoil_dist_file, chorddist_file, airfoil_dir, out_dir,
                    xfoil_exe, R_tip, rpm, J=0.0,
                    rho=1.225, mu=1.81e-5, a_sound=343.0,
                    alpha_i=-7.0, alpha_f=17.0, alpha_step=0.25,
                    n_iter=200, Ncrit=5, re_floor=10_000):
    """Generate one polar per radial station, at the given operating point.

    Writes <out_dir>/polar_rR*.csv and repoints the airfoildist's 'Aero file'
    column at them (backing that file up first). Returns `airfoil_dist_file`.

    Parameters
    ----------
    airfoil_dist_file, chorddist_file : str
        The blade's airfoil distribution and chord distribution CSVs.
    airfoil_dir : str
        Folder holding the contour files named in the airfoil distribution.
    out_dir : str
        Where the polars are written. This is what you later hand to
        ``PropGeom(polar_path=...)``.
    xfoil_exe : str
        Path to the XFOIL executable. Supplied by the caller: XFOIL is a separate
        GPL-2 program and is not installed with this package.
    R_tip, rpm, J : float
        Operating point. Re and Mach are derived per station from these, so pass
        the SAME values the solver runs at -- a polar set is only valid near the
        condition it was built at.
    rho, mu, a_sound : float
        Atmosphere. rho should match the solver's rho for consistency.
    alpha_i, alpha_f, alpha_step : float
        Sweep range [deg]. It must cover the alpha_eff the blade actually sees, or
        the solver's polar lookup extrapolates off the end of the table.
    """
    airfoil_dist_file = os.path.abspath(airfoil_dist_file)
    chorddist_file = os.path.abspath(chorddist_file)
    airfoil_dir = os.path.abspath(airfoil_dir)
    out_dir = os.path.abspath(out_dir)
    xfoil_exe = os.path.abspath(xfoil_exe)

    airfoil_dist = pd.read_csv(airfoil_dist_file)
    chorddist = pd.read_csv(chorddist_file)

    chord_interp = interp1d(chorddist['r/R'], chorddist['c/R'],
                            kind='linear', fill_value='extrapolate')

    dat_temp_dir = os.path.join(out_dir, '_dat_temp')
    os.makedirs(dat_temp_dir, exist_ok=True)

    aero_files = []

    print("=" * 70)
    print("XFOIL polar generator")
    print(f"  rpm = {rpm},  J = {J},  R_tip = {R_tip} m,  rho = {rho}")
    print(f"  alpha: {alpha_i} to {alpha_f} deg (step {alpha_step})")
    print(f"  in  <- {os.path.basename(airfoil_dist_file)}")
    print(f"  out -> {out_dir}")
    print("=" * 70)

    for _, row in airfoil_dist.iterrows():
        r_R = row['r/R']
        contour_file = row['Contour file']

        dat_name = os.path.splitext(contour_file)[0] + '.dat'
        csv_to_dat(os.path.join(airfoil_dir, contour_file),
                   os.path.join(dat_temp_dir, dat_name))

        chord_R = float(chord_interp(r_R))
        V, Re, Mach = local_conditions(r_R, chord_R, R_tip, rpm, J, rho, mu,
                                       a_sound, re_floor)

        tag = f"rR{r_R:.4f}".replace('.', '')
        polar_csv_name = f"polar_{tag}.csv"
        polar_txt_name = f"_polar_{tag}.txt"
        polar_txt_path = os.path.join(dat_temp_dir, polar_txt_name)

        floored = " (Re floored)" if Re == re_floor else ""
        print(f"\n  r/R = {r_R:.4f}   {contour_file}")
        print(f"    c/R = {chord_R:.5f}   V = {V:.2f} m/s   Re = {Re:.0f}{floored}   "
              f"Mach = {Mach:.4f}")

        if run_xfoil(xfoil_exe, dat_temp_dir, dat_name, polar_txt_name, Re, Mach,
                     alpha_i, alpha_f, alpha_step, n_iter, Ncrit):
            polar_df = (parse_xfoil_polar(polar_txt_path)
                        .drop_duplicates(subset='Alpha')
                        .sort_values('Alpha')
                        .reset_index(drop=True))
            polar_df.to_csv(os.path.join(out_dir, polar_csv_name), index=False)
            # print(f"    -> {polar_csv_name}  ({len(polar_df)} points)")
            aero_files.append(polar_csv_name)
        else:
            print("    XFOIL FAILED -- no polar for this station")
            aero_files.append("")

    shutil.rmtree(dat_temp_dir, ignore_errors=True)

    # Only rewrite 'Aero file' if every station produced a polar -- a partial write
    # leaves blank entries and PropGeom then fails on a file this tool corrupted.
    if any(not f for f in aero_files):
        print(f"\n  {sum(not f for f in aero_files)} station(s) failed -- airfoildist NOT updated")
    else:
        shutil.copy2(airfoil_dist_file, airfoil_dist_file + '.bak')
        airfoil_dist['Aero file'] = aero_files
        airfoil_dist.to_csv(airfoil_dist_file, index=False)
        print(f"\n  backed up -> {os.path.basename(airfoil_dist_file)}.bak")
        print(f"  updated 'Aero file' in {os.path.basename(airfoil_dist_file)}")

    print("\n" + "=" * 70)
    print(airfoil_dist[['r/R', 'Contour file', 'Aero file']].to_string(index=False))
    print("=" * 70)

    return airfoil_dist_file
