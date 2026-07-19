import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

class PropGeom:

    """Defines the physical blade geometry.

    ``polar_path`` is optional. Without it no 2D polar is loaded and
    :attr:`has_polars` is False, which leaves the blade usable for a purely
    inviscid run (the VLM then refuses to enable ``nonlinear_lift`` /
    ``profile_drag``, since both read the polar). Pass it to get the real
    lift slope, stall, and profile drag.
    """

    def __init__(self, airfoil_distribution_file, chorddist_file, pitchdist_file,
                 sweepdist_file=None, heightdist_file=None, airfoil_path=None,
                 polar_path=None, R_tip=None, R_hub=None, num_blades=1):

        self.num_blades = num_blades
        self.R_tip = R_tip
        self.R_hub = R_hub
        self.airfoil_path = airfoil_path
        self.polar_path = polar_path
        
        # Load distribution data
        self.airfoil_distribution = pd.read_csv(airfoil_distribution_file)
        self.chorddist = pd.read_csv(chorddist_file)
        self.pitchdist = pd.read_csv(pitchdist_file)
        self.sweepdist = pd.read_csv(sweepdist_file) if sweepdist_file else None
        self.heightdist = pd.read_csv(heightdist_file) if heightdist_file else None

        # Load airfoil and polars contours and create splines
        self.airfoil_contours = self._load_airfoil_contours()
        self.aero_data = self._load_aero_data()

        self._create_interpolation_splines()

    def _load_airfoil_contours(self):

        """Load airfoil contour data from files."""

        contours = {}
        for _, row in self.airfoil_distribution.iterrows():
            r_R = row['r/R']
            contour_file = row['Contour file']
            if self.airfoil_path:
                full_path = f"{self.airfoil_path}/{contour_file}"
            else:
                full_path = contour_file
            airfoil_data = pd.read_csv(full_path)
            contours[r_R] = airfoil_data
        return contours

    @property
    def has_polars(self):
        """True if 2D polar data was loaded (i.e. ``polar_path`` was given)."""
        return bool(self.aero_data)

    def _load_aero_data(self):

        """Load aerodynamic polar data from files.

        Returns an empty dict when ``polar_path`` is None: the polars are
        optional, and a blade without them still meshes and solves inviscidly.
        """

        if self.polar_path is None:
            return {}

        aero_data = {}
        for _, row in self.airfoil_distribution.iterrows():
            r_R = row['r/R']
            aero_file = row['Aero file']
            aero_data[r_R] = pd.read_csv(f"{self.polar_path}/{aero_file}")
        return aero_data
   
    def _create_interpolation_splines(self):

        """Create smooth splines for blade geometry distributions."""

        if self.heightdist is not None:
            self.height_spline = UnivariateSpline(
            self.heightdist['r/R'],
            self.heightdist['z/R  (height of leading edge from top face of hub)'],
            k=4, s=5e-7
            )
        else:
            self.height_spline = None
        
        self.cl_splines = {}
        self.cd_splines = {}

        # A polar truncated before stall (XFOIL died early, e.g. a thin tip
        # section) poisons every lookup at that station: get_cl/get_cd would
        # linearly extrapolate its last pre-stall segment to any alpha. Dropping
        # the station instead lets the nearest healthy one govern through the
        # r/R clamping in get_cl/get_cd.
        MIN_ALPHA_RANGE = 15.0   # deg, table must span at least this
        MIN_ALPHA_MAX = 10.0     # deg, table must reach at least this

        usable = {}
        for r_R, polar_df in self.aero_data.items():
            alpha = polar_df['Alpha']
            if (alpha.max() - alpha.min() >= MIN_ALPHA_RANGE
                    and alpha.max() >= MIN_ALPHA_MAX):
                usable[r_R] = polar_df
            else:
                print(f"WARNING: polar at r/R={r_R:g} only covers alpha "
                      f"[{alpha.min():g} .. {alpha.max():g}] deg -- truncated "
                      f"before stall; station dropped, nearest station governs.")
        if not usable and self.aero_data:
            print("WARNING: every polar failed the coverage check; keeping all.")
            usable = self.aero_data

        for r_R, polar_df in usable.items():
            self.cl_splines[r_R] = interp1d(
                polar_df['Alpha'],
                polar_df['Cl'],
                kind='linear',
                fill_value='extrapolate'
            )
            self.cd_splines[r_R] = interp1d(
                polar_df['Alpha'],
                polar_df['Cd'],
                kind='linear',
                fill_value='extrapolate'
            )

        self.chord_spline  = interp1d(self.chorddist['r/R'], self.chorddist['c/R'], kind='linear', fill_value='extrapolate')
        self.pitch_spline  = interp1d(self.pitchdist['r/R'], self.pitchdist['twist (deg)'], kind='linear', fill_value='extrapolate')

        if self.sweepdist is not None:
            self.sweep_spline = interp1d(self.sweepdist['r/R'], self.sweepdist['y/R (y-distance of LE from the middle point of hub)'], kind='linear', fill_value='extrapolate')
        else:
            self.sweep_spline = None
    
    def _interpolate_airfoil(self, r_R_target, n_points=100):

        """Interpolate airfoil shape at a given radial position."""
        
        r_R_values = np.array(sorted(self.airfoil_contours.keys()))
        
        lower_idx = np.searchsorted(r_R_values, r_R_target) - 1
        lower_idx = max(0, lower_idx)
        upper_idx = min(lower_idx + 1, len(r_R_values) - 1) 
        
        r_R_lower = r_R_values[lower_idx]
        r_R_upper = r_R_values[upper_idx]
        
        if r_R_lower == r_R_upper:
            airfoil = self.airfoil_contours[r_R_lower]
            t = np.linspace(0, 1, len(airfoil))
            t_new = np.linspace(0, 1, n_points)
            x_new = np.interp(t_new, t, airfoil['x/c'].values)
            y_new = np.interp(t_new, t, airfoil['y/c'].values)

            spl_x = UnivariateSpline(t_new, x_new, k=4, s=5e-7)
            spl_y = UnivariateSpline(t_new, y_new, k=4, s=5e-7)
            t_fine = np.linspace(0, 1, n_points)

            return pd.DataFrame({'x/c': spl_x(t_fine), 'y/c': spl_y(t_fine)})
        
        airfoil_lower = self.airfoil_contours[r_R_lower]
        airfoil_upper = self.airfoil_contours[r_R_upper]
        
        t_lower = np.linspace(0, 1, len(airfoil_lower))
        t_upper = np.linspace(0, 1, len(airfoil_upper))
        t_new = np.linspace(0, 1, n_points)
        
        x_lower = np.interp(t_new, t_lower, airfoil_lower['x/c'].values)
        y_lower = np.interp(t_new, t_lower, airfoil_lower['y/c'].values)
        
        x_upper = np.interp(t_new, t_upper, airfoil_upper['x/c'].values)
        y_upper = np.interp(t_new, t_upper, airfoil_upper['y/c'].values)
        
        # Linear interpolation between the two airfoils
        weight = (r_R_target - r_R_lower) / (r_R_upper - r_R_lower)
        x_interp = x_lower * (1 - weight) + x_upper * weight
        y_interp = y_lower * (1 - weight) + y_upper * weight
        
        return pd.DataFrame({'x/c': x_interp, 'y/c': y_interp})
 
    def get_cl(self, r_R, alpha):

        stations = sorted(self.cl_splines.keys())
        # Clamp to bounds
        if r_R <= stations[0]:
            return float(self.cl_splines[stations[0]](alpha))
        if r_R >= stations[-1]:
            return float(self.cl_splines[stations[-1]](alpha))
        # Find bracketing stations
        for i in range(len(stations) - 1):
            if stations[i] <= r_R <= stations[i + 1]:
                r_low, r_high = stations[i], stations[i + 1]
                cl_low = self.cl_splines[r_low](alpha)
                cl_high = self.cl_splines[r_high](alpha)
                weight = (r_R - r_low) / (r_high - r_low)
                return float((1 - weight) * cl_low + weight * cl_high)

    def get_cd(self, r_R, alpha):
        """Sectional profile drag coefficient from the 2D polar (mirror of get_cl)."""
        stations = sorted(self.cd_splines.keys())
        if r_R <= stations[0]:
            return float(self.cd_splines[stations[0]](alpha))
        if r_R >= stations[-1]:
            return float(self.cd_splines[stations[-1]](alpha))
        for i in range(len(stations) - 1):
            if stations[i] <= r_R <= stations[i + 1]:
                r_low, r_high = stations[i], stations[i + 1]
                cd_low = self.cd_splines[r_low](alpha)
                cd_high = self.cd_splines[r_high](alpha)
                weight = (r_R - r_low) / (r_high - r_low)
                return float((1 - weight) * cd_low + weight * cd_high)