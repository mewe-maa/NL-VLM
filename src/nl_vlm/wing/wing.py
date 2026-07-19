import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class SimpleAircraftGeometry:

    def __init__(self, main_wing_span=13.53, main_wing_chord=0.8787, sweep_angle=10.0,
                 h_tail_span=3.05, h_tail_chord=1.0,
                 v_tail_height=1.61, v_tail_chord=1.0,
                 wing_to_vtail_distance=4.937,
                 *, surfaces, airfoil_path, polar_path):
        """Aircraft geometry -- pass in the sizing dimensions and the airfoil/polar files.

        Dimensions [m, deg] (defaults = NASA tilt-wing):
            main_wing_span, main_wing_chord, sweep_angle   main wing
            h_tail_span, h_tail_chord                      horizontal tail
            v_tail_height, v_tail_chord                    vertical tail
            wing_to_vtail_distance                         x-offset of the tail from the wing
        Airfoil + polar per surface (REQUIRED -- passed from the notebook, nothing hardcoded here):
            surfaces  {surface: {'airfoil': contour_file, 'polar': polar_file}} for
                      'main_wing', 'horizontal_tail', 'vertical_tail'
            airfoil_path / polar_path  folders the above file names are read from
        """

        # Main-wing / tail sizing
        self.main_wing_span = main_wing_span
        self.main_wing_chord = main_wing_chord
        self.chord = main_wing_chord          # alias kept for back-compatibility
        self.sweep_angle = sweep_angle
        self.h_tail_span = h_tail_span
        self.h_tail_chord = h_tail_chord
        self.v_tail_height = v_tail_height
        self.v_tail_chord = v_tail_chord
        self.wing_to_vtail_distance = wing_to_vtail_distance

        # Per-surface airfoil / polar input (rotor-style: files + folders), passed in by the caller
        self.surfaces = surfaces
        self.airfoil_path = airfoil_path
        self.polar_path = polar_path

        self.airfoil_contours = {}   # surface -> contour DataFrame (x/c, y/c)
        self.polars = {}             # surface -> polar   DataFrame (Alpha, Cl, Cd, Cm, ...)
        self._cl = {}                # surface -> Cl(alpha) interpolator
        self._load_airfoils()

        # aliases used by the surface generators below
        self.main_airfoil  = self.airfoil_contours['main_wing']
        self.htail_airfoil = self.airfoil_contours['horizontal_tail']
        self.vtail_airfoil = self.airfoil_contours['vertical_tail']

    def _load_airfoils(self):
        """Load each surface's airfoil contour + 2D polar from file, and build the Cl(alpha) interpolator."""
        for surface, files in self.surfaces.items():
            self.airfoil_contours[surface] = pd.read_csv(f"{self.airfoil_path}/{files['airfoil']}")
            polar = pd.read_csv(f"{self.polar_path}/{files['polar']}").sort_values('Alpha')
            self.polars[surface] = polar
            self._cl[surface] = interp1d(polar['Alpha'], polar['Cl'], bounds_error=False,
                                         fill_value=(polar['Cl'].iloc[0], polar['Cl'].iloc[-1]))

    # --- accessors the WingVLM solver reads (keyed by the MESH surface key, e.g. 'Main_Wing') ---
    def chord_of(self, surface_key):
        """Reference chord [m] for a mesh surface key ('Main_Wing'/'Horizontal_Tail'/'Vertical_Tail')."""
        return {'main_wing': self.main_wing_chord,
                'horizontal_tail': self.h_tail_chord,
                'vertical_tail': self.v_tail_chord}[surface_key.lower()]

    def cl_interpolator(self, surface_key):
        """Cl(alpha_deg) interpolator for a surface, from its loaded 2D polar."""
        return self._cl[surface_key.lower()]

    def polar_dataframe(self, surface_key):
        """Full 2D polar DataFrame (Alpha, Cl, Cd, Cm, ...) for a surface."""
        return self.polars[surface_key.lower()]

    def compute_wing_camber_line(self, airfoil_data):
        """
        Compute the mean camber line for wing airfoil.
        Similar to compute_pairwise_midpoints_unsorted but for wing airfoils.
        """
        x_c = airfoil_data['x/c'].values
        y_c = airfoil_data['y/c'].values
        
        n = len(x_c)
        x_c_camber = []
        y_camber = []
        
        # Pair the points symmetrically to get camber line
        for i in range((n + 1) // 2):
            x_c_camber.append(x_c[i])
            y_camber.append((y_c[i] + y_c[n - 1 - i]) / 2)
        
        return pd.DataFrame({
            'x/c': x_c_camber,
            'y_c': y_camber
        })

    def generate_wing_surface(self, airfoil_data, span, chord_length, sweep_deg=0, n_span=30, n_chord=50):
        """
        Generate complete wing surface using mean camber line (entire span from -span/2 to +span/2).
        
        Args:
            airfoil_data: DataFrame with 'x/c' and 'y/c' columns
            span: Wing span (total, tip to tip)
            chord_length: Chord length for this surface (NEW PARAMETER)
            sweep_deg: Sweep angle in degrees
            n_span: Number of spanwise stations
            n_chord: Number of chordwise points
        """
        # Create spanwise positions (from -span/2 to +span/2)
        y_positions = np.linspace(-span/2, span/2, n_span)
        
        # Convert sweep angle to radians
        sweep_rad = np.radians(sweep_deg)
        
        # Initialize coordinate arrays - using (n_chord, n_span) layout
        X_flat, Y_flat, Z_flat = [], [], []
        
        # Generate surface
        for i, y_pos in enumerate(y_positions):
            # Get wing properties for this spanwise station
            twist_angle = 0.0  # No twist for now, but could add distribution
            
            sweep_offset = abs(y_pos) * np.tan(sweep_rad)
            camber_df = self.compute_wing_camber_line(airfoil_data)

            x_c_original = camber_df['x/c'].values
            y_camber_original = camber_df['y_c'].values

            _order = np.argsort(x_c_original)
            x_c_original = x_c_original[_order]
            y_camber_original = y_camber_original[_order]
            x_c_new = np.linspace(x_c_original.min(), x_c_original.max(), n_chord)
            y_camber_new = np.interp(x_c_new, x_c_original, y_camber_original)
            
            x_scaled = x_c_new * chord_length
            z_scaled = -y_camber_new * chord_length   # sign so +camber -> +lift at alpha=0 (mesh normals point -z)
            
            twist_rad = np.radians(twist_angle)
            x_rotated = x_scaled * np.cos(twist_rad) - z_scaled * np.sin(twist_rad)
            z_rotated = x_scaled * np.sin(twist_rad) + z_scaled * np.cos(twist_rad)
            
            # Apply sweep offset
            x_final = x_rotated + sweep_offset
            z_final = z_rotated
            y_final = np.full_like(x_final, y_pos)

            # Store in arrays (using column-wise storage)
            X_flat.append(x_final)
            Y_flat.append(y_final)
            Z_flat.append(z_final)
        
        return np.array(X_flat), np.array(Y_flat), np.array(Z_flat)

    def generate_main_wing(self, n_span=30, n_chord=50):
        """Generate complete main wing (entire span) with original chord."""
        return self.generate_wing_surface(
            self.main_airfoil, 
            self.main_wing_span, 
            self.main_wing_chord,  # Use main wing chord
            self.sweep_angle,
            n_span=n_span,
            n_chord=n_chord
        )
    
    def generate_horizontal_tail(self, n_span=30, n_chord=50):
        """Generate complete horizontal tail (entire span) with 1m chord."""
        X, Y, Z = self.generate_wing_surface(
            self.htail_airfoil,
            self.h_tail_span,
            self.h_tail_chord,     # Use horizontal tail chord (1.0m)
            self.sweep_angle,
            n_span=n_span,
            n_chord=n_chord
        )
        # Position horizontal tail
        X += self.wing_to_vtail_distance
        Z += self.v_tail_height
        
        return X, Y, Z
    
    def generate_vertical_tail(self, n_span=30, n_chord=50):
        """Generate vertical tail."""
        X, Y, Z = self.generate_wing_surface(
            self.vtail_airfoil, 
            self.v_tail_height, 
            self.v_tail_chord, 
            sweep_deg=0,
            n_span=n_span,
            n_chord=n_chord
        )
        # Rotate to vertical orientation
        X_new = X + self.wing_to_vtail_distance
        Y_new = np.zeros_like(Y)
        Z_new = Y + self.v_tail_height/2
        
        return X_new, Y_new, Z_new
