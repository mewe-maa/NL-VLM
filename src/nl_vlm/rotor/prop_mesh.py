import numpy as np
import pandas as pd

class PropMesh:

    """Creates computational mesh for a single propeller."""
    
    def __init__(self, propeller_geometry, span_resolution=17, chord_resolution=9):
        
        self.geometry = propeller_geometry
        self.span_resolution = span_resolution
        self.chord_resolution = chord_resolution

    def generate_propeller_mesh(self):

        """Generate the mesh for a single propeller with all blades."""

        # Generate blade surfaces from geometry
        X_flats, Y_flats, Z_flats, r_R_interpolated = self.generate_blade_surfaces(
            span_resolution=self.span_resolution,
            chord_resolution=self.chord_resolution
        )
        
        # Create mesh for each blade
        blades = {}
        for i in range(self.geometry.num_blades):
            blade_key = f'Blade_{i+1}'
            blades[blade_key] = self._compute_blade_mesh(X_flats[i], Y_flats[i], Z_flats[i], r_R_interpolated)

        return {
            'Blades': blades,
        }
    
    def generate_flat_blade_surface(self, span_resolution, chord_resolution):

        """Generate a flat blade surface following the mean camber line, for one blade."""

        geom = self.geometry
        # Cosine spacing in the radial direction
        theta = np.linspace(0, np.pi, span_resolution)
        r_R_interpolated = (geom.R_hub / geom.R_tip) + (1.0 - geom.R_hub / geom.R_tip) * (1 - np.cos(theta)) / 2

        X_flat, Y_flat, Z_flat = [], [], []

        for r_R in r_R_interpolated:
            # Sectional properties
            r_actual = r_R * geom.R_tip
            chord_length = geom.chord_spline(r_R) * geom.R_tip
            twist_angle = -np.radians(geom.pitch_spline(r_R))
            height = geom.height_spline(r_R) * geom.R_tip if geom.height_spline else 0
            sweep = geom.sweep_spline(r_R) * geom.R_tip if geom.sweep_spline else 0

            # Mean camber line, resampled to the chord resolution
            midpoints_df = self.compute_pairwise_midpoints_unsorted(r_R)
            x_c_original = midpoints_df['x/c'].values
            y_mid_original = midpoints_df['y_c'].values
            x_c_resampled = np.linspace(x_c_original.min(), x_c_original.max(), chord_resolution)
            y_mid_resampled = np.interp(x_c_resampled, x_c_original, y_mid_original)

            # Scale to chord
            x_scaled = x_c_resampled * chord_length
            z_scaled = y_mid_resampled * chord_length

            # Apply twist
            x_rotated = x_scaled * np.cos(twist_angle) - z_scaled * np.sin(twist_angle)
            z_rotated = x_scaled * np.sin(twist_angle) + z_scaled * np.cos(twist_angle)

            # Apply sweep and height
            x_final = x_rotated - sweep
            y_final = np.full_like(x_final, r_actual)
            z_final = z_rotated + height

            X_flat.append(x_final)
            Y_flat.append(y_final)
            Z_flat.append(z_final)

        return np.array(X_flat), np.array(Y_flat), np.array(Z_flat), r_R_interpolated

    def generate_blade_surfaces(self, span_resolution=17, chord_resolution=9):

        """Generate flat blade surfaces for all blades by rotating the first blade."""

        X1_flat, Y1_flat, Z1_flat, r_R_interpolated = self.generate_flat_blade_surface(span_resolution, chord_resolution)

        X_flats = [X1_flat]
        Y_flats = [Y1_flat]
        Z_flats = [Z1_flat]

        # Angular spacing between blades, then generate the remaining blades
        angle_spacing = 2 * np.pi / self.geometry.num_blades

        for i in range(1, self.geometry.num_blades):
            rotation_angle = i * angle_spacing
            rotation_matrix = np.array([
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
                [0, 0, 1]
            ])
            points = np.stack([X1_flat.ravel(), Y1_flat.ravel(), Z1_flat.ravel()], axis=1)
            points_rotated = points @ rotation_matrix.T
            X_flats.append(points_rotated[:, 0].reshape(X1_flat.shape))
            Y_flats.append(points_rotated[:, 1].reshape(Y1_flat.shape))
            Z_flats.append(points_rotated[:, 2].reshape(Z1_flat.shape))

        return X_flats, Y_flats, Z_flats, r_R_interpolated
    
    def _compute_blade_mesh(self, X_flat, Y_flat, Z_flat, r_R_interpolated):

        """Create panels, control points, vortex rings, and normals for a single blade."""
        
        panels = {}
        control_points = {}
        vortex_rings = {}
        normals = {}
        gamma = {}
        tangential_vectors = {}
        twist = {}
        r_R = {}

        spanwise_points, chordwise_points = X_flat.shape

        # Chord stored as a 2D array indexed [chordwise i, spanwise j]
        chord = np.zeros((chordwise_points - 1, spanwise_points - 1))
  
        # Sectional properties (once per spanwise station)
        sectional_r_R = {}
        sectional_twist = {}
        sectional_chord = {}
        
        for j in range(spanwise_points - 1):
            r_R_val = 0.5 * (r_R_interpolated[j] + r_R_interpolated[j + 1])
            sectional_r_R[j] = r_R_val
            sectional_twist[j] = float(self.geometry.pitch_spline(r_R_val))
            sectional_chord[j] = float(self.geometry.chord_spline(r_R_val) * self.geometry.R_tip)

        for i in range(chordwise_points - 1):
            for j in range(spanwise_points - 1):
                
                # Indexing logic
                leading_j = j
                trailing_j = j + 1
                top_i = i
                bottom_i = i + 1

                # Panel corners
                panel_corners = [
                    [X_flat[leading_j, top_i], Y_flat[leading_j, top_i], Z_flat[leading_j, top_i]],
                    [X_flat[leading_j, bottom_i], Y_flat[leading_j, bottom_i], Z_flat[leading_j, bottom_i]],
                    [X_flat[trailing_j, bottom_i], Y_flat[trailing_j, bottom_i], Z_flat[trailing_j, bottom_i]],
                    [X_flat[trailing_j, top_i], Y_flat[trailing_j, top_i], Z_flat[trailing_j, top_i]]
                ]

                # Panel key
                panel_index = (i, j) 
                panels[panel_index] = panel_corners

                # Control point (3/4 chord position)
                span_mid = (np.array(panel_corners[0]) + np.array(panel_corners[3])) / 2
                chord_mid = (np.array(panel_corners[1]) + np.array(panel_corners[2])) / 2
                control_point = span_mid + 0.75 * (chord_mid - span_mid)
                control_points[panel_index] = control_point
                
                # Twist and chord at this spanwise station (sectional)
                r_R[panel_index] = sectional_r_R[j]
                twist[panel_index] = sectional_twist[j]
                chord[panel_index] = sectional_chord[j]
                
                # Vortex ring (1/4 chord position)
                leading_edge_start = np.array(panel_corners[0]) + 0.25 * (np.array(panel_corners[1]) - np.array(panel_corners[0]))
                leading_edge_end = np.array(panel_corners[3]) + 0.25 * (np.array(panel_corners[2]) - np.array(panel_corners[3]))
                trailing_edge_start = np.array(panel_corners[1]) + 0.25 * (np.array(panel_corners[1]) - np.array(panel_corners[0]))
                trailing_edge_end = np.array(panel_corners[2]) + 0.25 * (np.array(panel_corners[2]) - np.array(panel_corners[3]))

                vortex_ring = {
                    'Vertices': [
                        leading_edge_start,
                        trailing_edge_start,
                        trailing_edge_end,
                        leading_edge_end
                    ],

                }
                vortex_rings[panel_index] = vortex_ring

                # Normal vector
                span_vector = np.array(panel_corners[2]) - np.array(panel_corners[0])
                chord_vector = np.array(panel_corners[3]) - np.array(panel_corners[1])
                normal = np.cross(span_vector, chord_vector)
                normals[panel_index] = normal / np.linalg.norm(normal)

                # Initialize gamma
                gamma[panel_index] = 1.0

                # Tangential vectors
                span_vector_1 = np.array(panel_corners[3]) - np.array(panel_corners[0])
                span_vector_2 = np.array(panel_corners[2]) - np.array(panel_corners[1])
                span_vector = 0.5 * (span_vector_1 + span_vector_2)
                tangential_i = span_vector

                chord_vector_1 = np.array(panel_corners[1]) - np.array(panel_corners[0])
                chord_vector_2 = np.array(panel_corners[2]) - np.array(panel_corners[3])
                chord_vector = 0.5 * (chord_vector_1 + chord_vector_2)
                tangential_j = chord_vector

                tangential_vectors[panel_index] = {
                    'Tangential i': tangential_i,
                    'Tangential j': tangential_j
                }

        return {
            'Panels': panels,
            'Control Points': control_points,
            'Vortex Rings': vortex_rings,   
            'Normals': normals,
            'Gamma': gamma,
            'Tangential Vectors': tangential_vectors,
            'Twist': twist,
            'Chord': chord,
            'r_R': r_R,
        }

    def compute_pairwise_midpoints_unsorted(self, r_R_target):

        """Compute the pairwise midpoints of the y/c values for the camber line."""

        airfoil = self.geometry._interpolate_airfoil(r_R_target)

        x_c = airfoil['x/c'].values
        y_c = airfoil['y/c'].values

        n = len(x_c)
        x_c_pair1 = []
        y_mid = []

        for i in range((n + 1) // 2):
            x_c_pair1.append(x_c[i])
            y_mid.append((y_c[i] + y_c[n - 1 - i]) / 2)

        return pd.DataFrame({
            'x/c': x_c_pair1,
            'y_c': y_mid
        })



