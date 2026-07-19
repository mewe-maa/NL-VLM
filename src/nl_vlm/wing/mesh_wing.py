import numpy as np

class WingMesh:
    def __init__(self, aircraft_geometry):
        """Wing mesh system. `aircraft_geometry` is a SimpleAircraftGeometry instance."""
        self.aircraft_geometry = aircraft_geometry

    def _compute_wing_mesh(self, X_surface, Y_surface, Z_surface, wing_type='main_wing'):
        """
        Compute wing mesh data including panels, control points, vortex rings, and normals.
        Similar to _compute_blade_mesh in propeller.
        
        Args:
            X_surface, Y_surface, Z_surface: Wing surface coordinates
            wing_type: Type of wing ('main_wing', 'horizontal_tail', 'vertical_tail')
        """
        panels = {}
        control_points = {}
        vortex_rings = {}
        normals = {}
        gamma = {}
        tangential_vectors = {}
        twist = {}
        chord = {}

        # Shape info for looping
        chordwise_points, spanwise_points = X_surface.shape

        # Loop through spanwise and chordwise sections
        for i in range(spanwise_points - 1):  
            for j in range(chordwise_points - 1): 
                leading_j = j  # Leading edge along span (j-direction)
                trailing_j = j + 1  # Trailing edge along span (j-direction)
                top_i = i  # Top edge along chord (i-direction)
                bottom_i = i + 1  # Bottom edge along chord (i-direction)

                # Panel corners (counter-clockwise from top-left)
                panel_corners = [
                    [X_surface[leading_j, top_i], Y_surface[leading_j, top_i], Z_surface[leading_j, top_i]],  # Top-left
                    [X_surface[leading_j, bottom_i], Y_surface[leading_j, bottom_i], Z_surface[leading_j, bottom_i]],  # Top-right
                    [X_surface[trailing_j, bottom_i], Y_surface[trailing_j, bottom_i], Z_surface[trailing_j, bottom_i]],  # Bottom-right
                    [X_surface[trailing_j, top_i], Y_surface[trailing_j, top_i], Z_surface[trailing_j, top_i]]  # Bottom-left
                ]
                # Panel key
                panel_index = (i, j)
                panels[panel_index] = panel_corners

                # Corners as vectors: convert once, then reuse (c0=top-left, c1=top-right,
                # c2=bottom-right, c3=bottom-left) instead of calling np.array on each corner repeatedly.
                c0, c1, c2, c3 = (np.array(pc) for pc in panel_corners)

                # Control point (3/4 chord position)
                leading_edge = (c0 + c3) / 2    # LE midpoint
                trailing_edge = (c1 + c2) / 2   # TE midpoint
                control_point = leading_edge + 0.75 * (trailing_edge - leading_edge)
                control_points[panel_index] = control_point

                # Basic wing parameters (simplified - no complex distributions)
                twist[panel_index] = 0.0                      # no twist for the simplified wing
                chord[panel_index] = self.aircraft_geometry.chord

                # Vortex ring: 1/4-chord offset from each corner
                leading_edge_start = c0 + 0.25 * (c1 - c0)   # 1/4 behind the leading edge, top
                leading_edge_end = c3 + 0.25 * (c2 - c3)     # 1/4 behind the leading edge, bottom
                trailing_edge_start = c1 + 0.25 * (c1 - c0)  # 1/4 behind the trailing edge, top
                trailing_edge_end = c2 + 0.25 * (c2 - c3)    # 1/4 behind the trailing edge, bottom
                vortex_rings[panel_index] = {
                    'Vertices': [leading_edge_start, trailing_edge_start, trailing_edge_end, leading_edge_end],
                    'Edge Vectors': [
                        trailing_edge_start - leading_edge_start,  # top edge
                        trailing_edge_end - trailing_edge_start,   # trailing edge
                        leading_edge_end - trailing_edge_end,      # bottom edge
                        leading_edge_start - leading_edge_end,     # leading edge
                    ],
                }

                # Normal vector (from the panel diagonals)
                normal = np.cross(c2 - c0, c3 - c1)
                normals[panel_index] = normal / np.linalg.norm(normal)

                # Initialize gamma (circulation strength)
                gamma[panel_index] = 1.0

                # Tangential vectors: spanwise (i) and chordwise (j), each the average of its two edges
                tangential_vectors[panel_index] = {
                    'Tangential i': 0.5 * ((c3 - c0) + (c2 - c1)),
                    'Tangential j': 0.5 * ((c1 - c0) + (c2 - c3)),
                }

        # Return the dictionary for the wing mesh
        return {
            'Panels': panels,
            'Control Points': control_points,
            'Vortex Rings': vortex_rings,   
            'Normals': normals,
            'Gamma': gamma,
            'Tangential Vectors': tangential_vectors,
            'Twist': twist,
            'Chord': chord,
        }
    
    def generate_wing_mesh(self, wing_type='main_wing', span_resolution=20, chord_resolution=12):
        """
        Generate mesh for a single wing surface.
        
        Args:
            wing_type: 'main_wing', 'horizontal_tail', or 'vertical_tail'
            span_resolution: Number of spanwise stations
            chord_resolution: Number of chordwise points
        """
        # Generate wing surface directly with desired resolution
        if wing_type == 'main_wing':
            X, Y, Z = self.aircraft_geometry.generate_main_wing(
                n_span=span_resolution,
                n_chord=chord_resolution
            )
        elif wing_type == 'horizontal_tail':
            X, Y, Z = self.aircraft_geometry.generate_horizontal_tail(
                n_span=span_resolution,
                n_chord=chord_resolution
            )
        elif wing_type == 'vertical_tail':
            X, Y, Z = self.aircraft_geometry.generate_vertical_tail(
                n_span=span_resolution,
                n_chord=chord_resolution
            )
    
        # Compute mesh data directly
        wing_mesh_data = self._compute_wing_mesh(X, Y, Z, wing_type)
        
        return {
            wing_type: wing_mesh_data,
            'Surface': (X, Y, Z)
        }

    def generate_complete_aircraft_mesh(self, span_resolution=15, chord_resolution=8):
        """
        Generate complete aircraft mesh (all surfaces) - NO MIRRORING.
        """
        aircraft_mesh = {}
        
        # Generate main wing mesh (complete wing)
        main_wing_data = self.generate_wing_mesh('main_wing', span_resolution, chord_resolution)
        aircraft_mesh['Main_Wing'] = main_wing_data['main_wing']
        
        # Generate horizontal tail mesh (complete tail)
        h_tail_data = self.generate_wing_mesh('horizontal_tail', span_resolution, chord_resolution)
        aircraft_mesh['Horizontal_Tail'] = h_tail_data['horizontal_tail']
        
        # Generate vertical tail mesh (single surface)
        v_tail_data = self.generate_wing_mesh('vertical_tail', span_resolution, chord_resolution)
        aircraft_mesh['Vertical_Tail'] = v_tail_data['vertical_tail']
        
        return aircraft_mesh
