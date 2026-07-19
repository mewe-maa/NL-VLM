import numpy as np

class Vehicle:

    """Assembles multiple propellers into a vehicle configuration."""
    
    def __init__(self, propeller_mesh, hub_positions=None, *, spin_directions):
        
        #   -1  ->  CW  (clockwise, negative omega_z)  -> rotor is mirrored across x
        #   +1  ->  CCW (counter-clockwise, positive omega_z) -> base geometry, not mirrored

        self.propeller_mesh = propeller_mesh

        if hub_positions is None:
            self.hub_positions =  np.array([[0, 0, 0]])
        else:
            self.hub_positions = np.array(hub_positions)

        # CW (negative) -> mirror across x;  CCW (positive) -> base geometry.
        self.mirror_flags = [s < 0 for s in spin_directions]

    def generate_vehicle(self):

        """
        Generate the vehicle mesh by creating individual propeller meshes and
        translating them to their hub positions. Rotors are as mirrored
        (CW spin) are reflected across x; the rest use the base geometry (CCW).
        Twist and polar data are carried over to each control point.
        """
        # Initialize quadcopter mesh
        vehicle_mesh = {}

        # Generate the mesh for a single propeller
        single_propeller_mesh = self.propeller_mesh.generate_propeller_mesh()

        # Loop over hub positions and create translated propellers
        for idx, hub_position in enumerate(self.hub_positions):
            propeller_key = f'Propeller_{idx + 1}'  # Unique key for each propeller
            mirror = self.mirror_flags[idx]  # True -> mirror this rotor across x (CW spin)

            # Translate blades
            translated_blades = {}
            for blade_key, blade_data in single_propeller_mesh['Blades'].items():

                # Translate Panels
                translated_panels = {}
                translated_normals = {}
                translated_tangential_vectors = {}
                translated_twist = {}
                translated_polar_data = {}  # Store translated polar data

                for (i, j), panel in blade_data['Panels'].items():
                    # Mirror across x for CW rotors, otherwise just translate.
                    if not mirror:
                        transformed_panel = [np.array(vertex) + hub_position for vertex in panel]
                    else:
                        transformed_panel = [
                            np.array([-vertex[0], vertex[1], vertex[2]]) + hub_position for vertex in panel
                        ]
                    translated_panels[(i, j)] = transformed_panel

                    if not mirror:
                        span_vector = transformed_panel[2] - transformed_panel[0]
                        chord_vector = transformed_panel[3] - transformed_panel[1]
                        normal = np.cross(span_vector, chord_vector)
                    else:
                        span_vector = transformed_panel[2] - transformed_panel[0]
                        chord_vector = transformed_panel[3] - transformed_panel[1]
                        normal = np.cross(span_vector, chord_vector)

                    # Normalize and ensure direction consistency
                    normal = normal / np.linalg.norm(normal)
                    translated_normals[(i, j)] = normal

                    # Transfer the twist information directly (mesh always provides it)
                    translated_twist[(i, j)] = blade_data['Twist'][(i, j)]

                    # Transfer polar data
                    if 'Polar Data' in blade_data and (i, j) in blade_data['Polar Data']:
                        translated_polar_data[(i, j)] = blade_data['Polar Data'][(i, j)]
                    else:
                        # If polar data is missing, set to None
                        translated_polar_data[(i, j)] = None

                    # Calculate tangential vectors based on the propeller index
                    span_vector_1 = transformed_panel[3] - transformed_panel[0]  # Chordwise edge 1
                    span_vector_2 = transformed_panel[2] - transformed_panel[1]  # Chordwise edge 2
                    avg_span_vector = 0.5 * (span_vector_1 + span_vector_2)

                    chord_vector_1 = transformed_panel[1] - transformed_panel[0]  # Chordwise edge 1
                    chord_vector_2 = transformed_panel[2] - transformed_panel[3]  # Chordwise edge 2
                    avg_chord_vector = 0.5 * (chord_vector_1 + chord_vector_2)

                    tangential_i = avg_span_vector
                    tangential_j = avg_chord_vector 
                
                    translated_tangential_vectors[(i, j)] = {
                        'Tangential i': tangential_i,
                        'Tangential j': tangential_j
                    }

                # Translate Control Points
                translated_control_points = {
                    (i, j): (
                        np.array(cp) + hub_position if not mirror
                        else np.array([-cp[0], cp[1], cp[2]]) + hub_position  # CW rotor: mirror across x
                    )
                    for (i, j), cp in blade_data['Control Points'].items()
                }
        
                # Translate Vortex Rings (only vertices; edge vectors remain the same)
                translated_vortex_rings = {
                    (i, j): {
                        'Vertices': [
                            np.array(vertex) + hub_position if not mirror
                            else np.array([-vertex[0], vertex[1], vertex[2]]) + hub_position  # CW rotor: mirror across x
                            for vertex in vortex_data['Vertices']
                        ]
                
                    }
                    for (i, j), vortex_data in blade_data['Vortex Rings'].items()
                }

                # Add all translated and recalculated components to the blade
                translated_blades[blade_key] = {
                    'Panels': translated_panels,
                    'Normals': translated_normals,
                    'Tangential Vectors': translated_tangential_vectors,
                    'Control Points': translated_control_points,
                    'Vortex Rings': translated_vortex_rings,
                    'Gamma': blade_data['Gamma'],
                    'Twist': translated_twist,
                    'Polar Data': translated_polar_data,
                    'Chord': blade_data['Chord'], 
                    'r_R': blade_data['r_R'],
                }
                
            # Add the translated blades to the quadcopter mesh
            vehicle_mesh[propeller_key] = {
                'Blades': translated_blades,
                'Hub Position': hub_position
            } 
        return vehicle_mesh
