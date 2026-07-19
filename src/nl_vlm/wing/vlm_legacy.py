import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import copy
from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.pyplot as plt
from wind import WindField

class WingVLM:
    def __init__(self, aircraft_mesh):
        """Initialize the Wing VLM solver with the given aircraft mesh."""
        self.aircraft_mesh = aircraft_mesh
    
    @staticmethod
    @nb.jit(nopython=True)
    def biot_savart(r1, r2, r0, gamma):
        """
        Calculate induced velocity using Biot-Savart law with a viscous core model
        
        Args:
            r1: Vector from vortex end point to field point
            r2: Vector from vortex start point to field point  
            r0: Vortex segment vector (r2 - r1)
            gamma: Vortex strength
        
        Returns:
            induced_velocity: 3D velocity vector induced at field point
        """
        cross_r1_r2 = np.cross(r1, r2)
        norm_cross_r1_r2 = np.linalg.norm(cross_r1_r2)
        norm_r0 = np.linalg.norm(r0)
        norm_r1 = np.linalg.norm(r1)
        norm_r2 = np.linalg.norm(r2)
        epsilon = 1e-9
        
        # Core radius parameters
        alpha = 0.05
        rc = alpha * norm_r0
        n = 2

        # Calculate induced velocity
        dot_term = np.dot(r0, (r1/np.linalg.norm(r1) - r2/np.linalg.norm(r2)))
        denominator = norm_cross_r1_r2 ** 2
        # denominator = ((norm_cross_r1_r2**(2*n)) + ((rc*norm_r0)**(2*n)))**(1/n)

        
        induced_velocity = (gamma / (4 * np.pi)) * cross_r1_r2 * dot_term / denominator
        
        return induced_velocity
    
    def calculate_bound_to_bound_induced_velocity_matrix(self, aircraft_mesh, alpha_deg=0.0):
        """
        Calculate the global induced velocity matrix for all control points and vortex rings.
        Optimized for wing analysis.
        
        Args:
            aircraft_mesh: Complete aircraft mesh data
            alpha_deg: Angle of attack in degrees
            
        Returns:
            dict: Global influence matrices for each wing surface
        """
        global_matrices = {}
        
        for surface_key, surface_data in aircraft_mesh.items():
            control_points = []
            vortex_rings = []
            panel_indices = []
            
            # Collect control points and vortex rings
            for panel_index, control_point in surface_data['Control Points'].items():
                control_points.append((panel_index, control_point))
                vortex_rings.append((panel_index, surface_data['Vortex Rings'][panel_index]))
                panel_indices.append(panel_index)
            
            # Initialize matrix dimensions
            num_points = len(control_points)
            global_matrix = np.zeros((num_points, num_points, 3))
            
            # Find maximum spanwise index for trailing edge detection
            max_spanwise = max(panel_idx[0] for panel_idx in panel_indices)
            
            # Calculate induced velocities
            for i, (cp_index, control_point) in enumerate(control_points):
                for j, (vr_index, vortex_ring) in enumerate(vortex_rings):
                    # Initialize induced velocity for this pair
                    total_induced_velocity = np.zeros(3)
                    
                    # Extract vertices and process each vortex filament
                    vertices = vortex_ring['Vertices']
                    
                    # Handle panel indexing (i, j) format
                    spanwise_idx = vr_index[0]
                    chordwise_idx = vr_index[1]
                    is_trailing_edge = (spanwise_idx == max_spanwise)
                    
                    # Process each filament of the vortex ring
                    for k in range(4):
                        vertex_start = np.array(vertices[k])
                        vertex_end = np.array(vertices[(k + 1) % 4])
                        
                        r1 = control_point - vertex_end
                        r2 = control_point - vertex_start
                        r0 = r1 - r2
                        
                        # Calculate induced velocity contribution
                        induced_velocity = self.biot_savart(r1, r2, r0, gamma=1.0)
                        total_induced_velocity += induced_velocity
                    
                    # Handle trailing edge with Kutta condition
                    if is_trailing_edge:
                        # Extend trailing edge vortices to infinity
                        infinity_factor = 100
                        
                        # Get flow direction (simplified - along x-axis with angle of attack)
                        alpha_rad = np.radians(alpha_deg)
                        flow_direction = np.array([np.cos(alpha_rad), 0, np.sin(alpha_rad)])
                        
                        # Create semi-infinite vortices
                        new_vertices = [
                            np.array(vertices[1]),  # Point 0 (original trailing edge point)
                            np.array(vertices[1]) + infinity_factor * flow_direction,  # Point 1 (to infinity)
                            np.array(vertices[2]) + infinity_factor * flow_direction,  # Point 2 (to infinity)
                            np.array(vertices[2])   # Point 3 (original trailing edge point)
                        ]
                        
                        # Calculate influence from semi-infinite vortices
                        for k in range(4):
                            vertex_start = new_vertices[k]
                            vertex_end = new_vertices[(k + 1) % 4]
                            
                            r1 = control_point - vertex_end
                            r2 = control_point - vertex_start
                            r0 = vertex_start - vertex_end
                            
                            # Calculate induced velocity contribution
                            induced_velocity = self.biot_savart(r1, r2, r0, gamma=1.0)
                            total_induced_velocity += induced_velocity
                    
                    # Store in global matrix
                    global_matrix[i, j] = total_induced_velocity
            
            global_matrices[surface_key] = global_matrix
        
        return global_matrices
    
    def calculate_gamma(self, aircraft_mesh, bound_to_bound_global_matrices, 
                       alpha_deg, airspeed, wind_field, reference_point):
        """
        Calculate gamma (circulation strength) for each wing surface using the Neumann boundary condition.

        """
        gamma_matrices = {}
        induced_velocities = {}
        
        # Convert angles to radians
        alpha_rad = np.radians(alpha_deg)
        
        # Define freestream velocity vector
        freestream_velocity = airspeed 
        wind_func = WindField.update_wind_function(wind_field, reference_point)
        # wind_velocity = wind_field

        for surface_key, surface_data in aircraft_mesh.items():
            # Collect data for control points
            control_points = []
            normals = []
            rhs = []
            
            for panel_index, control_point in surface_data['Control Points'].items():
                control_point = np.array(control_point)
                control_points.append(control_point)
                
                all_panel_indices = list(surface_data['Control Points'].keys())
                min_i = min(idx[0] for idx in all_panel_indices)
                min_j = min(idx[1] for idx in all_panel_indices)
                max_i = max(idx[0] for idx in all_panel_indices)
                max_j = max(idx[1] for idx in all_panel_indices)

                control_point_right = surface_data['Control Points'][(panel_index[0], max_j)]
                control_point_left  = surface_data['Control Points'][(panel_index[0], min_j)]

                wind_velocity = wind_func(control_point + reference_point)
                wind_velocity_jacob = wind_field.get_jacobian_approximated_velocity(control_point + reference_point, reference_point)

            
                if (control_point[1]) > 0:
                    wind_velocity_cg = wind_field.get_wind_velocity(reference_point)
                    wind_velcoity_tip = wind_func(control_point_right + reference_point)
                    gradient = (wind_velcoity_tip - wind_velocity_cg) / (control_point_right)
                    wind_velocity_grad = wind_velocity_cg + gradient * control_point

                elif (control_point[1]) < 0:
                    wind_velocity_cg = wind_field.get_wind_velocity(reference_point)
                    wind_velcoity_tip = wind_func(control_point_left + reference_point)
                    gradient = (wind_velcoity_tip - wind_velocity_cg) / (control_point_left)
                    wind_velocity_grad = wind_velocity_cg + gradient * control_point

                print('wind_velocity', wind_velocity )
                print('wind_velocity_jacob', wind_velocity_jacob)
                print('wind_velocity_grad', wind_velocity_grad )

                print('-------')
                wind_velocity =  np.array([0, 0, 0 ])
                

                # Get panel normal
                normal = surface_data['Normals'][panel_index]
                normals.append(normal)

                # Store freestream velocity for each panel
                if "Freestream_Velocity" not in surface_data:
                    surface_data['Freestream_Velocity'] = {}
                surface_data['Freestream_Velocity'][panel_index] = freestream_velocity
                
                if "Wind Velocity" not in surface_data:
                    surface_data['Wind Velocity'] = {}
                surface_data['Wind Velocity'][panel_index] = wind_velocity

                # Calculate right-hand side of boundary condition
                # Neumann condition: V_n = 0 (no flow through surface)
                # print('wind_velocity', wind_velocity)
                velocity_term = freestream_velocity + wind_velocity
                print('freestream_velocity', freestream_velocity)
                velocity_term = -freestream_velocity + wind_velocity
                
                rhs_value = -np.dot(velocity_term, normal)
                rhs.append(rhs_value)
            
            # Convert to numpy arrays
            normals = np.array(normals)
            rhs = np.array(rhs).reshape(-1, 1)
            
            # Get influence matrix for this surface
            bound_to_bound_induced_matrix = bound_to_bound_global_matrices[surface_key]
            
            # Create boundary condition matrix
            A_matrix = np.einsum('ijk,ik->ij', bound_to_bound_induced_matrix, normals)
            
            # Solve for circulation strengths
            gamma = np.linalg.solve(A_matrix, rhs)
            
            # Store results
            gamma_matrices[surface_key] = gamma.flatten()
            
            # Calculate induced velocities
            induced_vel = np.einsum('ijk,j->ik', bound_to_bound_induced_matrix, gamma.flatten())
            induced_velocities[surface_key] = induced_vel
            
            # Update gamma values in the mesh
            gamma_index = 0
            for panel_index in surface_data['Control Points'].keys():
                if 'Gamma' not in surface_data:
                    surface_data['Gamma'] = {}
                if 'Induced_Velocities' not in surface_data:
                    surface_data['Induced_Velocities'] = {}
                
                surface_data['Gamma'][panel_index] = float(gamma[gamma_index])
                surface_data['Induced_Velocities'][panel_index] = induced_vel[gamma_index]
                gamma_index += 1
        
        return gamma_matrices, induced_velocities
    
    def calculate_pressure_difference(self, aircraft_mesh, alpha_deg, airspeed, rho, wind_field, reference_point):
        """
        Calculate the pressure difference for each panel using Bernoulli's equation.

        Returns:
            dict: Pressure differences for each surface
        """
        pressure_differences = {}
        
        # Convert angle to radians
        alpha_rad = np.radians(alpha_deg)
        
        for surface_key, surface_data in aircraft_mesh.items():
            if 'Pressure_Difference' not in surface_data:
                surface_data['Pressure_Difference'] = {}
            control_points = surface_data['Control Points']
            normals = surface_data['Normals']
            tangential_vectors = surface_data['Tangential Vectors']
            gamma_old = surface_data.get('Gamma Old', {})
            
            tangential_vectors = surface_data['Tangential Vectors']    
            for panel_index, control_point in surface_data['Control Points'].items():
                
                tangent_span = tangential_vectors[panel_index]['Tangential i']
                tangent_chord = tangential_vectors[panel_index]['Tangential j']
                normal = surface_data['Normals'][panel_index]
                
                # Get velocities
                freestream_velocity = surface_data['Freestream_Velocity'][panel_index]
                induced_velocity = surface_data['Induced_Velocities'][panel_index]
                wind_velocity = surface_data['Wind Velocity'][panel_index]

                # Total velocity at control point
                total_velocity = -freestream_velocity + wind_velocity
                
                # Get current and previous circulation values
                gamma_current = surface_data['Gamma'][panel_index]
                gamma_previous_span = surface_data['Gamma'].get((panel_index[0] - 1, panel_index[1]), 0) if panel_index[0] > 0 else 0
                gamma_previous_chord = surface_data['Gamma'].get((panel_index[0], panel_index[1] - 1), 0) if panel_index[1] > 0 else 0

                # Calculate gamma differences (normalized)
                gamma_diff_span = (gamma_current - gamma_previous_span) / np.linalg.norm(tangent_chord)
                gamma_diff_chord = (gamma_current - gamma_previous_chord) / np.linalg.norm(tangent_span)

                panel = surface_data['Panels'][panel_index]
                panel_array = np.array(panel)

                panel_center = panel_array.mean(axis=0)
                area_triangle_1 = 0.5 * np.linalg.norm(np.cross(panel_array[1] - panel_array[0], panel_array[3] - panel_array[0]))
                area_triangle_2 = 0.5 * np.linalg.norm(np.cross(panel_array[2] - panel_array[1], panel_array[3] - panel_array[1]))

                panel_area = area_triangle_1 + area_triangle_2
                
                if gamma_old == {}:
                    gamma_previous = 0
                else:
                    gamma_previous = gamma_old[panel_index]       
                
                # gamma_dot = (gamma_current - gamma_previous) / dt

                pressure = rho * (
                    np.dot(total_velocity, tangent_chord / np.linalg.norm(tangent_chord)  * gamma_diff_span) +
                    np.dot(total_velocity, tangent_span / np.linalg.norm(tangent_span) * gamma_diff_chord)     
                )
                
                pressure = np.linalg.norm(rho * (gamma_current - gamma_previous_span) * np.cross((total_velocity + induced_velocity), tangent_span) ) /  (panel_area)
                
                all_panel_indices = list(surface_data['Control Points'].keys())
                max_i = max(idx[0] for idx in all_panel_indices)
                max_j = max(idx[1] for idx in all_panel_indices)
                
                i, j = panel_index
                
                # Detect different types of "last panels"
                is_trailing_edge = (i == max_i)  # Last in spanwise direction
                is_wing_tip = (j == max_j)       # Last in chordwise direction
                is_corner_panel = (i == max_i and j == max_j)  # Corner panel


                surface_data['Pressure_Difference'][panel_index] = pressure
            
            pressure_differences[surface_key] = surface_data['Pressure_Difference']
        
        return pressure_differences
    
    def calculate_total_forces_and_moments(self, aircraft_mesh, alpha_deg, airspeed, 
                                         rho, wind_field, cog, reference_point):
        """
        Calculate aerodynamic forces and moments for each panel and total aircraft.
        
        """
        # First calculate influence matrices and gamma
        bound_to_bound_matrices = self.calculate_bound_to_bound_induced_velocity_matrix(
            aircraft_mesh, alpha_deg)
        
        gamma_matrices, induced_velocities = self.calculate_gamma(
            aircraft_mesh, bound_to_bound_matrices, alpha_deg, airspeed=airspeed, wind_field=wind_field, reference_point=reference_point)
        
        # Calculate pressure differences
        pressure_differences = self.calculate_pressure_difference(
            aircraft_mesh, alpha_deg, airspeed, rho, wind_field=wind_field, reference_point=reference_point)
        
        # Initialize results
        surface_forces_and_moments = {}
        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        control = cog

        for surface_key, surface_data in aircraft_mesh.items():
            # Initialize surface totals
            surface_force = np.zeros(3)
            surface_moment = np.zeros(3)
            
            # Initialize storage for panel forces
            if 'Panel_Forces' not in surface_data:
                surface_data['Panel_Forces'] = {}
            if 'Panel_Moments' not in surface_data:
                surface_data['Panel_Moments'] = {}
            
            for panel_index, pressure_diff in surface_data['Pressure_Difference'].items():
                # Get panel geometry
                panel_corners = surface_data['Panels'][panel_index]
                panel_array = np.array(panel_corners)
                panel_center = panel_array.mean(axis=0)

                # Calculate panel area using cross product
                # Divide quadrilateral into two triangles
                triangle1_area = 0.5 * np.linalg.norm(
                    np.cross(panel_array[1] - panel_array[0], panel_array[3] - panel_array[0]))
                triangle2_area = 0.5 * np.linalg.norm(
                    np.cross(panel_array[2] - panel_array[1], panel_array[3] - panel_array[1]))
                panel_area = triangle1_area + triangle2_area
                
                # Get panel normal (pointing into the flow)
                normal = surface_data['Normals'][panel_index]
                
                # Calculate force on panel
                panel_force = (pressure_diff * panel_area) * normal
                
                # Get control point for moment calculation
                control_point = surface_data['Control Points'][panel_index]
                moment_arm = control_point - control
                
                # Calculate moment about reference point
                panel_moment = np.cross(moment_arm, panel_force)
                
                # Store panel results
                surface_data['Panel_Forces'][panel_index] = panel_force
                surface_data['Panel_Moments'][panel_index] = panel_moment
                
                # Add to surface totals
                surface_force += panel_force
                surface_moment += panel_moment
            
            # Store surface results
            surface_forces_and_moments[surface_key] = {
                'force': surface_force,
                'moment': surface_moment
            }
            
            # Add to aircraft totals
            total_force += surface_force
            total_moment += surface_moment
            
            # print(f"{surface_key} - Force: {surface_force}, Moment: {surface_moment}")
        
        # Store total results
        surface_forces_and_moments['Total_Aircraft'] = {
            'force': total_force,
            'moment': total_moment
        }
        
        print(f"Total Aircraft - Force: {total_force}, Moment: {total_moment}")
        
        return surface_forces_and_moments

        """
        Plot gamma distribution across all wing surfaces
        for a fixed chordwise position while varying spanwise index.
        
        Args:
            aircraft_mesh: Complete aircraft mesh data
            fixed_chordwise_index: Fixed chordwise position (cp_index[1])
        """
        # Filter surfaces to plot (exclude vertical tail for spanwise analysis)
        wing_surfaces = ['Main_Wing', 'Horizontal_Tail']
        available_surfaces = [key for key in wing_surfaces if key in aircraft_mesh]
        
        n_surfaces = len(available_surfaces)
        n_cols = min(2, n_surfaces)
        n_rows = (n_surfaces + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 12))
        if n_surfaces == 1:
            axs = [axs]
        elif n_rows == 1:
            axs = axs if n_surfaces > 1 else [axs]
        else:
            axs = axs.flatten()
        
        fig.suptitle(f'Gamma Distribution Across Wing Surfaces (Fixed Chordwise Index: {fixed_chordwise_index})')
        
        for surf_idx, surface_key in enumerate(available_surfaces):
            if surf_idx < len(axs):
                ax = axs[surf_idx]
                surface_data = aircraft_mesh[surface_key]
                
                # Extract gamma values and spanwise positions
                gamma_values = []
                spanwise_positions = []
                
                # Get values for fixed chordwise position
                for cp_index, control_point in surface_data['Control Points'].items():
                    if cp_index[1] == fixed_chordwise_index:  # Fixed chordwise position
                        if 'Gamma' in surface_data:
                            gamma = surface_data['Gamma'][cp_index]
                        else:
                            gamma = 0.0  # Default if no gamma calculated yet
                        
                        gamma_values.append(gamma)
                        spanwise_positions.append(cp_index[0])  # Spanwise index
                
                # Sort by spanwise position
                if spanwise_positions:
                    sorted_indices = np.argsort(spanwise_positions)
                    spanwise_positions = np.array(spanwise_positions)[sorted_indices]
                    gamma_values = np.array(gamma_values)[sorted_indices]
                    
                    # Plot for this surface
                    ax.plot(spanwise_positions, gamma_values, 'bo-',
                        linewidth=2, markersize=8, label=surface_key)
                
                ax.set_title(f'{surface_key}')
                ax.set_xlabel('Spanwise Position (cp_index[0])')
                ax.set_ylabel('Gamma')
                ax.grid(True)
                ax.legend()
        
        # Hide unused subplots
        for i in range(n_surfaces, len(axs)):
            axs[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()


        """
        Create slice-based visualization similar to your PyVista velocity slices
        
        Args:
            aircraft_mesh: Complete aircraft mesh data
            n_slices: Number of slices to create
            slice_direction: 'x', 'y', or 'z' for slice direction
        """
        
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Collect all coordinates to determine slice bounds
        all_coords = []
        for surface_key, surface_data in aircraft_mesh.items():
            for panel in surface_data['Panels'].values():
                all_coords.extend(panel)
        
        all_coords = np.array(all_coords)
        
        # Determine slice bounds based on direction
        if slice_direction == 'x':
            slice_min, slice_max = all_coords[:, 0].min(), all_coords[:, 0].max()
            coord_idx = 0
        elif slice_direction == 'y':
            slice_min, slice_max = all_coords[:, 1].min(), all_coords[:, 1].max()
            coord_idx = 1
        else:  # z
            slice_min, slice_max = all_coords[:, 2].min(), all_coords[:, 2].max()
            coord_idx = 2
        
        # Create slice positions
        slice_positions = np.linspace(slice_min, slice_max, n_slices)
        tolerance = (slice_max - slice_min) / (n_slices * 2)  # Slice thickness
        
        # Collect all pressure values for normalization
        all_pressures = []
        for surface_key, surface_data in aircraft_mesh.items():
            if 'Pressure_Difference' in surface_data:
                all_pressures.extend(surface_data['Pressure_Difference'].values())
        
        max_abs_pressure = max(abs(min(all_pressures)), abs(max(all_pressures))) if all_pressures else 1.0
        
        # Process each slice
        for slice_pos in slice_positions:
            for surface_key, surface_data in aircraft_mesh.items():
                if 'Pressure_Difference' not in surface_data:
                    continue
                    
                for panel_index, panel in surface_data['Panels'].items():
                    panel_array = np.array(panel)
                    
                    # Check if panel intersects this slice
                    panel_coords = panel_array[:, coord_idx]
                    if (panel_coords.min() <= slice_pos + tolerance and 
                        panel_coords.max() >= slice_pos - tolerance):
                        
                        # Get pressure and normalize
                        pressure = surface_data['Pressure_Difference'][panel_index]
                        normalized_pressure = pressure / max_abs_pressure if max_abs_pressure != 0 else 0
                        
                        # Create surface with transparency
                        try:
                            X = np.array([[panel_array[0, 0], panel_array[1, 0]], 
                                        [panel_array[3, 0], panel_array[2, 0]]])
                            Y = np.array([[panel_array[0, 1], panel_array[1, 1]], 
                                        [panel_array[3, 1], panel_array[2, 1]]])
                            Z = np.array([[panel_array[0, 2], panel_array[1, 2]], 
                                        [panel_array[3, 2], panel_array[2, 2]]])
                            
                            color_value = 0.5 + 0.4 * normalized_pressure
                            color = plt.cm.plasma(color_value)
                            
                            ax.plot_surface(X, Y, Z, 
                                        color=color, 
                                        alpha=0.3,  # High transparency for slice effect
                                        edgecolor='none',
                                        shade=False,
                                        linewidth=0)
                        except:
                            continue
        
        # Styling similar to PyVista
        ax.view_init(elev=30, azim=45)
        
        # Colorbar
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        norm = Normalize(vmin=-1, vmax=1)
        sm = ScalarMappable(norm=norm, cmap='plasma')
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Pressure Difference\n', size=20, fontfamily='arial')
        cbar.ax.tick_params(labelsize=16)
        
        # Set bounds
        max_range = np.array([
            all_coords[:, 0].max() - all_coords[:, 0].min(),
            all_coords[:, 1].max() - all_coords[:, 1].min(),
            all_coords[:, 2].max() - all_coords[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
        mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
        mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Clean axes
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        plt.tight_layout()
        plt.show()