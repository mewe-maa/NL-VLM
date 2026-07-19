import numpy as np
from scipy.interpolate import griddata

from nl_vlm.reporting import SolverReporter


class VLM:
    def __init__(self, vehicle_mesh, propeller_mesh, verbose=True,
                 nonlinear_lift=None, profile_drag=None):
        """Initialize the VLM solver with the given vehicle/propeller mesh.

        Parameters
        ----------
        verbose : bool, optional
            If True (default), report analysis progress to stdout via a
            :class:`~nl_vlm.reporting.SolverReporter` (startup banner, a live
            convergence bar, and a result summary). ``verbose=False`` attaches a
            disabled reporter, so the solver runs silently.
        nonlinear_lift, profile_drag : bool, optional
            Split of the viscous coupling. ``nonlinear_lift`` converges gamma
            against the 2D-polar Cl; ``profile_drag`` adds the sectional
            0.5*rho*V^2*c*ds*Cd term. Both default to whether the blade carries a
            2D polar; both require one.
        """

        self.vehicle_mesh = vehicle_mesh
        self.propeller_mesh = propeller_mesh

        has_polars = getattr(getattr(propeller_mesh, 'geometry', None), 'has_polars', True)
        for name, value in (('nonlinear_lift', nonlinear_lift), ('profile_drag', profile_drag)):
            if value and not has_polars:
                raise ValueError(
                    f"{name}=True needs a 2D polar, but this blade has none. Construct "
                    f"PropGeom with polar_path=... , or pass {name}=False for an "
                    f"inviscid run."
                )
        self.nonlinear_lift = has_polars if nonlinear_lift is None else nonlinear_lift
        self.profile_drag = has_polars if profile_drag is None else profile_drag

        self.reporter = SolverReporter(enabled=verbose)

    @staticmethod
    def _biot_savart_filament(cp, v_start, v_end):

        """Vectorized Biot-Savart for one straight filament (unit circulation).

        induced = (1 / 4*pi) * (r1 x r2) * [r0 . (r1/|r1| - r2/|r2|)] / denom

        Potential form by default. To enable a viscous core (Vatistas, n=2) that
        regularizes the near-filament singularity, comment out the potential
        `denominator` line and uncomment the two core lines below it.

        cp, v_start, v_end broadcast to a common (..., 3) shape; the return has
        that same shape.
        """
        r1 = cp - v_end
        r2 = cp - v_start
        r0 = r1 - r2
        cross = np.cross(r1, r2)
        norm_cross_sq = np.einsum('...k,...k->...', cross, cross)
        n1 = np.sqrt(np.einsum('...k,...k->...', r1, r1))
        n2 = np.sqrt(np.einsum('...k,...k->...', r2, r2))
        dot_term = np.einsum('...k,...k->...', r0, r1 / n1[..., None] - r2 / n2[..., None])

        # Core radius and exponent for the optional viscous-core denominator
        n = 2.0
        rc = 2.94e-5

        denominator = norm_cross_sq
        # Viscous core
        # norm_r0_sq = np.einsum('...k,...k->...', r0, r0)
        # denominator = (norm_cross_sq**(n) + (rc**2 * norm_r0_sq)**(n))**(1.0 / n)

        with np.errstate(divide='ignore', invalid='ignore'):
            coeff = (1.0 / (4.0 * np.pi)) * dot_term / denominator
        return cross * coeff[..., None]

    def calculate_bound_to_bound_induced_velocity_matrix(self, quad_propeller_mesh, omega_dict, freestream):

        """Global induced-velocity (AIC) matrix for all control points and vortex rings.

        Builds the full (N, N, 3) influence matrix per propeller with broadcasted
        Biot-Savart, so the O(N^2 * 4) filament evaluations run as a handful of
        array operations instead of a Python double loop.
        """
        INFINITY_FACTOR = 1000.0
        global_matrices = {}

        for propeller_key, propeller_data in quad_propeller_mesh.items():

            # Collect per-panel data in the SAME order the looped version iterates,
            # so AIC[i, j] keeps the same meaning (control point i, vortex ring j).
            
            control_points = []
            ring_vertices = []
            spanwise_indices = []
            tangent_j = []
            for blade_key, blade_data in propeller_data['Blades'].items():
                tangential_vectors = blade_data['Tangential Vectors']
                for cp_index, control_point in blade_data['Control Points'].items():
                    control_points.append(np.asarray(control_point, dtype=float))
                    ring_vertices.append(np.asarray(blade_data['Vortex Rings'][cp_index]['Vertices'], dtype=float))
                    spanwise_indices.append(cp_index[0])
                    tangent_j.append(np.asarray(tangential_vectors[cp_index]['Tangential j'], dtype=float))

            CP = np.array(control_points)        # (N, 3)
            VR = np.array(ring_vertices)         # (N, 4, 3)
            spanwise_indices = np.array(spanwise_indices)
            TAN_J = np.array(tangent_j)          # (N, 3)
            N = CP.shape[0]

            cp_i = CP[:, None, :]                # (N, 1, 3): control point i vs ring j

            AIC = np.zeros((N, N, 3))

            # Bound vortex ring: 4 finite filaments.
            for k in range(4):
                v_start = VR[:, k, :][None, :, :]            # (1, N, 3) over rings j
                v_end = VR[:, (k + 1) % 4, :][None, :, :]
                AIC += self._biot_savart_filament(cp_i, v_start, v_end)

            # Trailing edge: each TE ring sheds a semi-infinite horseshoe (closed at
            # infinity along the blade-tangent direction). Add only to TE columns.
            max_spanwise = spanwise_indices.max()
            te = np.where(spanwise_indices == max_spanwise)[0]
            if te.size:
                td = TAN_J[te] / np.linalg.norm(TAN_J[te], axis=1, keepdims=True)   # (M, 3)
                v1 = VR[te, 1, :]                # (M, 3)
                v2 = VR[te, 2, :]
                new_vertices = np.stack([
                    v1,
                    v1 + INFINITY_FACTOR * td,
                    v2 + INFINITY_FACTOR * td,
                    v2,
                ], axis=1)                       # (M, 4, 3)

                te_contrib = np.zeros((N, te.size, 3))
                for k in range(4):
                    v_start = new_vertices[:, k, :][None, :, :]
                    v_end = new_vertices[:, (k + 1) % 4, :][None, :, :]
                    te_contrib += self._biot_savart_filament(cp_i, v_start, v_end)
                AIC[:, te, :] += te_contrib

            global_matrices[propeller_key] = AIC

        return global_matrices

    def calculate_gamma(self, quad_propeller_mesh, bound_to_bound_global_matrices,
               wake_to_bound_induced_velocity_matrices, omega_dict, 
               body_velocity, freestream, wind_field, com_position, time_step, euler_angles):
    
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            
            effective_omega = omega_dict[propeller_key]
            omega_sign = np.sign(effective_omega[2])
            hub_position = np.array(propeller_data['Hub Position'])

            # Collect control points and build RHS
            control_points = []
            normals = []
            rhs = []

            for blade_key, blade_data in propeller_data['Blades'].items():
                for cp_index, control_point in blade_data['Control Points'].items():
                    control_point = np.array(control_point)
                    normal = blade_data['Normals'][cp_index]
                    normals.append(normal)
                    
                    if wind_field is None:
                        wind_velocity = np.array([0, 0, 0])
                    elif getattr(self, 'wind_mode', 'distributed') == 'jacobian':
                        wind_velocity = -wind_field.get_jacobian_approximated_velocity(control_point, com_position)
                    else:  # distributed (samples the urban wind at each blade control point)
                        wind_velocity = -wind_field.get_wind_velocity(control_point)

                    radius_vector = control_point - hub_position
                    omega_cross_r = np.cross(effective_omega, radius_vector)

                    if "Omega_Cross_R" not in blade_data:
                        blade_data['Omega_Cross_R'] = {}
                    blade_data['Omega_Cross_R'][cp_index] = omega_cross_r

                    if "Wind Velocity" not in blade_data:
                        blade_data['Wind Velocity'] = {}
                    blade_data['Wind Velocity'][cp_index] = wind_velocity

                    velocity_term = freestream -omega_cross_r + wind_velocity
                    rhs_value = -np.dot(velocity_term, normal)
                    rhs.append(rhs_value)
                    control_points.append(control_point)

            normals = np.array(normals)
            rhs = np.array(rhs).reshape(-1, 1)

            # Build AIC matrix and initial solve
            bound_to_bound_induced_matrix = bound_to_bound_global_matrices[propeller_key]
            Ab = np.einsum('ijk,ik->ij', bound_to_bound_induced_matrix, normals)
            gamma = np.linalg.solve(Ab, rhs)

            # Get max chordwise index
            max_chordwise = max(idx[0] for idx in
                list(propeller_data['Blades'].values())[0]['Control Points'].keys())

            # Viscous coupling: converge gamma against the 2D-polar Cl
            gamma, strip_data = self._nonlinear_lift_coupling(
                propeller_data, Ab, bound_to_bound_induced_matrix, rhs, gamma,
                omega_sign, freestream, max_chordwise, label=propeller_key)

            # Final induced velocities (used in the per-panel post-processing below)
            induced_vel = np.einsum('ijk,j->ik', bound_to_bound_induced_matrix, gamma.flatten())

            # Post-process: store gamma, induced velocities, and alpha_eff per panel
            gamma_index = 0
            for blade_key, blade_data in propeller_data['Blades'].items():
                blade_data['Gamma'] = {}
                blade_data['Alpha_Eff'] = {}
                blade_data['Induced_Velocities'] = {}

                for cp_index in blade_data['Control Points'].keys():
                    blade_data['Gamma'][cp_index] = float(gamma[gamma_index, 0])
                    blade_data['Induced_Velocities'][cp_index] = induced_vel[gamma_index]

                    omega_cross_r = blade_data['Omega_Cross_R'][cp_index]
                    wind = blade_data['Wind Velocity'][cp_index]

                    induced_velocity = induced_vel[gamma_index]
                    V_local = freestream - omega_cross_r + induced_velocity + wind

                    omega_cross_r_mag = np.linalg.norm(omega_cross_r)
                    if omega_cross_r_mag > 1e-10:
                        tangential_dir = omega_cross_r / omega_cross_r_mag
                    else:
                        tangential_dir = np.array([1, 0, 0])

                    V_tangential = np.dot(V_local, tangential_dir)
                    V_axial = V_local[2]
                    twist_angle = np.radians(blade_data['Twist'][cp_index])
                    phi = np.arctan2(-V_axial, abs(V_tangential))
                    alpha_eff = twist_angle - phi
                    blade_data['Alpha_Eff'][cp_index] = alpha_eff

                    gamma_index += 1

            for blade_key, blade_data in propeller_data['Blades'].items():
                blade_data['Strip_Data'] = {}
                for strip_key, sdata in strip_data.items():
                    if strip_key[0] == blade_key:
                        blade_data['Strip_Data'][strip_key] = sdata
                        

    def _nonlinear_lift_coupling(self, propeller_data, Ab, bound_to_bound_induced_matrix, rhs, gamma,
                          omega_sign, freestream, max_chordwise, label=""):

        """Fixed-point couple the VLM circulation to the 2D-polar Cl.

        Each iteration: build per-strip flow state from the current gamma, compare
        the VLM sectional Cl to the airfoil-polar Cl, relax the RHS by that residual
        and re-solve, until gamma converges. Returns (converged gamma, strip_data).
        """
        max_iterations = 2000 if self.nonlinear_lift else 1
        tolerance = 1e-6
        relaxation = 0.2

        if self.nonlinear_lift:
            self.reporter.start_solve(tolerance)
        converged = False

        strip_data = {}
        for iteration in range(max_iterations):
            gamma_old_iter = gamma.copy()
            induced_vel = np.einsum('ijk,j->ik', bound_to_bound_induced_matrix, gamma.flatten())

            # Pass 1: collect indices, induced velocities, and geometry per strip
            strip_data = {}
            gamma_index = 0
            for blade_key, blade_data in propeller_data['Blades'].items():
                for cp_index in blade_data['Control Points'].keys():
                    spanwise_j = cp_index[1]
                    strip_key = (blade_key, spanwise_j)

                    if strip_key not in strip_data:
                        strip_data[strip_key] = {
                            'gamma_indices': [],
                            'induced_vel_sum': np.zeros(3),
                            'omega_cross_r_sum': np.zeros(3),
                            'wind_sum': np.zeros(3),
                            'twist_angle': 0,
                            'chord': 0,
                            'r_R': 0,
                        }

                    strip_data[strip_key]['gamma_indices'].append(gamma_index)
                    strip_data[strip_key]['induced_vel_sum'] += induced_vel[gamma_index]
                    strip_data[strip_key]['omega_cross_r_sum'] += blade_data['Omega_Cross_R'][cp_index]
                    strip_data[strip_key]['wind_sum'] += blade_data['Wind Velocity'][cp_index]

                    if cp_index[0] == max_chordwise:
                        strip_data[strip_key]['gamma_te'] = float(gamma[gamma_index, 0])
                        strip_data[strip_key]['twist_angle'] = np.radians(blade_data['Twist'][cp_index])
                        strip_data[strip_key]['r_R'] = blade_data['r_R'][cp_index]
                        strip_data[strip_key]['chord'] = blade_data['Chord'][cp_index]

                    gamma_index += 1

            # Pass 2: compute flow conditions using strip-averaged induced velocity
            for strip_key, sdata in strip_data.items():
                avg_induced = sdata['induced_vel_sum'] / (max_chordwise + 1)
                omega_cross_r = sdata['omega_cross_r_sum'] / (max_chordwise + 1)
                wind = sdata['wind_sum'] / (max_chordwise + 1)

                V_local = freestream - omega_cross_r + avg_induced + wind

                V_mag = np.linalg.norm(V_local)
                omega_cross_r_mag = np.linalg.norm(omega_cross_r)
                tangential_dir = omega_cross_r / omega_cross_r_mag
                V_tangential = np.dot(V_local, tangential_dir)
                V_axial = V_local[2]
                phi = np.arctan2(-V_axial, abs(V_tangential))
                alpha_eff = sdata['twist_angle'] - phi

                sdata['V_mag'] = V_mag
                sdata['V_local'] = V_local
                sdata['alpha_eff'] = alpha_eff

                if self.nonlinear_lift:
                    sdata['cl_table'] = self.propeller_mesh.geometry.get_cl(
                        sdata['r_R'], np.degrees(alpha_eff))

            # Without the lift coupling there is nothing to feed back; keep the
            # inviscid gamma and stop after building the strip state.
            if not self.nonlinear_lift:
                break

            # Pass 3: compute residual and update RHS
            residual = np.zeros_like(rhs)
            for strip_key, sdata in strip_data.items():
                cl_vlm = omega_sign * 2.0 * sdata['gamma_te'] / (sdata['V_mag'] * sdata['chord'])

                r_cl = cl_vlm - sdata['cl_table']

                for idx in sdata['gamma_indices']:
                    residual[idx] = omega_sign * r_cl

            rhs = rhs + relaxation * residual
            gamma = np.linalg.solve(Ab, rhs)

            delta = np.max(np.abs(gamma - gamma_old_iter))
            self.reporter.progress(label, iteration, delta)

            if delta < tolerance:
                converged = True
                break

        if self.nonlinear_lift:
            self.reporter.finish_solve(label, iteration, delta, converged, max_iterations)

        return gamma, strip_data

    def pressure_difference(self, quad_propeller_mesh, bound_to_bound_global_matrices, wake_to_bound_induced_velocity_matrices, body_velocity, freestream, omega, time_step, dt, rho):
        
        """
        Calculate the pressure difference for each panel.
        
        """
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            
            for blade_key, blade_data in propeller_data['Blades'].items():
                control_points = blade_data['Control Points']
                normals = blade_data['Normals']
                pressure_difference = {}
                force = {}


                # Strip-averaged induced velocity per spanwise station. Using a
                # chord-CONSTANT velocity in the Kutta-Joukowski force makes the
                # chordwise circulation differences telescope to the strip's total
                # circulation, so the force is grid-convergent in chord. The
                # per-panel bound-induced velocity varies chordwise (near-field
                # self-induction) and makes the force drift with chord resolution.
                induced_sum, induced_count = {}, {}
                for cp_index in control_points:
                    j = cp_index[1]
                    induced_sum[j] = induced_sum.get(j, np.zeros(3)) + blade_data['Induced_Velocities'][cp_index]
                    induced_count[j] = induced_count.get(j, 0) + 1
                strip_induced = {j: induced_sum[j] / induced_count[j] for j in induced_sum}

                for panel_index, control_point in control_points.items():
                    normal = normals[panel_index]

                    # Local flow velocity at the panel (induced part strip-averaged)
                    omega_cross_r = blade_data['Omega_Cross_R'][panel_index]
                    wind_velocity = blade_data['Wind Velocity'][panel_index]
                    total_velocity = freestream - omega_cross_r + strip_induced[panel_index[1]] + wind_velocity

                    # Panel area (sum of the two corner triangles)
                    panel_array = np.array(blade_data['Panels'][panel_index])
                    area_triangle_1 = 0.5 * np.linalg.norm(np.cross(panel_array[1] - panel_array[0], panel_array[3] - panel_array[0]))
                    area_triangle_2 = 0.5 * np.linalg.norm(np.cross(panel_array[2] - panel_array[1], panel_array[3] - panel_array[1]))
                    panel_area = area_triangle_1 + area_triangle_2

                    # Kutta-Joukowski strip force from the spanwise circulation gradient
                    gamma_current = blade_data['Gamma'][panel_index]
                    gamma_previous_span = blade_data['Gamma'].get((panel_index[0] - 1, panel_index[1]), 0) if panel_index[0] > 0 else 0
                    delta_gamma = gamma_current - gamma_previous_span

                    vr = blade_data['Vortex Rings'][panel_index]
                    dl_span = np.array(vr['Vertices'][3]) - np.array(vr['Vertices'][0])
                    strip_force = rho * np.cross(total_velocity, delta_gamma * dl_span)
                    pressure = np.dot(strip_force, normal) / panel_area

                    pressure_difference[panel_index] = pressure
                    force[panel_index] = strip_force


                blade_data['Pressure Difference'] = pressure_difference
                blade_data['force'] = force


    def calculate_total_forces_and_moments(self, propeller_mesh, dt, time_step, rho, body_velocity, freestream, omega, wind_field, com_position, euler_angles=None):
       
        """
        Calculate aerodynamic forces and moments for each panel of each propeller
        using VLM. This includes updating the pressure difference for each panel.

        """
        self.reporter.announce(
            propeller_mesh, omega, rho, freestream,
            span_res=getattr(self.propeller_mesh, 'span_resolution', None),
            chord_res=getattr(self.propeller_mesh, 'chord_resolution', None),
        )

        # Compute the global induced velocity matrix
        bound_to_bound_global_matrices = self.calculate_bound_to_bound_induced_velocity_matrix(propeller_mesh, omega, freestream)
        wake_to_bound_induced_velocity_matrices = None
        
        # Calculate gamma for each panel
        self.calculate_gamma(
            propeller_mesh,
            bound_to_bound_global_matrices,
            wake_to_bound_induced_velocity_matrices,
            omega,
            body_velocity,
            freestream,
            wind_field, 
            com_position, 
            time_step,
            euler_angles=euler_angles,
        )

        # Update the pressure differences for all panels
        self.pressure_difference(
            propeller_mesh,
            bound_to_bound_global_matrices,
            wake_to_bound_induced_velocity_matrices,
            body_velocity,
            freestream,
            omega,
            time_step,
            dt,
            rho
        )
        total_forces_and_moments = {}
        # Calculate forces and moments for each panel
        for propeller_key, propeller_data in propeller_mesh.items():
            hub_position = np.array(propeller_data['Hub Position'])
            total_force = np.zeros(3)
            total_moment = np.zeros(3)
            total_body_moment = np.zeros(3)
            for blade_key, blade_data in propeller_data['Blades'].items():
                blade_data['Panel Forces'] = {}
                blade_data['Panel Moments'] = {}

                for panel_index, pressure in blade_data['Pressure Difference'].items():
                    panel = blade_data['Panels'][panel_index]
                    panel_array = np.array(panel)
                    control_point = blade_data['Control Points'][panel_index]
                    
                    # Calculate the normal vector and panel area
                    normal = blade_data['Normals'][panel_index]
                    
                    area_triangle_1 = 0.5 * np.linalg.norm(np.cross(panel_array[1] - panel_array[0], panel_array[3] - panel_array[0]))
                    area_triangle_2 = 0.5 * np.linalg.norm(np.cross(panel_array[2] - panel_array[1], panel_array[3] - panel_array[1]))

                    panel_area = area_triangle_1 + area_triangle_2
                    panel_force = (pressure * panel_area) * normal
                    moment_arm_local = control_point - hub_position
                    moment_arm_global = control_point - com_position
                    
                    panel_moment = np.cross(moment_arm_local, panel_force)
                    body_moment =np.cross(moment_arm_global, panel_force)
                    
                    blade_data['Panel Forces'][panel_index] = panel_force
                    blade_data['Panel Moments'][panel_index] = panel_moment

                    # Accumulate to total force and moment
                    total_force += panel_force
                    total_moment += panel_moment
                    total_body_moment+= body_moment

            # ---- Profile (viscous) drag per spanwise strip, from the 2D-polar Cd ----
            # Kutta-Joukowski above gives lift + induced drag only. Add the sectional
            # profile drag 0.5*rho*V^2*chord*ds*Cd along the local relative-wind direction.
            if self.profile_drag:
                for blade_key, blade_data in propeller_data['Blades'].items():
                    for strip_key, sdata in blade_data.get('Strip_Data', {}).items():
                        spanwise_j = strip_key[1]
                        strip_panels = [idx for idx in blade_data['Control Points'] if idx[1] == spanwise_j]
                        if not strip_panels:
                            continue
                        vr = blade_data['Vortex Rings'][strip_panels[0]]
                        ds = np.linalg.norm(np.array(vr['Vertices'][3]) - np.array(vr['Vertices'][0]))
                        pos = np.mean([np.array(blade_data['Control Points'][idx]) for idx in strip_panels], axis=0)
                        V_mag = sdata['V_mag']
                        if V_mag < 1e-9:
                            continue
                        cd = self.propeller_mesh.geometry.get_cd(sdata['r_R'], np.degrees(sdata['alpha_eff']))
                        drag_mag = 0.5 * rho * V_mag ** 2 * sdata['chord'] * ds * cd
                        drag_force = drag_mag * (sdata['V_local'] / V_mag)
                        total_force += drag_force
                        total_moment += np.cross(pos - hub_position, drag_force)
                        total_body_moment += np.cross(pos - com_position, drag_force)

            total_forces_and_moments[propeller_key] = {'force': total_force, 'moment': total_moment, 'total_body_moment': total_body_moment}

        self.reporter.summary(total_forces_and_moments)

        return  total_forces_and_moments
    
    
