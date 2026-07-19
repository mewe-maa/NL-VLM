import numpy as np
from scipy.interpolate import interp1d as _interp1d
from nl_vlm.environment.wind import WindField
# NOTE: plotting lives in the notebook (fixed_wing.ipynb), not in the solver.


class WingVLM:
    # --- solver options ---
    include_viscous_coupling = False   # match each strip to its 2D polar (-> alpha_L0 -4.2, stall) AND add 2D-polar profile drag
    couple_wind = False                # sample the wind field per control point (default OFF -> still air)
    visc_relax = 0.10                  # RHS relaxation for the viscous fixed-point
    visc_maxit = 4000
    visc_tol = 1e-5

    def __init__(self, aircraft_mesh, geometry=None,
                 include_viscous_coupling=None,
                 couple_wind=None, visc_relax=None, visc_maxit=None, visc_tol=None):
        """Wing VLM solver.

        `geometry` is the SimpleAircraftGeometry that built the mesh; the solver reads its per-surface
        chord and 2D airfoil polars from there (mirroring how the rotor VLM reads PropGeom.get_cl), so
        NO aircraft-specific data is hardcoded in the solver -- configure the airfoils / chords / polar
        folder in your notebook (fixed_wing.ipynb) on the geometry. geometry may be omitted for a pure
        inviscid run (no polar / chord needed). include_viscous_coupling turns on the whole viscous
        treatment: polar Cl-matching AND the 2D-polar profile drag (one switch, not two). The option
        kwargs, when given, override the class defaults for this instance.
        """
        self.aircraft_mesh = aircraft_mesh
        self.geometry = geometry
        for _name, _value in (('include_viscous_coupling', include_viscous_coupling),
                              ('couple_wind', couple_wind),
                              ('visc_relax', visc_relax), ('visc_maxit', visc_maxit),
                              ('visc_tol', visc_tol)):
            if _value is not None:
                setattr(self, _name, _value)
        self._cd_cache = {}   # surface_key -> Cd(Cl) interpolator (built on first use, from geometry polar)

    # --- per-surface 2D polar lookups, read from the geometry ---
    def _cl_of_alpha(self, surface_key):
        """Cl(alpha_deg) interpolator for a surface, from the geometry's 2D polar."""
        return self.geometry.cl_interpolator(surface_key)

    def _cd_of_cl(self, surface_key):
        """Cd(Cl) interpolator (attached branch) for a surface, from the geometry's 2D polar."""
        key = surface_key.lower()
        if key not in self._cd_cache:
            p = self.geometry.polar_dataframe(surface_key)
            att = p[(p.Alpha > -8) & (p.Alpha < 12)]        # attached Cd-vs-Cl branch
            self._cd_cache[key] = _interp1d(att.Cl, att.Cd, bounds_error=False,
                                            fill_value=(att.Cd.min(), att.Cd.iloc[-1]))
        return self._cd_cache[key]
    
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
    
    def calculate_bound_to_bound_induced_velocity_matrix(self, aircraft_mesh, alpha_deg=0.0):

        """Global induced-velocity (AIC) matrix for all control points and vortex rings.

        Builds the full (N, N, 3) influence matrix per surface with broadcasted Biot-Savart.
        
        """

        INFINITY_FACTOR = 100.0
        alpha_rad = np.radians(alpha_deg)
        flow_direction = np.array([np.cos(alpha_rad), 0.0, np.sin(alpha_rad)])   # trailing-wake direction

        global_matrices = {}
        for surface_key, surface_data in aircraft_mesh.items():

            # Collect per-panel data in dict order, so AIC[i, j] (control point i, vortex ring j).
            control_points = []
            ring_vertices = []
            spanwise_indices = []
            for cp_index, control_point in surface_data['Control Points'].items():
                control_points.append(np.asarray(control_point, dtype=float))
                ring_vertices.append(np.asarray(surface_data['Vortex Rings'][cp_index]['Vertices'], dtype=float))
                spanwise_indices.append(cp_index[0])

            CP = np.array(control_points)        # (N, 3)
            VR = np.array(ring_vertices)         # (N, 4, 3)
            spanwise_indices = np.array(spanwise_indices)
            N = CP.shape[0]

            cp_i = CP[:, None, :]                # (N, 1, 3): control point i vs ring j
            AIC = np.zeros((N, N, 3))

            # Bound vortex ring: 4 finite filaments.
            for k in range(4):
                v_start = VR[:, k, :][None, :, :]            # (1, N, 3) over rings j
                v_end = VR[:, (k + 1) % 4, :][None, :, :]
                AIC += self._biot_savart_filament(cp_i, v_start, v_end)

            # Trailing edge: each TE ring sheds a semi-infinite horseshoe closed at infinity along
            # the flow direction. Add only to the TE columns.
            max_spanwise = spanwise_indices.max()
            te = np.where(spanwise_indices == max_spanwise)[0]
            if te.size:
                v1 = VR[te, 1, :]                # (M, 3)
                v2 = VR[te, 2, :]
                new_vertices = np.stack([
                    v1,
                    v1 + INFINITY_FACTOR * flow_direction,
                    v2 + INFINITY_FACTOR * flow_direction,
                    v2,
                ], axis=1)                       # (M, 4, 3)

                te_contrib = np.zeros((N, te.size, 3))
                for k in range(4):
                    v_start = new_vertices[:, k, :][None, :, :]
                    v_end = new_vertices[:, (k + 1) % 4, :][None, :, :]
                    te_contrib += self._biot_savart_filament(cp_i, v_start, v_end)
                AIC[:, te, :] += te_contrib

            global_matrices[surface_key] = AIC

        return global_matrices
    
    def calculate_gamma(self, aircraft_mesh, bound_to_bound_global_matrices, 
                       alpha_deg, airspeed, wind_field, reference_point):
        """Solve each surface's circulation from the Neumann (flow-tangency) boundary condition.

        Per control point:  (-freestream + wind) . normal + (AIC @ gamma) . normal = 0.
        Wind is sampled per control point (distributed) -- pass
        wind_field=None for still air, or set self.wind_mode='jacobian' for the linearized field.
        """
        gamma_matrices = {}
        induced_velocities = {}
        freestream_velocity = np.asarray(airspeed, dtype=float)

        for surface_key, surface_data in aircraft_mesh.items():

            # Collect normals and build the RHS (onset flow through each panel)
            normals = []
            rhs = []
            for cp_index, control_point in surface_data['Control Points'].items():
                control_point = np.array(control_point)
                normal = surface_data['Normals'][cp_index]
                normals.append(normal)

                if wind_field is None or not self.couple_wind:
                    wind_velocity = np.zeros(3)                        # still air (default)
                elif getattr(self, 'wind_mode', 'distributed') == 'jacobian':
                    wind_velocity = wind_field.get_jacobian_approximated_velocity(
                        control_point + reference_point, reference_point)
                else:  # distributed: sample the urban wind at each control point
                    wind_velocity = wind_field.get_wind_velocity(control_point + reference_point)

                surface_data.setdefault('Freestream_Velocity', {})[cp_index] = freestream_velocity
                surface_data.setdefault('Wind Velocity', {})[cp_index] = wind_velocity

                velocity_term = -freestream_velocity + wind_velocity
                rhs.append(-np.dot(velocity_term, normal))

            normals = np.array(normals)
            rhs = np.array(rhs).reshape(-1, 1)

            # Flow-tangency solve  A gamma = rhs, then the wake-induced velocity at each panel
            bound_to_bound_induced_matrix = bound_to_bound_global_matrices[surface_key]
            A_matrix = np.einsum('ijk,ik->ij', bound_to_bound_induced_matrix, normals)
            gamma = np.linalg.solve(A_matrix, rhs)
            induced_vel = np.einsum('ijk,j->ik', bound_to_bound_induced_matrix, gamma.flatten())

            gamma_matrices[surface_key] = gamma.flatten()
            induced_velocities[surface_key] = induced_vel

            # Store gamma + induced velocity per panel
            surface_data.setdefault('Gamma', {})
            surface_data.setdefault('Induced_Velocities', {})
            for gamma_index, cp_index in enumerate(surface_data['Control Points'].keys()):
                surface_data['Gamma'][cp_index] = float(gamma[gamma_index])
                surface_data['Induced_Velocities'][cp_index] = induced_vel[gamma_index]

        return gamma_matrices, induced_velocities
    
    def calculate_pressure_difference(self, aircraft_mesh, alpha_deg, airspeed, rho, wind_field, reference_point, cog=None):
        """
        Calculate the per-panel pressure difference AND the surface forces/moments (Kutta-Joukowski).

        Per panel: the signed KJ load / area gives Pressure_Difference (used by the plots). Per spanwise
        strip: the near-field KJ force with the freestream gives LIFT / side force / moment, and the
        far-field (Trefftz) integral against the wake downwash gives the INDUCED DRAG.

        Returns:
            dict: {'force', 'moment', 'induced_drag'} per surface and under 'Total_Aircraft'.
        """
        freestream = np.asarray(airspeed, float)
        cog = np.zeros(3) if cog is None else cog
        results = {}
        total_force, total_moment, total_drag, total_profile = np.zeros(3), np.zeros(3), 0.0, 0.0

        for surface_key, surface_data in aircraft_mesh.items():

            if 'Pressure_Difference' not in surface_data:
                surface_data['Pressure_Difference'] = {}
            tangential_vectors = surface_data['Tangential Vectors']
            for panel_index, control_point in surface_data['Control Points'].items():
                tangent_span = tangential_vectors[panel_index]['Tangential i']
                normal = surface_data['Normals'][panel_index]
                total_velocity = -surface_data['Freestream_Velocity'][panel_index] + surface_data['Wind Velocity'][panel_index]
                induced_velocity = surface_data['Induced_Velocities'][panel_index]
                gamma_current = surface_data['Gamma'][panel_index]
                gamma_previous_span = surface_data['Gamma'].get((panel_index[0] - 1, panel_index[1]), 0) if panel_index[0] > 0 else 0

                panel_array = np.array(surface_data['Panels'][panel_index])
                panel_area = (0.5 * np.linalg.norm(np.cross(panel_array[1] - panel_array[0], panel_array[3] - panel_array[0])) +
                              0.5 * np.linalg.norm(np.cross(panel_array[2] - panel_array[1], panel_array[3] - panel_array[1])))

                # full KJ force vector: (total_velocity + induced_velocity) already includes the downwash
                _kj = rho * (gamma_current - gamma_previous_span) * np.cross(total_velocity + induced_velocity, tangent_span)
                surface_data.setdefault('KJ_Force', {})[panel_index] = _kj
                if 'vertical' in surface_key.lower():
                    surface_data['Pressure_Difference'][panel_index] = np.dot(_kj, normal) / panel_area
                else:
                    surface_data['Pressure_Difference'][panel_index] = -np.dot(_kj, normal) / panel_area   # mesh normals point -z

            # --- surface force & moment via Kutta-Joukowski, per spanwise strip ---
            # Group panels into spanwise strips; per strip take the 1/4-chord bound vortex and its
            # trailing-wake edges (from the leading panel's ring), Gamma = -gamma_TE, and mean wind.
            columns = {}
            for panel_index in surface_data['Control Points']:
                columns.setdefault(panel_index[1], []).append(panel_index)

            bound_vector, quarter_chord, width, circ, strip_wind = [], [], [], [], []
            y_c, y_left, y_right = [], [], []
            for ps in columns.values():
                te_panel = max(ps, key=lambda p: p[0])
                corners = np.array(surface_data['Vortex Rings'][min(ps, key=lambda p: p[0])]['Vertices'])
                dl = corners[3] - corners[0]                          # spanwise bound segment
                span_dir = dl / (np.linalg.norm(dl) + 1e-12)
                edge_a, edge_b = np.dot(corners[0], span_dir), np.dot(corners[3], span_dir)
                bound_vector.append(dl); width.append(np.linalg.norm(dl))
                quarter_chord.append(0.5 * (corners[0] + corners[3]))
                y_c.append(np.dot(0.5 * (corners[0] + corners[3]), span_dir))
                y_left.append(min(edge_a, edge_b)); y_right.append(max(edge_a, edge_b))
                circ.append(-float(surface_data['Gamma'][te_panel]))    # Gamma = -gamma_TE
                strip_wind.append(np.mean([surface_data['Wind Velocity'].get(p, np.zeros(3)) for p in ps], axis=0))
            circ = np.array(circ)
            y_c, y_left, y_right = np.array(y_c), np.array(y_left), np.array(y_right)

            # Trailing-wake downwash per strip:  w = sum_k Gamma_k/(4pi)[1/(y-yL) - 1/(y-yR)]
            with np.errstate(divide='ignore'):
                leg_l = np.where(np.abs(y_c[:, None] - y_left[None, :]) > 1e-9, 1.0 / (y_c[:, None] - y_left[None, :]), 0.0)
                leg_r = np.where(np.abs(y_c[:, None] - y_right[None, :]) > 1e-9, 1.0 / (y_c[:, None] - y_right[None, :]), 0.0)
            downwash = (leg_l - leg_r) @ circ / (4.0 * np.pi)

            force, moment, induced_drag = np.zeros(3), np.zeros(3), 0.0
            for k in range(len(circ)):
                lift = rho * circ[k] * np.cross(-freestream + strip_wind[k], bound_vector[k])   # near-field KJ
                force += lift
                moment += np.cross(quarter_chord[k] - cog, lift)
                induced_drag += rho * circ[k] * downwash[k] * width[k]                           # far-field Trefftz

            # 2D-polar profile drag (computed with the viscous coupling, on the same strips)
            pd = surface_data.get('Profile_Drag')
            profile_drag = pd['scalar'] if pd else 0.0
            if pd:
                force = force + pd['force']
                moment = moment + pd['moment']

            results[surface_key] = {'force': force, 'moment': moment,
                                    'induced_drag': induced_drag, 'profile_drag': profile_drag}
            total_force += force
            total_moment += moment
            total_drag += induced_drag
            total_profile += profile_drag

        results['Total_Aircraft'] = {'force': total_force, 'moment': total_moment,
                                     'induced_drag': total_drag, 'profile_drag': total_profile}
        return results

    def _viscous_couple(self, aircraft_mesh, bound_matrices, airspeed, rho, cog):
        """Re-solve the circulation so each spanwise strip's VLM Cl matches its 2D airfoil polar,
        and (from the same polar, on the same strips) accumulate the 2D profile drag.

        Wing analogue of the rotor's polar coupling (solvers/vlm.py). The inviscid thin-camber solve
        caps the zero-lift angle near -3 deg; matching the section Cl to the polar at the effective
        angle alpha_eff = alpha_geo - alpha_i  (alpha_i = wake downwash / V, NO span-efficiency factor)
        recovers the real alpha_L0 (-4.2 deg for GA(W)-1) and stall. Each iteration relaxes the tangency
        RHS by (cl_vlm - cl_polar) and re-solves. Overwrites sd['Gamma'] and also stores
        sd['Profile_Drag'] = {'force','moment','scalar'} (D = 0.5 rho V^2 c ds Cd(Cl) per strip).
        """
        freestream = np.asarray(airspeed, float)
        speed = np.linalg.norm(freestream)

        for surface_key, surface_data in aircraft_mesh.items():
            panels = list(surface_data['Control Points'].keys())
            row_of = {p: i for i, p in enumerate(panels)}
            normals = np.array([surface_data['Normals'][p] for p in panels])
            wind = {p: surface_data['Wind Velocity'].get(p, np.zeros(3)) for p in panels}

            # Flow-tangency system  A gamma = rhs  (control points at 3/4 chord)
            A = np.einsum('ijk,ik->ij', bound_matrices[surface_key], normals)
            rhs = np.array([[-np.dot(-freestream + wind[p], surface_data['Normals'][p])] for p in panels])
            gamma = np.linalg.solve(A, rhs)

            chord = self.geometry.chord_of(surface_key)
            is_vertical = 'vertical' in surface_key.lower()
            cl_polar = self._cl_of_alpha(surface_key)

            # Group panels into spanwise strips; per strip keep the TE panel (circulation), the
            # trailing-wake edges y_left/y_right, and the geometric AoA from the local onset flow.
            columns = {}
            for p in panels:
                columns.setdefault(p[1], []).append(p)
            te_panels, strip_panels, alpha_geo, y_c, y_left, y_right = [], [], [], [], [], []
            for ps in columns.values():
                corners = np.array(surface_data['Vortex Rings'][min(ps, key=lambda p: p[0])]['Vertices'])
                span_dir = (corners[3] - corners[0]) / (np.linalg.norm(corners[3] - corners[0]) + 1e-12)
                edge_a, edge_b = np.dot(corners[0], span_dir), np.dot(corners[3], span_dir)
                y_c.append(np.dot(0.5 * (corners[0] + corners[3]), span_dir))
                y_left.append(min(edge_a, edge_b)); y_right.append(max(edge_a, edge_b))
                te_panels.append(max(ps, key=lambda p: p[0])); strip_panels.append(ps)
                onset = -freestream + np.mean([wind[p] for p in ps], axis=0)
                alpha_geo.append(np.degrees(np.arctan2(-onset[2], onset[0])))
            y_c, y_left, y_right = np.array(y_c), np.array(y_left), np.array(y_right)

            for _ in range(self.visc_maxit):
                gamma_prev = gamma.copy()

                # Trailing-wake downwash per strip:  w = sum_k Gamma_k/(4pi)[1/(y-yL) - 1/(y-yR)]
                circ = np.array([-float(gamma[row_of[te]]) for te in te_panels])   # Gamma = -gamma_TE
                with np.errstate(divide='ignore'):
                    leg_l = np.where(np.abs(y_c[:, None] - y_left[None, :]) > 1e-9, 1.0 / (y_c[:, None] - y_left[None, :]), 0.0)
                    leg_r = np.where(np.abs(y_c[:, None] - y_right[None, :]) > 1e-9, 1.0 / (y_c[:, None] - y_right[None, :]), 0.0)
                downwash = (leg_l - leg_r) @ circ / (4.0 * np.pi)

                # Match sectional Cl to the polar at the effective angle; relax the RHS by the residual
                residual = np.zeros_like(rhs)
                for k, ps in enumerate(strip_panels):
                    cl_vlm = -2.0 * float(gamma[row_of[te_panels[k]]]) / (speed * chord)
                    cl_target = 0.0 if is_vertical else float(cl_polar(alpha_geo[k] - np.degrees(downwash[k] / speed)))
                    for p in ps:
                        residual[row_of[p]] = -(cl_vlm - cl_target)

                rhs = rhs + self.visc_relax * residual
                gamma = np.linalg.solve(A, rhs)
                if np.max(np.abs(gamma - gamma_prev)) < self.visc_tol:
                    break

            for p in panels:
                surface_data['Gamma'][p] = float(gamma[row_of[p]])

            # Profile drag from the same 2D polar, on the same strips: D = 0.5 rho V^2 c ds Cd(Cl), along -V.
            if speed > 1e-9:
                ddir = -freestream / speed
                cd_of_cl = self._cd_of_cl(surface_key)
                Fp, Mp, Dp = np.zeros(3), np.zeros(3), 0.0
                for k, ps in enumerate(strip_panels):
                    cl = -2.0 * float(gamma[row_of[te_panels[k]]]) / (speed * chord)
                    ds = np.linalg.norm(surface_data['Tangential Vectors'][min(ps, key=lambda p: p[0])]['Tangential i'])
                    pos = np.mean([np.array(surface_data['Control Points'][p]) for p in ps], axis=0)
                    d_force = 0.5 * rho * speed ** 2 * chord * ds * float(cd_of_cl(cl)) * ddir
                    Fp += d_force; Mp += np.cross(pos - cog, d_force); Dp += np.linalg.norm(d_force)
                surface_data['Profile_Drag'] = {'force': Fp, 'moment': Mp, 'scalar': Dp}

    def calculate_total_forces_and_moments(self, aircraft_mesh, alpha_deg, airspeed,
                                         rho, wind_field, cog, reference_point):
        """Aerodynamic forces and moments for each surface and the total aircraft.

        Solves the AIC + circulation, then the viscous polar coupling (which also builds the 2D-polar
        profile drag), then the Kutta-Joukowski forces (lift + far-field induced drag) + profile drag
        in calculate_pressure_difference. Returns {'force','moment','induced_drag','profile_drag'} per
        surface + total.
        """
        bound_to_bound_matrices = self.calculate_bound_to_bound_induced_velocity_matrix(aircraft_mesh, alpha_deg)
        self.calculate_gamma(aircraft_mesh, bound_to_bound_matrices, alpha_deg,
                             airspeed=airspeed, wind_field=wind_field, reference_point=reference_point)

        # Viscous coupling: match each strip's Cl to its 2D polar and build the
        # profile drag from the same polar/strips.
        if self.include_viscous_coupling:
            self._viscous_couple(aircraft_mesh, bound_to_bound_matrices, airspeed, rho, cog)

        # Kutta-Joukowski forces (lift + far-field induced drag) + profile drag + per-panel pressures.
        forces = self.calculate_pressure_difference(
            aircraft_mesh, alpha_deg, airspeed, rho, wind_field, reference_point, cog=cog)

        return forces
