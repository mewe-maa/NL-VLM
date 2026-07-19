import numpy as np
from scipy.spatial import cKDTree

class WindField:
    def __init__(self, mesh_data):
        """
        Initialize wind field using direct mesh data with pre-computed data structures
        """
        self.mesh = mesh_data
        
        # Pre-compute and store all points and velocities
        self.points = mesh_data.points
        self.velocities = mesh_data.get_array('U')
        
        # Create KD-tree for fast nearest neighbor lookup
        self.kdtree = cKDTree(self.points)

        self._compute_jacobian()
        # print("Jacobian computed successfully")

        # self._compute_hessian()
        # print("Hessian computed successfully")

    def _compute_jacobian(self):
        """Compute spatial gradients (Jacobian) for the wind field"""
        # Extract velocity components
        self.u = self.velocities[:, 0]  # u component
        self.v = self.velocities[:, 1]  # v component  
        self.w = self.velocities[:, 2]  # w component
        
        # Add as scalar fields for gradient computation
        self.mesh['u_scalar'] = self.u
        self.mesh['v_scalar'] = self.v  
        self.mesh['w_scalar'] = self.w
            
        # Compute gradients using PyVista
        u_grad = self.mesh.compute_derivative(scalars='u_scalar')['gradient']
        v_grad = self.mesh.compute_derivative(scalars='v_scalar')['gradient']
        w_grad = self.mesh.compute_derivative(scalars='w_scalar')['gradient']
        
        # Store all 9 Jacobian components
        self.dudx = u_grad[:, 0]
        self.dudy = u_grad[:, 1] 
        self.dudz = u_grad[:, 2]
        
        self.dvdx = v_grad[:, 0]
        self.dvdy = v_grad[:, 1]
        self.dvdz = v_grad[:, 2]
        
        self.dwdx = w_grad[:, 0]
        self.dwdy = w_grad[:, 1]
        self.dwdz = w_grad[:, 2]

    def get_jacobian_at_point(self, position):
        """
        Get the Jacobian matrix at a specific position
        """
        # Find nearest point using existing KD-tree
        distance, idx = self.kdtree.query(position)
        
        # Build 3x3 Jacobian matrix at that point
        jacobian_matrix = np.array([
            [self.dudx[idx], self.dudy[idx], self.dudz[idx]],
            [self.dvdx[idx], self.dvdy[idx], self.dvdz[idx]], 
            [self.dwdx[idx], self.dwdy[idx], self.dwdz[idx]]
        ])
        
        return jacobian_matrix
        
    def get_jacobian_approximated_velocity(self, position, com_position):
        """
        Get wind velocity using Jacobian approximation from COM
        """
        com_velocity = self.get_wind_velocity(com_position)
        
        com_jacobian = self.get_jacobian_at_point(com_position)
        
        delta_r = np.array(position) - np.array(com_position)
        
        approximated_velocity = com_velocity + com_jacobian @ delta_r
        
        return approximated_velocity 
    

    def _compute_hessian(self):
        """Compute second derivatives (Hessian) for the wind field"""
        # Add first derivatives as scalar fields
        self.mesh['dudx_scalar'] = self.dudx
        self.mesh['dudy_scalar'] = self.dudy
        self.mesh['dudz_scalar'] = self.dudz
        
        self.mesh['dvdx_scalar'] = self.dvdx
        self.mesh['dvdy_scalar'] = self.dvdy
        self.mesh['dvdz_scalar'] = self.dvdz
        
        self.mesh['dwdx_scalar'] = self.dwdx
        self.mesh['dwdy_scalar'] = self.dwdy
        self.mesh['dwdz_scalar'] = self.dwdz
        
        # Compute second derivatives for u-component
        dudx_grad = self.mesh.compute_derivative(scalars='dudx_scalar')['gradient']
        dudy_grad = self.mesh.compute_derivative(scalars='dudy_scalar')['gradient']
        dudz_grad = self.mesh.compute_derivative(scalars='dudz_scalar')['gradient']
        
        self.d2udx2 = dudx_grad[:, 0] 
        self.d2udxdy = dudx_grad[:, 1]  
        self.d2udxdz = dudx_grad[:, 2]  
        self.d2udydx = dudy_grad[:, 0]  
        self.d2udy2 = dudy_grad[:, 1]
        self.d2udydz = dudy_grad[:, 2]  
        self.d2udzdx = dudz_grad[:, 0]  
        self.d2udzdy = dudz_grad[:, 1]  
        self.d2udz2 = dudz_grad[:, 2] 
        
        # Compute second derivatives for v-component
        dvdx_grad = self.mesh.compute_derivative(scalars='dvdx_scalar')['gradient']
        dvdy_grad = self.mesh.compute_derivative(scalars='dvdy_scalar')['gradient']
        dvdz_grad = self.mesh.compute_derivative(scalars='dvdz_scalar')['gradient']
        
        self.d2vdx2 = dvdx_grad[:, 0]
        self.d2vdxdy = dvdx_grad[:, 1]
        self.d2vdxdz = dvdx_grad[:, 2]
        self.d2vdydx = dvdy_grad[:, 0]
        self.d2vdy2 = dvdy_grad[:, 1]
        self.d2vdydz = dvdy_grad[:, 2]
        self.d2vdzdx = dvdz_grad[:, 0]
        self.d2vdzdy = dvdz_grad[:, 1]
        self.d2vdz2 = dvdz_grad[:, 2]
        
        # Compute second derivatives for w-component
        dwdx_grad = self.mesh.compute_derivative(scalars='dwdx_scalar')['gradient']
        dwdy_grad = self.mesh.compute_derivative(scalars='dwdy_scalar')['gradient']
        dwdz_grad = self.mesh.compute_derivative(scalars='dwdz_scalar')['gradient']
        
        self.d2wdx2 = dwdx_grad[:, 0]
        self.d2wdxdy = dwdx_grad[:, 1]
        self.d2wdxdz = dwdx_grad[:, 2]
        self.d2wdydx = dwdy_grad[:, 0]
        self.d2wdy2 = dwdy_grad[:, 1]
        self.d2wdydz = dwdy_grad[:, 2]
        self.d2wdzdx = dwdz_grad[:, 0]
        self.d2wdzdy = dwdz_grad[:, 1]
        self.d2wdz2 = dwdz_grad[:, 2]

    def get_hessian_at_point(self, position):
        """
        Get the Hessian tensor (3x3x3) at a specific position
        Returns 3 matrices: one for each velocity component (u, v, w)
        """
        # Find nearest point using existing KD-tree
        distance, idx = self.kdtree.query(position)
        
        # Build Hessian matrices for each velocity component
        hessian_u = np.array([
            [self.d2udx2[idx], self.d2udxdy[idx], self.d2udxdz[idx]],
            [self.d2udydx[idx], self.d2udy2[idx], self.d2udydz[idx]],
            [self.d2udzdx[idx], self.d2udzdy[idx], self.d2udz2[idx]]
        ])
        
        hessian_v = np.array([
            [self.d2vdx2[idx], self.d2vdxdy[idx], self.d2vdxdz[idx]],
            [self.d2vdydx[idx], self.d2vdy2[idx], self.d2vdydz[idx]],
            [self.d2vdzdx[idx], self.d2vdzdy[idx], self.d2vdz2[idx]]
        ])
        
        hessian_w = np.array([
            [self.d2wdx2[idx], self.d2wdxdy[idx], self.d2wdxdz[idx]],
            [self.d2wdydx[idx], self.d2wdy2[idx], self.d2wdydz[idx]],
            [self.d2wdzdx[idx], self.d2wdzdy[idx], self.d2wdz2[idx]]
        ])
        
        return {
            'hessian_u': hessian_u,
            'hessian_v': hessian_v, 
            'hessian_w': hessian_w
        }
    
    def get_hessian_approximated_velocity(self, position, com_position):
        """
        Get wind velocity using Hessian approximation from COM (second-order Taylor expansion)
        """
        com_velocity = self.get_wind_velocity(com_position)
        
        com_jacobian = self.get_jacobian_at_point(com_position)
        
        com_hessian = self.get_hessian_at_point(com_position)
        
        delta_r = np.array(position) - np.array(com_position)
        
        # First-order term (same as Jacobian method)
        first_order = com_jacobian @ delta_r
        
        # Second-order terms for each velocity component
        second_order_u = 0.5 * delta_r.T @ com_hessian['hessian_u'] @ delta_r
        second_order_v = 0.5 * delta_r.T @ com_hessian['hessian_v'] @ delta_r
        second_order_w = 0.5 * delta_r.T @ com_hessian['hessian_w'] @ delta_r
        
        second_order = np.array([second_order_u, second_order_v, second_order_w])
        
        # Final approximation
        approximated_velocity = com_velocity + first_order + second_order
        
        return approximated_velocity

    def get_wind_velocity(self, position):
        """
        Get wind velocity using pre-computed KD-tree for fast lookup
        """
        # Find nearest point using KD-tree
        distance, idx = self.kdtree.query(position)
        
        return self.velocities[idx]

    @staticmethod
    def update_wind_function(wind_field, com_position):
        """
        Creates a wind function that provides wind velocity relative to the COM position.
        """
        def wind_velocity(position):
            
            absolute_position = position
            return wind_field.get_wind_velocity(absolute_position)


        return wind_velocity