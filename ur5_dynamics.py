import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from scipy.integrate import odeint
from spatialmath.base import trotz, transl

class UR5DynamicModel:
    def __init__(self, verbose=True):
        # Center of mass of each link in its local frame (m)
        self.mass = np.array([3.7, 8.393, 2.33, 1.219, 1.219, 0.1879]) # Mass (kg)
        # Center of mass of each link in its local frame (m)
        self.r = np.array([
            [0.0, -0.02561, 0.00193], # L1
            [0.2125, 0.0, 0.11336],   # L2
            [0.15, 0.0, 0.0265],      # L3
            [0.0, -0.0018, 0.01634],  # L4
            [0.0, 0.0018, 0.01634],   # L5
            [0.0, 0.0, -0.001159],    # L6
        ])
        # Principal inertia about the COM (diagonal only) [Ixx, Iyy, Izz] (kg.m^2)
        self.I = np.array([
            [0.010267, 0.010267, 0.00666],
            [0.226890, 0.226890, 0.0151074],
            [0.049443, 0.049443, 0.004095],
            [0.111172, 0.111172, 0.21942],
            [0.111172, 0.111172, 0.21942],
            [0.017136, 0.017136, 0.033822],
        ])
        # Gravity acceleration vector (m/s²)
        self.gravity = np.array([0, 0, -9.81])
        # Denavit-Hartenberg (DH) Parameters [a, d, alpha, theta_offset]
        self.dh_params = [
            [0, 0.089159, 0, np.pi / 2],
            [0, 0, -0.425, 0],
            [0, 0, -0.39225, 0],
            [0, 0.10915, 0, np.pi / 2],
            [0, 0.09465, 0, -np.pi / 2],
            [0, 0.0823, 0, 0],
        ]

        self.l = [0.425, 0.392, 0.1, 0.1, 0.1, 0.1]  # lengths (m)

        # Joint limits
        self.tau_max = np.array([150, 150, 150, 28, 28, 28]) # (Nm)
        
        # Option to enable/disable logging
        self.verbose = verbose

    def forward_kinematics(self, q):
        '''!
        Calculates forward kinematics for each link.

        \param   q (np.ndarray): Joint positions (radians).
        \return  List[SE3]: Homogeneous transforms from base to each link.
        '''
        T = []
        current_T = SE3()
        
        for i in range(6):
            a, d, alpha, theta = self.dh_params[i]
            theta += q[i]
            
            Ti = SE3.Rz(theta) * SE3.Tz(d) * SE3.Tx(a) * SE3.Rx(alpha) # link transform
            current_T = current_T * Ti
            T.append(current_T)
        
        return T
    
    def jacobian(self, q):
        '''!
        Calculates the Jacobian for each center of mass.
        
        \param   q (np.ndarray): Joint positions (radians).
        \return  List[Tuple[np.ndarray, np.ndarray]]: List of tuples containing linear and angular Jacobians for each link.
        '''
        T = self.forward_kinematics(q)
        J = []
        
        for i in range(6):
            # Linear and angular Jacobian
            Jvi = np.zeros((3, 6))
            Jwi = np.zeros((3, 6))
            p_com_global = (T[i] * SE3(transl(self.r[i]))).t
            
            for j in range(6):
                if j <= i:
                    # rotational axis in global frame
                    z_axis = T[j].R[:, 2]
                    
                    # angular Jacobian
                    Jwi[:, j] = z_axis
                    
                    # linear jacobian
                    p_com = p_com_global - T[j].t
                    Jvi[:, j] = np.cross(z_axis, p_com)

            J.append((Jvi, Jwi))

        return J
    
    def inertia_matrix(self, q):
        '''!
        Calculates the manipulator's inertia matrix D(q) in joint space.

        \param   q (np.ndarray): Joint positions (radians).
        \return  np.ndarray: The 6x6 inertia matrix (D).
        '''
        J = self.jacobian(q)
        D = np.zeros((6, 6))
        
        for i in range(6):
            Jv_i, Jw_i = J[i]
            R_i = self.forward_kinematics(q)[i].R
            
            # Inertia tensor on global frame
            I_global = R_i @ np.diag(self.I[i]) @ R_i.T

            D += self.mass[i] * Jv_i.T @ Jv_i + Jw_i.T @ I_global @ Jw_i

        return D

    def coriolis_matrix(self, q, qd):
        '''!
        Calculates the Coriolis and centrifugal forces matrix C(q, qd).

        \param   q (np.ndarray): Joint positions (radians).
        \param   qd (np.ndarray): Joint velocities (radians/s).
        \param   D (np.ndarray, optional): Pre-calculated inertia matrix D(q).
        \return  np.ndarray: The 6x6 Coriolis matrix (C).
        '''
        D = self.inertia_matrix(q)
        C = np.zeros((6, 6))
        h = 1e-6

        for k in range(6):
            for j in range(6):
                for i in range(6):
                    # Calculate partial derivatives using finite differences
                    dq_i = np.zeros(6)
                    dq_i[i] = h
                    D_plus = self.inertia_matrix(q + dq_i)
                    D_minus = self.inertia_matrix(q - dq_i)
                    dDkj_dqi = (D_plus[k,j] - D_minus[k,j]) / (2*h)
                    
                    dq_j = np.zeros(6)
                    dq_j[j] = h
                    D_plus = self.inertia_matrix(q + dq_j)
                    D_minus = self.inertia_matrix(q - dq_j)
                    dDki_dqj = (D_plus[k,i] - D_minus[k,i]) / (2*h)
                    
                    dq_k = np.zeros(6)
                    dq_k[k] = h
                    D_plus = self.inertia_matrix(q + dq_k)
                    D_minus = self.inertia_matrix(q - dq_k)
                    dDij_dqk = (D_plus[i,j] - D_minus[i,j]) / (2*h)
                    
                    C[k,j] += 0.5 * (dDkj_dqi + dDki_dqj - dDij_dqk) * qd[i]
        
        return C
    
    def gravity_vector(self, q):
        '''!
        Calculates the gravity force vector G(q) in joint space.

        \param   q (np.ndarray): Joint positions (radians).
        \return  np.ndarray: The 6x1 gravity force vector (G).
        '''
        T = self.forward_kinematics(q)
        G = np.zeros(6)
        
        for i in range(6):
            # Posição do centro de massa no frame global
            com_global = T[i] * SE3(transl(self.r[i]))
            
            # Energia potencial
            P = self.mass[i] * self.gravity @ com_global.t
            
            # Gradiente da energia potencial
            for j in range(6):
                # Aproximação numérica do gradiente
                epsilon = 1e-6
                q_plus = q.copy()
                q_plus[j] += epsilon
                T_plus = self.forward_kinematics(q_plus)
                com_plus = T_plus[i] * SE3(transl(self.r[i]))
                P_plus = self.mass[i] * self.gravity @ com_plus.t
                
                G[j] += (P_plus - P) / epsilon
        
        return G
    
    def friction(self, qd):
        '''!
        Calculates friction forces in joint space (viscous model).

        \param   qd (np.ndarray): Joint velocities (radians/s).
        \return  np.ndarray: The 6x1 friction force vector.
        '''
        return 0.05 * qd
    
    def simulate(self, q0, qd0, tau, t_span, verbose=False):
        '''!
        Simulates the robot dynamics using numerical integration with optional logging.
        
        \param   q0 (np.ndarray): Initial joint positions (radians).
        \param   qd0 (np.ndarray): Initial joint velocities (radians/s).
        \param   tau (np.ndarray): Joint torques (Nm).
        \param   t_span (np.ndarray): Time span for the simulation (seconds).
        \param   verbose (bool): If True, prints simulation progress.
        \return  np.ndarray: The state trajectory [q, qd] over time.
        '''
        y0 = np.concatenate((q0, qd0))
        cache = {'last_q': None, 'last_D': None, 'last_C': None, 'last_G': None}
        self.verbose = verbose
        
        def dynamics_with_logging(y, t, tau):
            '''!
            ODE function to compute the dynamics with logging.
            dy/dt = [qd, qdd]
            
            \param   y (np.ndarray): State vector [q, qd].
            \param   t (float): Current time.
            \param   tau (np.ndarray): Joint torques (Nm).
            \return  np.ndarray: The derivative dy/dt.
            '''
            q = y[:6]
            qd = y[6:]
            if self.verbose:
                print(f"\rSimulating t = {t:.2f}s", end='', flush=True)
            if cache['last_q'] is None or np.any(np.abs(q - cache['last_q']) > 1e-6):
                D = self.inertia_matrix(q)
                C = self.coriolis_matrix(q, qd)
                G = self.gravity_vector(q)
                cache.update({'last_q': q.copy(), 'last_D': D, 'last_C': C, 'last_G': G})
            else:
                D, C, G = cache['last_D'], cache['last_C'], cache['last_G']
            F = self.friction(qd)
            qdd = np.linalg.solve(D, (tau - C @ qd - G - F))
            return np.concatenate((qd, qdd))
        
        if self.verbose:
            print("...")
        try:
            sol = odeint(
                dynamics_with_logging,
                y0,
                t_span,
                args=(tau,),
                rtol=1e-6,
                atol=1e-8,
                mxstep=1000
            )
            if self.verbose:
                print("\nSimulação concluída com sucesso!")
            return sol
        except Exception as e:
            print(f"\nErro durante a integração: {str(e)}")
            raise


if __name__ == "__main__":
    ur5_model = UR5DynamicModel()
    q_example = np.array([0.1, 0.2, 0.0, 0.4, 0.0, 0.6])
    qd_example = np.array([0.01, 0.02, 0.0, 0.04, 0.0, 0.06])

    print("--- UR5 Dynamic Model Test ---")
    T_links = ur5_model.forward_kinematics(q_example)
    print(f"End-effector Transform:\n{T_links[5].t}\n")

    J_links = ur5_model.jacobian(q_example)
    print(f"Link 3 Linear Jacobian (first 3 rows):\n{J_links[2][0][:3, :3]}\n")

    D_matrix = ur5_model.inertia_matrix(q_example)
    print(f"Inertia Matrix (D) shape: {D_matrix.shape}\n")

    C_matrix = ur5_model.coriolis_matrix(q_example, qd_example)
    print(f"Coriolis Matrix (C) shape: {C_matrix.shape}\n")

    G_vector = ur5_model.gravity_vector(q_example)
    print(f"Gravity Vector (G) shape: {G_vector.shape}\n")

    F_friction = ur5_model.friction(qd_example)
    print(f"Friction Forces (F_friction) shape: {F_friction.shape}\n")
