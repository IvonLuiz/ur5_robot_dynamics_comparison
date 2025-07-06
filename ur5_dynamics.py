import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from scipy.integrate import odeint
from spatialmath.base import trotz, transl

class UR5DynamicModel:
    def __init__(self):
        # Parâmetros do UR5 (valores aproximados)
        self.m = [3.7, 8.4, 2.33, 1.65, 1.25, 0.8]  # massas dos links (kg)
        self.l = [0.425, 0.392, 0.1, 0.1, 0.1, 0.1]  # comprimentos (m)
        self.r = [  # Centros de massa (relativos ao frame do link)
            [0, 0, 0.1],
            [0.2, 0, 0],
            [0, 0, -0.05],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0.02]
        ]
        
        # Tensores de inércia (simplificados) [Ixx, Iyy, Izz] (kg.m^2)
        self.I = [
            [0.1, 0.1, 0.05],
            [0.2, 0.2, 0.1],
            [0.05, 0.05, 0.03],
            [0.03, 0.03, 0.01],
            [0.02, 0.02, 0.01],
            [0.01, 0.01, 0.005]
        ]
        
        # Limites das juntas
        self.tau_max = np.array([150, 150, 150, 28, 28, 28])  # Torques máximos (Nm)
        self.qlim = [
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi]
        ]
        
        # Parâmetros DH (Denavit-Hartenberg)
        self.dh_params = [
            [0, 0.089159, 0, np.pi/2],
            [0, 0, -0.425, 0],
            [0, 0, -0.39225, 0],
            [0, 0.10915, 0, np.pi/2],
            [0, 0.09465, 0, -np.pi/2],
            [0, 0.0823, 0, 0]
        ]
        
        # Gravidade
        self.g = np.array([0, 0, -9.81])
        
    def forward_kinematics(self, q):
        """Calcula a cinemática direta para cada junta"""
        T = []
        current_T = SE3()
        
        for i in range(6):
            a, d, alpha, theta = self.dh_params[i]
            theta += q[i]
            
            Ti = SE3.Rz(theta) * SE3.Tz(d) * SE3.Tx(a) * SE3.Rx(alpha)
            current_T = current_T * Ti
            T.append(current_T)
        
        return T
    
    def jacobian(self, q):
        """Calcula o Jacobiano para cada centro de massa"""
        T = self.forward_kinematics(q)
        J = []
        
        for i in range(6):
            # Posição do centro de massa no frame global
            com_global = T[i] * SE3(transl(self.r[i]))
            
            # Jacobiano linear e angular
            Jvi = np.zeros((3, 6))
            Jwi = np.zeros((3, 6))
            
            for j in range(6):
                if j <= i:
                    # Eixo de rotação no frame global
                    z_axis = T[j].R[:, 2]
                    
                    # Jacobiano angular
                    Jwi[:, j] = z_axis
                    
                    # Jacobiano linear
                    p_com = com_global.t - T[j].t
                    Jvi[:, j] = np.cross(z_axis, p_com)
                else:
                    Jwi[:, j] = 0
                    Jvi[:, j] = 0
            
            J.append({'linear': Jvi, 'angular': Jwi})
        
        return J
    
    def inertia_matrix(self, q):
        """Calcula a matriz de inércia D(q)"""
        J = self.jacobian(q)
        D = np.zeros((6, 6))
        
        for i in range(6):
            Jv = J[i]['linear']
            Jw = J[i]['angular']
            R = self.forward_kinematics(q)[i].R
            
            # Tensor de inércia no frame global
            I_global = R @ np.diag(self.I[i]) @ R.T
            
            D += self.m[i] * Jv.T @ Jv + Jw.T @ I_global @ Jw
        
        return D
    
    def coriolis_matrix(self, q, qd):
        """Calcula a matriz de Coriolis C(q, qd)"""
        D = np.array(self.inertia_matrix(q))
        C = np.zeros((6, 6))
        h = 1e-6  # Passo para diferenças finitas

        for k in range(6):
            for j in range(6):
                for i in range(6):
                    # Calcula derivadas parciais via diferenças finitas
                    q_plus = q.copy()
                    q_plus[i] += h
                    D_plus = self.inertia_matrix(q_plus)
                    
                    q_minus = q.copy()
                    q_minus[i] -= h
                    D_minus = self.inertia_matrix(q_minus)
                    
                    dDkj_qi = (D_plus[k, j] - D_minus[k, j]) / (2 * h)
                    dDki_qj = (D_plus[k, i] - D_minus[k, i]) / (2 * h)
                    dDij_qk = (D_plus[i, j] - D_minus[i, j]) / (2 * h)
                    
                    # Símbolos de Christoffel
                    c_ijk = 0.5 * (dDkj_qi + dDki_qj - dDij_qk)
                    C[k, j] += c_ijk * qd[i]
        
        return C
    
    def gravity_vector(self, q):
        """Calcula o vetor de gravidade G(q)"""
        T = self.forward_kinematics(q)
        G = np.zeros(6)
        
        for i in range(6):
            # Posição do centro de massa no frame global
            com_global = T[i] * SE3(transl(self.r[i]))
            
            # Energia potencial
            P = self.m[i] * self.g @ com_global.t
            
            # Gradiente da energia potencial
            for j in range(6):
                # Aproximação numérica do gradiente
                epsilon = 1e-6
                q_plus = q.copy()
                q_plus[j] += epsilon
                T_plus = self.forward_kinematics(q_plus)
                com_plus = T_plus[i] * SE3(transl(self.r[i]))
                P_plus = self.m[i] * self.g @ com_plus.t
                
                G[j] += (P_plus - P) / epsilon
        
        return G
    
    def friction(self, qd):
        """Modelo de atrito simples"""
        return 0.1 * qd  # Atrito viscoso
    
    def dynamics(self, y, t, tau):
        """Equações dinâmicas para integração"""
        q = y[:6]
        qd = y[6:]
        
        D = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, qd)
        G = self.gravity_vector(q)
        F = self.friction(qd)
        
        qdd = np.linalg.inv(D) @ (tau - C @ qd - G - F)
        
        return np.concatenate((qd, qdd))
    
    def simulate(self, q0, qd0, tau, t_span):
        """Simula a dinâmica do robô"""
        y0 = np.concatenate((q0, qd0))
        sol = odeint(self.dynamics, y0, t_span, args=(tau,))
        return sol

def create_ur5_rtb():
    """Cria o modelo do UR5 na Robotics Toolbox"""
    # Parâmetros DH do UR5
    return DHRobot([
        RevoluteDH(d=0.089159, alpha=np.pi/2),
        RevoluteDH(a=0.425),
        RevoluteDH(a=0.39225),
        RevoluteDH(d=0.10915, alpha=np.pi/2),
        RevoluteDH(d=0.09465, alpha=-np.pi/2),
        RevoluteDH(d=0.0823)
    ], name="UR5")

def compare_models():
    """Compara nosso modelo com a Robotics Toolbox"""
    # Configuração inicial
    q0 = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    qd0 = np.zeros(6)
    
    # Torques aleatórios dentro dos limites
    np.random.seed(42)
    tau_max = np.array([150, 150, 150, 28, 28, 28])
    tau = np.random.uniform(-0.5, 0.5, 6) * tau_max
    
    # Tempo de simulação
    t_span = np.linspace(0, 5, 500)
    
    # Simulação com nosso modelo
    ur5 = UR5DynamicModel()
    sol_custom = ur5.simulate(q0, qd0, tau, t_span)
    
    # Simulação com Robotics Toolbox
    ur5_rtb = create_ur5_rtb()
    
    # Para a Robotics Toolbox, precisamos definir parâmetros dinâmicos
    # (usando os mesmos parâmetros do nosso modelo para comparação justa)
    ur5_rtb.links[0].m = ur5.m[0]
    ur5_rtb.links[1].m = ur5.m[1]
    ur5_rtb.links[2].m = ur5.m[2]
    ur5_rtb.links[3].m = ur5.m[3]
    ur5_rtb.links[4].m = ur5.m[4]
    ur5_rtb.links[5].m = ur5.m[5]
    
    ur5_rtb.links[0].r = ur5.r[0]
    ur5_rtb.links[1].r = ur5.r[1]
    ur5_rtb.links[2].r = ur5.r[2]
    ur5_rtb.links[3].r = ur5.r[3]
    ur5_rtb.links[4].r = ur5.r[4]
    ur5_rtb.links[5].r = ur5.r[5]
    
    # Como a Robotics Toolbox não tem uma função direta para simulação dinâmica com torques,
    # vamos usar a função fdyn que simula a dinâmica com controle de torque
    def control(t, x):
        return tau  # Aplicamos os mesmos torques constantes
    
    sol_rtb = ur5_rtb.fdyn(5, q0, qd0, control=control, dt=0.01)
    
    # Plotar resultados
    plt.figure(figsize=(14, 10))
    
    # Posições das juntas
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.plot(t_span, sol_custom[:, i], label='Nosso Modelo')
        plt.plot(sol_rtb.t, sol_rtb.q[:, i], '--', label='Robotics Toolbox')
        plt.title(f'Junta {i+1}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Posição (rad)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle('Comparação entre Nosso Modelo e Robotics Toolbox', y=1.02)
    plt.show()
    
    # Calcular e mostrar erros
    q_custom = sol_custom[-1, :6]  # Posições finais do nosso modelo
    q_rtb = sol_rtb.q[-1, :]       # Posições finais da Robotics Toolbox
    
    errors = q_custom - q_rtb
    print("\nErros nas posições finais das juntas:")
    for i in range(6):
        print(f"Junta {i+1}: {errors[i]:.6f} rad")
    
    print(f"\nErro RMS: {np.sqrt(np.mean(errors**2)):.6f} rad")

if __name__ == "__main__":
    compare_models()
