import numpy as np
import matplotlib.pyplot as plt
import os
from roboticstoolbox import DHRobot, RevoluteDH
from src.ur5_dynamics import UR5DynamicModel


def compare_models(tau, custom_model: UR5DynamicModel, sol_custom, rtb_robot: DHRobot, t_span: np.ndarray):
    """
    Compares custom UR5 model simulation with Robotics Toolbox.

    Simulates both models under identical conditions (zero gravity, constant torque)
    and plots joint positions and velocities, saving them to a 'results' folder.

    \param tau: Torque input for the simulation
    \param custom_model: UR5Simulator instance with custom model
    \param sol_custom: Simulation results from the custom model
    \param rtb_robot: DHRobot instance from Robotics Toolbox
    \param t_span: Time span for the simulation
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving plots to: {os.path.abspath(results_dir)}")

    q0 = np.zeros(6)
    qd0 = np.zeros(6)

    q_custom, qd_custom = sol_custom[:,:6], sol_custom[:,6:]

    print("Configuring and simulating Robotics Toolbox model...")
    # Set dynamic parameters for RTB from the custom model
    rtb_robot.gravity = custom_model.gravity
    for i, link in enumerate(rtb_robot.links):
        link.m = float(custom_model.mass[i])
        link.r = custom_model.r[i].tolist()
        Ix, Iy, Iz = custom_model.I[i]
        link.I = np.diag([Ix, Iy, Iz])  # Convert to diagonal inertia matrix
        link.B = custom_model.B

    print("RTB model with parameters:")
    for i, link in enumerate(rtb_robot.links):
        print(f"Link {i+1}: m={link.m}, r={link.r}, I={link.I}")
    print("Custom model parameters:")
    for i in range(len(custom_model.I)):
        print(f"Link {i+1}: m={custom_model.mass[i]}, r={custom_model.r[i]}, I={custom_model.I[i]}")

    def Q_rtb(robot_instance: DHRobot, t: float, q_val: np.ndarray, qd_val: np.ndarray) -> np.ndarray:
        return tau

    # RTB's fdyn returns q, qd, qdd. We'll only use q and qd.
    sol_rtb = rtb_robot.fdyn(t_span[-1], q0, Q=Q_rtb, qd0=qd0, solver="RK45", dt=0.001)
    q_rtb, qd_rtb = sol_rtb.q, sol_rtb.qd
    t_rtb = sol_rtb.t

    # --- Plotting ---
    plot_types = ["Position", "Velocity"]
    plot_units = ["rad", "rad/s"]
    plot_filenames = ["position.png", "velocity.png"]

    plt.style.use('seaborn-v0_8-darkgrid')

    for k, (custom_data, rtb_data) in enumerate(
        [(q_custom, q_rtb), (qd_custom, qd_rtb)]
    ):
        print(custom_data.shape, rtb_data.shape)
        print(t_span.shape, t_rtb.shape)
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i in range(6):
            ax = axes[i]
            ax.plot(t_span, custom_data[:, i], label="Custom Model", linewidth=2, color='#1f77b4')
            ax.plot(t_rtb, rtb_data[:, i], "--", label="RTB Model", linewidth=1.5, alpha=0.7, color='#ff7f0e')
            
            ax.set_title(f"Junta {i+1} – {plot_types[k]}", fontsize=12)
            ax.set_xlabel("Tempo (s)", fontsize=10)
            ax.set_ylabel(plot_units[k], fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=9)
            
            if i == 0:
                ax.legend(fontsize=10, loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f"Comparison: {plot_types[k]} Joint", y=0.98, fontsize=18, fontweight='bold')

        filepath = os.path.join(results_dir, plot_filenames[k])
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("\nPlotting complete. Figures saved to the 'results' folder.")

    # --- Error Analysis (Final Position) ---
    err_final_q = q_custom[-1, :] - q_rtb[-1, :]
    
    print("\nFinal Joint Position Errors (Custom vs RTB):")
    for i in range(6):
        print(f"Joint {i+1}: {err_final_q[i]:.6e} rad")

    rms_error_final_q = np.sqrt(np.mean(err_final_q ** 2))
    print(f"\nRMS Error of Final Joint Positions: {rms_error_final_q:.6e} rad")
    
    print("\nNote: Small discrepancies are expected due to differing internal solvers/precision.")
