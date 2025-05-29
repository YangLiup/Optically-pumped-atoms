import numpy as np
import matplotlib.pyplot as plt

def polarization_after_waveplate(theta_in_deg, delta_deg, waveplate_angle_deg=0):
    """
    Calculate the polarization state after a linearly polarized light passes through a waveplate.
    
    Parameters:
    theta_in_deg: Angle of input polarization (degrees)
    delta_deg: Phase retardation of waveplate (degrees)
    waveplate_angle_deg: Fast axis angle of waveplate (degrees, default=0)
    
    Returns:
    Ex, Ey: Electric field components in x and y directions
    """
    # Convert degrees to radians
    theta_in = np.radians(theta_in_deg)
    delta = np.radians(delta_deg)
    waveplate_angle = np.radians(waveplate_angle_deg)
    
    # Input Jones vector
    E_in = np.array([np.cos(theta_in), np.sin(theta_in)])
    
    # Waveplate Jones matrix
    J = np.array([[np.exp(-1j * delta / 2), 0],
                  [0, np.exp(1j * delta / 2)]])
    
    # Rotate to waveplate coordinates if needed
    if waveplate_angle != 0:
        c, s = np.cos(waveplate_angle), np.sin(waveplate_angle)
        R = np.array([[c, s], [-s, c]])
        R_inv = np.array([[c, -s], [s, c]])
        J = R_inv @ J @ R
    
    # Output Jones vector
    E_out = J @ E_in
    
    # Time evolution
    t = np.linspace(0, 2*np.pi, 1000)
    Ex = np.real(E_out[0] * np.exp(1j * t))
    Ey = np.real(E_out[1] * np.exp(1j * t))
    
    return Ex, Ey

def plot_polarization_ellipse(Ex, Ey,input_angle):
    """Plot the polarization ellipse with equal aspect ratio"""
    plt.figure(figsize=(6, 6))
    
    # Plot the trajectory
    plt.plot(Ex, Ey, 'b-', linewidth=2)
    plt.plot(np.arange(-1,1,0.01),np.arange(-1,1,0.01)*np.tan(np.radians(input_angle)))
    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Set axis limits based on data range
    max_val = max(np.max(np.abs(Ex)), np.max(np.abs(Ey))) * 1.1
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    
    # Add labels and grid
    plt.xlabel('Ex (a.u.)', fontsize=12)
    plt.ylabel('Ey (a.u.)', fontsize=12)
    plt.title('Polarization State After Waveplate', fontsize=14)
    plt.grid(True)
        
    plt.tight_layout()
    plt.show()

# Parameters for the simulation
input_angle = 30  # degrees
retardation = 70  # degrees
fast_axis_angle = 0  # degrees (along x-axis)

# Calculate the output polarization
Ex, Ey = polarization_after_waveplate(input_angle, retardation, fast_axis_angle)

# Plot the results
plot_polarization_ellipse(Ex, Ey,input_angle)
