import sys
# sys.path.append(r"/Users/liyang/Documents/GitHub/Optically_polarized_atoms/my_functions")
sys.path.append(r"D:\Optically-pumped-atoms\my_functions")
import matplotlib.pyplot as plt
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import numpy as np
from matplotlib import ticker
from tqdm import trange
import scienceplots
from scipy import signal


def master_equation(I,Rse,omega_0,theta_B,phi_B,omega_pi,theta_pi,phi_pi,Rop,Rsd,T):
    a = round(I + 1 / 2)
    b = round(I - 1 / 2)

    # --------------------------------Generate the angular momentum operators-----------------------------------#
    U = alkali_atom_uncoupled_to_coupled(round(2 * I))
    ax, ay, az, bx, by, bz = spin_operators_of_2or1_alkali_metal_atoms(1, I)
    Sx = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmax().full()))
    Sx = U.T.conjugate() @ Sx @ U
    Sy = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmay().full()))
    Sy = U.T.conjugate() @ Sy @ U
    Sz = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmaz().full()))
    Sz = U.T.conjugate() @ Sz @ U

    # --------------------------------Characterize interactions envolved-----------------------------------#
    
    theta_op=0
    sx=np.sin(theta_op)
    sz=np.cos(theta_op)
    # --------------------------------Define the initial state-----------------------------------#
    theta = np.pi/4
    phi = np.pi/2
    a_theta = spin_Jx(a) * np.sin(theta) * np.cos(phi) + spin_Jy(a) * np.sin(theta) * np.sin(phi) + spin_Jz(a) * np.cos(
        theta)
    b_theta = spin_Jx(b) * np.sin(theta) * np.cos(phi) + spin_Jy(b) * np.sin(theta) * np.sin(phi) + spin_Jz(b) * np.cos(
        theta)
    qa, va = np.linalg.eig(np.array(a_theta.full()))
    qb, vb = np.linalg.eig(np.array(b_theta.full()))
    v = block_diag(va, vb)
    q = np.hstack((qa, qb))
    Rho_ini = np.zeros(2 * (a + b + 1))

    # # -----------------spin temperature state-----------------#
    P = 1e-10
    beta = np.log((1 + P) / (1 - P))
    for i in np.arange(0, 2 * (a + b + 1), 1):
        Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
    Rho_ini = Rho_ini / np.trace(Rho_ini)

    # -----------------eigenstates-----------------#

    # Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

    # --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
    H0 =omega_0z* (az - bz)+omega_0x* (ax - bx)+omega_0y* (ay - by)  # 投影定理
    Rhot = Rho_ini
    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    Py = np.zeros(round(T / dt))
    Px = np.zeros(round(T / dt))
    may = np.zeros(round(T / dt))
    for n in trange(0, round(T / dt), 1):
        Hpi = omega_pi[n]*np.cos(theta_pi) * (az - bz)+omega_pi[n]*np.sin(theta_pi) *np.cos(phi_pi)* (ax - bx)+omega_pi[n]*np.sin(theta_pi) *np.sin(phi_pi)* (ay - by)
        H=H0+Hpi
        qH, vH = np.linalg.eig(H)
        evolving_B = vH @ np.diag(np.exp(-1j * qH * dt)) @ np.linalg.inv(vH)
        x1 = Rhot @ Sx
        x2 = Rhot @ Sy
        x3 = Rhot @ Sz
        AS = 3 / 4 * Rhot - (Sx @ x1 + Sy @ x2 + Sz @ x3)
        alpha = Rhot - AS
        mSx = np.trace(x1)
        mSy = np.trace(x2)
        mSz = np.trace(x3)
        mSS = mSx * Sx + mSy * Sy + mSz * Sz
        ER = -Rsd * AS

        OP = Rop[n] * (2 * alpha @ (sx*Sx+sz*Sz) - AS)

        Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect

        Rhot = Rse * (alpha + 4 * alpha @ mSS - Rhot) * dt + (
                ER + OP) * dt + Rhot
        Rhot = hyperfine * Rhot
        # -----------------Observables-----------------#
        Px[n] = np.trace(2*Sx@Rhot)
        Py[n] = np.trace(2*Sy@Rhot)
        may[n] = np.trace(ay@Rhot)
    return Px,Py,may


global dt
dt=0.0001
T=3
t=np.arange(0,T,dt)


duration_op=2 #ms
amplitude_op = 50  #kHz#
frequency_op = 0.1 #kHz#
duty_op=duration_op*frequency_op
Rop = amplitude_op * signal.square(2 * np.pi * frequency_op * (t), duty=duty_op)+amplitude_op 

integer=160
frequency_pi = integer/8 #kHz，integer是个整数，这是为了使得光泵浦和pi脉冲的相位稳定
amplitude_pi = 300  #kHz#
duration_pi=np.pi/(2*amplitude_pi)  #ms#

duty_pi=duration_pi*frequency_pi
omega_pi = amplitude_pi * signal.square(2 * np.pi * frequency_pi * (t), duty=duty_pi)+amplitude_pi
theta_pi=np.pi/180*0.
phi_pi=np.pi/2


for k in np.arange(0, round(T / dt), 1): 
    if Rop[k]==2*amplitude_op:
        omega_pi[k]=0

Rsd = 100e-3   #20Hz#
Rse = 0.001    #5 Hz@50度#
omega_0x=0.1e-1
omega_0y=0.1e-1
omega_0z=0.5e-1





Px,Py,may=master_equation(3/2,Rse,omega_0x,omega_0y,omega_0z,omega_pi,theta_pi,phi_pi,Rop,Rsd,T)

t=np.arange(0,T,dt)
plt.style.use(['science'])
with plt.style.context(['science']):
    plt.rc('font',family='Times New Roman')
    fig = plt.figure(figsize=(4,8))
    
    ax1 = fig.add_subplot(311)
    ax1.plot(t,Py)
    ax1.set_ylabel('$P_y$', fontsize=8)
    # ax1.set_xlabel('$t(1/R_{\\text{se}})$', fontsize=8)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.tick_params(axis='both', which='minor', labelsize=8)
    ax1.set_xticklabels([])
    plt.title('$\omega_x=1$ rad/s')
    ax2 = fig.add_subplot(312)
    ax2.plot([],[])
    ax2.plot([],[])
    ax2.plot(t,omega_pi)
    ax2.set_ylabel('$\omega_{\pi}$ (kHz)', fontsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.tick_params(axis='both', which='minor', labelsize=8)
    ax2.set_xticklabels([])

    ax3 = fig.add_subplot(313)
    ax3.plot([],[])
    ax3.plot(t,Rop)
    ax3.set_ylabel('$R_{\\text{op}}$ (kHz)', fontsize=8)
    ax3.set_xlabel('t (ms)',fontsize=8)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax3.tick_params(axis='both', which='minor', labelsize=8)
    # ax1.set_xlim([2,5])
    # ax2.set_xlim([2,5])
    # ax3.set_xlim([2,5])
    plt.grid()
    plt.savefig('signal.png', dpi=1000)
plt.show()