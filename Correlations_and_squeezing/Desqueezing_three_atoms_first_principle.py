# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月17日
"""
import sys
sys.path.append(r"/Users/liyang/Documents/GitHub/Optically_polarized_atoms/my_functions")
# sys.path.append(r"D:\Optically-pumped-atoms\my_functions")

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from Generate_a_squeezed_state_by_QND import Generate_a_squeezed_state_by_QND
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from scipy.linalg import *
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from sympy.physics.quantum.spin import JzKet, JxKet
from sympy.physics.quantum.represent import represent
from matplotlib import rc
import scienceplots
from tqdm import trange
# ax, ay, az, bx, by, bz, a1x, a2x, a1y, a2y, a1z, a2z, Fx, Fy, Fz

# ----------------------Squeezing----------------------------#
# N is the number of atoms, T is the squeezing time, F is the spin of atom, s is the spin of light and alpha is the coupling constant
N = 3
I = 3 / 2
T_sq = 3.5
a = round(I + 1 / 2)
b = round(I - 1 / 2)
s = 5
alpha = 0.2
dt = 0.02
S = 1 / 2
U = alkali_atom_uncoupled_to_coupled(round(2 * I))
# ----------------------spin operators----------------------#
ax = spin_Jx(a)
ay = spin_Jy(a)
az = spin_Jz(a)
bx = spin_Jx(b)
by = spin_Jy(b)
bz = spin_Jz(b)

omega_e= 40
Ahf=6000
T = 350
dt1 = 1e-4
dt2 = 1e-5

if N == 3:
    a1x, a2x, a3x, a1y, a2y, a3y, a1z, a2z, a3z, b1x, b2x, b3x, b1y, b2y, b3y, b1z, b2z, b3z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(
        3, I)
    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    hyperfine = np.kron(hyperfine, np.kron(hyperfine, hyperfine))  # 三个原子
    
    Fx = a1x + a2x + a3x + b1x + b2x + b3x
    Fz = a1z + a2z + a3z + b1z + b2z + b3z
    Sx = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmax().full()))
    Sx = U.T.conjugate() @ Sx @ U
    Sy = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmay().full()))
    Sy = U.T.conjugate() @ Sy @ U
    Sz = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmaz().full()))
    Sz = U.T.conjugate() @ Sz @ U

    S1x = np.kron(np.kron(Sx, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    S2x = np.kron(np.kron(np.eye(2 * (a + b + 1)), Sx), np.eye(2 * (a + b + 1)))
    S3x = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), Sx)
    S1y = np.kron(np.kron(Sy, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    S2y = np.kron(np.kron(np.eye(2 * (a + b + 1)), Sy), np.eye(2 * (a + b + 1)))
    S3y = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), Sy)
    S1z = np.kron(np.kron(Sz, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    S2z = np.kron(np.kron(np.eye(2 * (a + b + 1)), Sz), np.eye(2 * (a + b + 1)))
    S3z = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), Sz)

    Ps13 = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) - (S1x @ S3x + S1y @ S3y + S1z @ S3z)
    Pt13 = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) + (S1x @ S3x + S1y @ S3y + S1z @ S3z)
    Pe13 = Pt13 - Ps13

    Ps12 = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) - (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    Pt12 = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) + (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    Pe12 = Pt12 - Ps12

    Ps23 = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) - (S2x @ S3x + S2y @ S3y + S2z @ S3z)
    Pt23 = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) + (S2x @ S3x + S2y @ S3y + S2z @ S3z)
    Pe23 = Pt23 - Ps23

    Ix = np.kron(spin_Jx(I).full(), np.eye(round(2 * S+ 1)))
    Ix = U.T.conjugate() @ Ix @ U
    Iy = np.kron(spin_Jy(I).full(), np.eye(round(2 * S + 1)))
    Iy = U.T.conjugate() @ Iy @ U
    Iz = np.kron(spin_Jz(I).full(), np.eye(round(2 * S + 1)))
    Iz = U.T.conjugate() @ Iz @ U

    I1x = np.kron(np.kron(Ix, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    I2x = np.kron(np.kron(np.eye(2 * (a + b + 1)), Ix), np.eye(2 * (a + b + 1)))
    I3x = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), Ix)
    I1y = np.kron(np.kron(Iy, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    I2y = np.kron(np.kron(np.eye(2 * (a + b + 1)), Iy), np.eye(2 * (a + b + 1)))
    I3y = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), Iy)
    I1z = np.kron(np.kron(Iz, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    I2z = np.kron(np.kron(np.eye(2 * (a + b + 1)), Iz), np.eye(2 * (a + b + 1)))
    I3z = np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), Iz)

    mathcal_F=S1z+S2z+S3z
    """
    Zeeman effect
    """
    H_z = omega_e * (S1x+S2x+S3x)  # 两个原子
    """
      Hyperfine interaction
    """
    #等效法
    # hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    # hyperfine = np.kron(hyperfine, hyperfine)  # 两个原子
    # 第一性原理
    H_h = Ahf *( I1x@S1x+I1y@S1y+I1z@S1z+I2x@S2x+I2y@S2y+I2z@S2z+I3x@S3x+I3y@S3y+I3z@S3z) # 两个原子
# ----------------------squeezing----------------------#
ini_Rho_atom, Rho_atomi = Generate_a_squeezed_state_by_QND(3, I, T_sq, s, alpha, dt)

# ----------------------Evolution of the spin under magnetic field and hyperfine interaction----------------------#
n1 = round(T / dt1)
n2 = round(T / dt2)
C_1 = [None] * n1
C_2 = [None] * n2

qz, vz = np.linalg.eig(H_z)
qh, vh = np.linalg.eig(H_h)
evolving_B = vz @ np.diag(np.exp(-1j * qz * dt)) @ np.linalg.inv(vz)
evolving_h = vh @ np.diag(np.exp(-1j * qh * dt)) @ np.linalg.inv(vh)

Rho_atom = Rho_atomi
for t in trange(0, n1, 1):
    C_1[t] = np.trace(Rho_atom @ mathcal_F@ mathcal_F) - np.trace(Rho_atom @mathcal_F) ** 2
    hh=np.random.uniform()
    if hh<0.5:
        r = np.random.uniform()
        if r  < 0.33:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe12
        elif r < 0.66:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe13
        else:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe23
        Rho_atom = sec @ Rho_atom @ sec.T.conjugate()
    Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    Rho_atom = evolving_h @ Rho_atom @ evolving_h.T.conjugate()


Rho_atom = Rho_atomi
for t in np.arange(0, n2, 1):
    C_2[t] =np.trace(Rho_atom @ mathcal_F@ mathcal_F) - np.trace(Rho_atom @mathcal_F) ** 2
    hh=np.random.uniform()
    if hh<0.05:
        r = np.random.uniform()
        if r  < 0.33:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe12
        elif r < 0.66:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe13
        else:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe23
        Rho_atom = sec @ Rho_atom @ sec.T.conjugate()
    Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    Rho_atom = evolving_h @ Rho_atom @ evolving_h.T.conjugate()

C_1=np.array(C_1)
C_2=np.array(C_2)

CSS=np.array([1,0,0,0,0,0,0,0])
CSS3=np.kron(np.kron(CSS,CSS),CSS)
Rho_CSS=np.outer(CSS3,CSS3)
mathcal_Fx=S1x+S2x+S3x
mathcal_Fz=S1z+S2z+S3z
Varcss=np.trace(Rho_CSS@mathcal_Fx@mathcal_Fx)-np.trace(Rho_CSS@mathcal_Fx)**2
Varsss=np.trace(Rho_atomi@mathcal_Fz@mathcal_Fz)-np.trace(Rho_atomi@mathcal_Fz)**2

tt1 = np.arange(0, n1, 1)*dt1
tt2 = np.arange(0, n2, 1)*dt2

with plt.style.context(['science']):
    fig = plt.figure(figsize=(3,6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    p1, = ax1.plot(tt1, C_1 )
    p6, = ax1.plot(tt1, np.ones(len(tt1))*Varcss)
    p7, = ax1.plot(tt1, np.ones(len(tt1))*Varsss)

    p2, = ax2.plot(tt2, C_2) 
    p6, = ax2.plot(tt2, np.ones(len(tt2))*Varcss)
    p7, = ax2.plot(tt2, np.ones(len(tt2))*Varsss)
    # ax1.legend([p1,p6,p7],
    #            ["$R_{\\text{se}}=36$ Hz,$\omega_e=60$ rad/s,dt=1e-5/6 s","CSS","SSS"],ncol=1)
    ax1.set_xlabel('$t$ (s)')
    ax1.set_ylabel('Var $( \mathcal S_{x})$')
    ax2.set_xlabel('$t$ (s)')
    ax2.set_ylabel('Var $( \mathcal S_{x})$')
    # ax2.set_xlim(0, 18)
    # ax1.set_xlim(0, 18)
    # plt.ylim(0.1,0.55)
    plt.savefig('desqueezing.png', dpi=600)
