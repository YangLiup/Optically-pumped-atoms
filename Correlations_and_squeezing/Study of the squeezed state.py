# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
# -*- coding:utf-8 -*-
import sys
sys.path.append(r"D:\Optically-pumped-atoms\my_functions")
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from Generate_a_squeezed_state_by_QND import Generate_a_squeezed_state_by_QND
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from scipy.linalg import *
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from sympy.physics.quantum.spin import JzKet, JxKet
from sympy.physics.quantum.represent import represent
from ptr import ptr
from scipy.linalg import *
import scienceplots
from matplotlib.ticker import FuncFormatter

# ax, ay, az, bx, by, bz, a1x, a2x, a1y, a2y, a1z, a2z, Fx, Fy, Fz

# ----------------------Squeezing----------------------------#
# N is the number of atoms, T is the squeezing time, F is the spin of atom, s is the spin of light and alpha is the coupling constant
N = 2
I = 3 / 2
T = 2
a = round(I + 1 / 2)
b = round(I - 1 / 2)
s = 5
alpha = 0.25
dt = 0.02
S = 1 / 2
U = alkali_atom_uncoupled_to_coupled(round(2 * I))
# ----------------------squeezing----------------------#

a = round(I + 1 / 2)
b = round(I - 1 / 2)
if N == 1:
    ax, ay, az, bx, by, bz, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(N, I)
if N == 2:
    a1x, a2x, a1y, a2y, a1z, a2z, b1x, b2x, b1y, b2y, b1z, b2z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(
        N, I)
sx = spin_Jx(s)
sy = spin_Jy(s)
sz = spin_Jz(s)
qy, vy = sy.eigenstates()

# initiation
H = alpha * np.kron(Fz, sz.full())

XiF_ini = np.vstack((np.array(spin_coherent(a, np.pi / 2, 0).full()), np.array(zero_ket(2 * b + 1).full())))
if N == 2:
    XiF_ini = np.kron(XiF_ini, XiF_ini)

ini_Rho_atom = np.outer(XiF_ini, XiF_ini)
Xis_ini = np.array(spin_coherent(s, np.pi / 2, 0).full())
Rhos_ini = np.outer(Xis_ini, Xis_ini)
Rho_ini = np.kron(ini_Rho_atom, Rhos_ini)
Rhot = Rho_ini
n = round(T / dt)
C1 = [None] * n
C2 = [None] * n
Vz0 = np.trace(ini_Rho_atom @(a1z+a2z-b1z-b2z) @ (a1z+a2z-b1z-b2z))/16

# evolving
for i in np.arange(0, n, 1):
    Rhot = dt * (H @ Rhot - Rhot @ H) / 1j + Rhot
    Rhot_sample = Rhot
    read = tensor(tensor(qeye(2 * (a + b + 1)), qeye(2 * (a + b + 1)), vy[1] * vy[1].dag())).full()
    Rho_r = read @ Rhot_sample @ read.T.conjugate()
    Rho_r = Rho_r / Rho_r.trace()
    Rho_atom = ptr(Rho_r, 2 * s + 1, (2 * (a + b + 1)) ** N)
    C1[i] = 1 - (np.trace(Rho_atom @ (a1z+a2z-b1z-b2z) @ (a1z+a2z-b1z-b2z))/16-np.trace(Rho_atom @ (a1z+a2z-b1z-b2z))**2)/16 / Vz0
    C2[i] =  np.trace(Rho_atom @ (a1x+a2x-b1x-b2x)/4)

tt = np.arange(0, T, dt)
plt.style.use(['science'])
with plt.style.context(['science']):
    plt.figure()
    plt.plot(C1,C2,linewidth='1.5')
    # plt.gca().invert_xaxis() 
    # plt.xlim(0, 1)
    # plt.ylim(0, 0.4)
    plt.xlabel('$1-\\frac {\mathrm{var}(\mathcal S_x)_{\mathrm{s}}} {\mathrm{var}(\mathcal S_x)_0}$')
    plt.ylabel('$\langle \mathcal S_x \\rangle_{\mathrm{s}}$')
    plt.savefig('squeezing.png', dpi=600)

# plt.figure()
# plt.plot(t, C_1x2x)
# plt.figure()
# plt.plot(t, (np.array(C_1x2x)+np.array(C_1x1x)))

plt.show()
