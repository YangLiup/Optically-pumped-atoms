# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2024年03月07日
"""
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from my_functions.Generate_a_squeezed_state_by_QND import Generate_a_squeezed_state_by_QND
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from scipy.linalg import *
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from sympy.physics.quantum.spin import JzKet, JxKet
from sympy.physics.quantum.represent import represent
from my_functions.ptr import ptr
from scipy.linalg import *
import scienceplots
from matplotlib.ticker import FuncFormatter
import random
# ax, ay, az, bx, by, bz, a1x, a2x, a1y, a2y, a1z, a2z, Fx, Fy, Fz

# ----------------------Squeezing----------------------------#
# N is the number of atoms, T is the squeezing time, F is the spin of atom, s is the spin of light and alpha is the coupling constant
T = 10
F= 2
s = 10
alpha = 0.1
dt = 0.01
# ----------------------squeezing----------------------#

sx = np.array(spin_Jx(s))
sy = np.array(spin_Jy(s))
sz = np.array(spin_Jz(s))
qy, vy = np.linalg.eig(sy)
Fx = np.array(spin_Jx(F))
qF, vF = np.linalg.eig(Fx)
Fy = np.array(spin_Jy(F))
Fz = np.array(spin_Jz(F))
# initiation
H = alpha * np.kron(Fz, sz)


plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    plt.figure()
    for j in np.arange(0, 1, 1):
        # XiF_ini = np.array(spin_coherent(F, np.pi / 2, 0))
        XiF_ini = vF[:,3]
        ini_Rho_atom = np.outer(XiF_ini, XiF_ini)
        Xis_ini = np.array(spin_coherent(s, np.pi / 2, 0))
        Rhos_ini = np.outer(Xis_ini, Xis_ini)
        Rho_ini = np.kron(ini_Rho_atom, Rhos_ini)
        Rhot = Rho_ini
        n = round(T / dt)
        C_1z1z = [None] * n
        C_1z2z = [None] * n
        for i in np.arange(0, n, 1):
            Rhot = dt * (H @ Rhot - Rhot @ H) / 1j + Rhot
            k=random.randint(0,20)
            read = np.kron(np.eye(2*F+1), np.outer(vy[:,k] , vy[:,k].T.conjugate()))
            Rho_r = read @ Rhot @ read.T.conjugate()
            Rho_r = Rho_r / Rho_r.trace()
            Rho_atom = ptr(Rho_r, 2 * s + 1, (2*F+1) )
            Rhot = np.kron(Rho_atom, Rhos_ini)    
            # C_1z1z[i] = np.trace(Rho_atom @ Fz )
            C_1z1z[i] = np.trace(Rho_atom @ Fy@Fy)-np.trace(Rho_atom @ Fy)**2

        tt = np.arange(0, T, dt)
        plt.plot(tt, C_1z1z)
        # plt.plot(tt, C_1z2z)

        # plt.xlim(0, 1)
        # plt.ylim(0, 0.4)
    
    plt.xlabel('t (arb. units)', fontsize=12)
    plt.ylabel('$\langle F_x \\rangle$', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('imag\squeezing.png', dpi=600)

    # plt.figure()
    # plt.plot(t, C_1x2x)
    # plt.figure()
    # plt.plot(t, (np.array(C_1x2x)+np.array(C_1x1x)))

    plt.show()
