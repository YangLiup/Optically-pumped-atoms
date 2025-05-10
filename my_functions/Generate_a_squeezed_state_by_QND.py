# -*- coding:utf-8 -*-
"""
作者：Li Yang
日期：2023年12月16日
"""
# this function is used to Generate_a_squeezed_state_by_QND_Takahashi#
import sys
sys.path.append(r"D:\python\pythonProject\Optically_pumped_atoms\my_functions")
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from my_functions.ptr import ptr
from scipy.linalg import *
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms


def Generate_a_squeezed_state_by_QND(N, I, T, s, alpha, dt):
    # ----------------------Parameters that can be modify----------------#
    # T is the squeezing time
    # F is the spin of the atom
    # s is the spin of light
    # alpha is coupling constant
    # N is the number under squeezing
    # some necessary operators and states
    a = round(I + 1 / 2)
    b = round(I - 1 / 2)
    if N == 1:
        ax, ay, az, bx, by, bz, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(N, I)
    if N == 2:
        a1x, a2x, a1y, a2y, a1z, a2z, b1x, b2x, b1y, b2y, b1z, b2z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(N, I)

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
    # evolving
    for t in np.arange(0, T, dt):
        Rhot = dt * (H @ Rhot - Rhot @ H) / 1j + Rhot

    # measurement
    if N == 1:
        read = np.array(tensor(qeye(2 * (a + b + 1)), vy[5] * vy[5].dag()).full())
    if N == 2:
        read = np.array(
            tensor(tensor(qeye(2 * (a + b + 1)), qeye(2 * (a + b + 1)), vy[5] * vy[5].dag())).full())
    Rho_r = read @ Rhot @ read.T.conjugate()
    Rho_r = Rho_r / Rho_r.trace()
    Rho_atom = ptr(Rho_r, 2 * s + 1, (2 * (a + b + 1)) ** N)

    return ini_Rho_atom, Rho_atom
