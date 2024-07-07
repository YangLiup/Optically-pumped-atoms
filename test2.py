# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月24日
"""
import numpy as np
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from sympy.physics.quantum.spin import Rotation
from sympy import pi
from scipy.linalg import *
import scienceplots

# --------------------------------Properties of the alkali metal atom-----------------------------------#
I=3/2
Rse=1
omega_0=0.05
T=4000
N=1
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
x=np.array([])
# --------------------------------Characterize interactions envolved-----------------------------------#
dt1=0.01
dt2=0.05
dt3=0.1
for dt in [dt1, dt2, dt3]:
    PP=np.zeros(round(T/dt))
    Rop = 0.
    Rsd = 0.
    sx=np.sqrt(1)/(2)
    sz=np.sqrt(1)/(2)
    # --------------------------------Define the initial state-----------------------------------#
    theta = np.pi / 2
    phi = 0
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
    P = 0.99
    beta = np.log((1 + P) / (1 - P))
    for i in np.arange(0, 2 * (a + b + 1), 1):
        Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
    Rho_ini = Rho_ini / np.trace(Rho_ini)

    # -----------------eigenstates-----------------#

    # Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

    # --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
    Rhot = Rho_ini
    t = np.arange(0, T, dt)
    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子

    H = omega_0 * (az - bz)  # 投影定理
    q, v = np.linalg.eig(H)
    evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
    for n in np.arange(0, round(T / dt), 1):
        # -----------------Evolution-----------------#
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
        OP = Rop * (2 * alpha @ (sx*Sx+sz*Sz) - AS)
        Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect
        Rhot = Rse * (alpha + 4 * alpha @ mSS - Rhot) * dt + (
                ER + OP) * dt + Rhot
        Rhot = hyperfine * Rhot
        # -----------------Observables-----------------#
        PP[n] = np.sqrt(mSx**2+mSy**2)*2
        # V[n] = Vx
    x=np.hstack((x,PP))
plt.style.use(['science'])
with plt.style.context(['science']):
    plt.figure()
    p1=plt.plot(np.arange(0,T,dt1),x[0:round(T/dt1):1])
    p2=plt.plot(np.arange(0,T,dt2),x[round(T/dt1):round(T/dt1)+round(T/dt2):1])
    p3=plt.plot(np.arange(0,T,dt3),x[round(T/dt1)+round(T/dt2):round(T/dt1)+round(T/dt2)+round(T/dt3):1])
    # plt.xlim(0, 200)
    # plt.ylim(0, 18)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend([p1, p3, p2], ["dt1", "dt2", "dt3"], loc='upper right',
               prop={'size': 10})

    plt.xlabel('Time $(1/R_{se})$', fontsize=12)
    plt.ylabel('Polarization', fontsize=12)
    plt.savefig('imag/Evolution1.png', dpi=600)
    plt.show()