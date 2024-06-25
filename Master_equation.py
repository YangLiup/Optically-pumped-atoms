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
I = 3 / 2
a = round(I + 1 / 2)
b = round(I - 1 / 2)

# --------------------------------Generate the angular momentum operators-----------------------------------#
U = alkali_atom_uncoupled_to_coupled(round(2 * I))
ax, ay, az, bx, by, bz = spin_operators_of_2or1_alkali_metal_atoms(1, I)
Sx = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmax()))
Sx = U.T.conjugate() @ Sx @ U
Sy = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmay()))
Sy = U.T.conjugate() @ Sy @ U
Sz = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmaz()))
Sz = U.T.conjugate() @ Sz @ U

# --------------------------------Characterize interactions envolved-----------------------------------#
Rse = 1
omega_0 = 0.05
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
qa, va = np.linalg.eig(np.array(a_theta))
qb, vb = np.linalg.eig(np.array(b_theta))
v = block_diag(va, vb)
q = np.hstack((qa, qb))
Rho_ini = np.zeros(2 * (a + b + 1))

# # -----------------spin temperature state-----------------#
P = 0.95
beta = np.log((1 + P) / (1 - P))
for i in np.arange(0, 2 * (a + b + 1), 1):
    Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
Rho_ini = Rho_ini / np.trace(Rho_ini)

# -----------------eigenstates-----------------#

# Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

# --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
Rhot = Rho_ini
dt = 0.01
T = 5000
t = np.arange(0, T, dt)
hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
MSx = np.zeros(round(T / dt))
MSz = np.zeros(round(T / dt))
V = np.zeros(round(T / dt))

H = omega_0 * (az - bz)  # 投影定理
q, v = np.linalg.eig(H)
evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
for n in np.arange(0, round(T / dt), 1):
    # -----------------Evolution-----------------#
    if n==round(T / dt)/2:
        Rop = 0.0
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
    MSz[n] = mSz
    MSx[n] = mSx
    # Vx = np.trace(Rhot @ ((ax + bx) @ (ax + bx) + (ay + by) @ (ay + by) + (az + bz) @ (az + bz)))
    # V[n] = Vx

# ---------------------------Bloch Equation-----------------------------------
Rop = 0.
transverse = np.zeros(round(T / dt))
longitude = np.zeros(round(T / dt))
Px = P * np.sin(theta)
Pz = P * np.cos(theta)
Py = 0
for n in np.arange(0, round(T / dt), 1):
    if n==round(T / dt)/2:
        Rop = 0.0
    transverse[n] = Px
    longitude[n] = Pz
    #xiao
    qnm = 2 * (3 + P ** 2) / (1 + P ** 2)
    Qnm = 2 * (3 + P ** 4) / ((1 + P ** 2) ** 2)
    Gamma = 4 * (-4 + qnm) * (4 + qnm) * omega_0 ** 2 / 3 / qnm ** 2 * qnm / Qnm

    #Mr Zhao
    # qnm = 2 * (3 + P ** 2) / (1 + P ** 2)
    # Qnm = 2 * (3 + P ** 4) / ((1 + P ** 2) ** 2)
    # Gamma = 4 * (-4 + qnm) * (4 + qnm)**2 * omega_0 ** 2 / qnm ** 3 /5* qnm / Qnm

    T2 = (1 - ((qnm - Qnm) / qnm) * Pz ** 2 / P ** 2) * Gamma
    T1 = T2-Gamma/qnm*Qnm

    Pz = Pz + (-1 / Qnm * (Rsd + Rop) * Pz + Rop *
            ((1 / Qnm - 1 / qnm) * (Px *sx+Pz*sz)*(Pz-Px *sx*sz-Pz*sz*sz) / P ** 2)+Rop *sz*
            ((1 / Qnm  * (Px *sx+Pz*sz)**2/ P ** 2)+(1 / qnm  * (1-(Px *sx+Pz*sz)**2/ P ** 2)))
            - Pz * T1) * dt
    Px = Px + (-1 / Qnm * (Rsd + Rop) * Px +Rop *
            ((1 / Qnm - 1 / qnm) * (Px *sx+Pz*sz)*(Px-Px *sx*sx-Pz*sz*sx) / P ** 2)+Rop *sx*
            ((1 / Qnm  * (Px *sx+Pz*sz)**2/ P ** 2)+(1 / qnm  * (1-(Px *sx+Pz*sz)**2/ P ** 2)))
            - 1 / qnm * omega_0 * 4 * Py - Px * T2) * dt
    Py = Py + (-1 / Qnm * (Rsd + Rop) * Py + Rop * (
            1 / Qnm - 1 / qnm) * (Px*sx+Pz*sz) / P ** 2 * Py + 1 / qnm * omega_0 * 4 * Px - Py * T2) * dt
    P = np.sqrt(Px ** 2 + Py ** 2 + Pz ** 2)

plt.style.use(['science'])
with plt.style.context(['science']):
    plt.figure()
    p1, = plt.plot(t, MSx * 2)
    p2, = plt.plot(t, MSz * 2)
    p3, = plt.plot(t, transverse)
    p4, = plt.plot(t, longitude)
    # plt.xlim(0, 200)
    # plt.ylim(0, 18)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.legend([p1, p3, p2, p4], ["$P_x^{\mathrm{DM}}$", "$P_x^{\mathrm{NB}}$", "$P_z^{\mathrm{DM}}$", "$P_z^{\mathrm{NB}}$"], loc='upper right',
    #            prop={'size': 10})

    plt.xlabel('Time $(1/R_{se})$', fontsize=12)
    plt.ylabel('Polarization', fontsize=12)
    plt.savefig('Evolution3.png', dpi=600)
plt.show()