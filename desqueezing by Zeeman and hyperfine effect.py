# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月17日
"""
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from my_functions.Generate_a_squeezed_state_by_QND import Generate_a_squeezed_state_by_QND
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from scipy.linalg import *
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from sympy.physics.quantum.spin import JzKet, JxKet
from sympy.physics.quantum.represent import represent
from matplotlib import rc
import scienceplots

# ax, ay, az, bx, by, bz, a1x, a2x, a1y, a2y, a1z, a2z, Fx, Fy, Fz

# ----------------------Squeezing----------------------------#
# N is the number of atoms, T is the squeezing time, F is the spin of atom, s is the spin of light and alpha is the coupling constant
N = 3
I = 3 / 2
T_sq = 3
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
if N == 2:
    a1x, a2x, a1y, a2y, a1z, a2z, b1x, b2x, b1y, b2y, b1z, b2z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(
        2, I)
if N == 3:
    a1x, a2x, a3x, a1y, a2y, a3y, a1z, a2z, a3z, b1x, b2x, b3x, b1y, b2y, b3y, b1z, b2z, b3z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(
        3, I)
    Fx = a1x + a2x + a3x + b1x + b2x + b3x
    Fz = a1z + a2z + a3z + b1z + b2z + b3z
# ----------------------squeezing----------------------#

ini_Rho_atom, Rho_atom = Generate_a_squeezed_state_by_QND(2, I, T_sq, s, alpha, dt)

if N == 3:
    q, v = np.linalg.eig(ax)
    Rho_atom3 = np.outer(np.vstack((v[:, [1]], np.array(zero_ket(2 * b + 1)))),
                         np.vstack((v[:, [1]], np.array(zero_ket(2 * b + 1)))))
    Rho_atomi = np.kron(Rho_atom, Rho_atom3)
# ----------------------Evolution of the spin under magnetic field and hyperfine interaction----------------------#

# ----------------------electron spin----------------------#
if N == 2:
    Sx = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmax()))
    Sx = U.T.conjugate() @ Sx @ U
    Sy = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmay()))
    Sy = U.T.conjugate() @ Sy @ U
    Sz = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmaz()))
    Sz = U.T.conjugate() @ Sz @ U

    S1x = np.kron(Sx, np.eye(2 * (a + b + 1)))
    S2x = np.kron(np.eye(2 * (a + b + 1)), Sx)
    S1y = np.kron(Sy, np.eye(2 * (a + b + 1)))
    S2y = np.kron(np.eye(2 * (a + b + 1)), Sy)
    S1z = np.kron(Sz, np.eye(2 * (a + b + 1)))
    S2z = np.kron(np.eye(2 * (a + b + 1)), Sz)

    Ps12 = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) - (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    Pt12 = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) + (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    Pe12 = Pt12 - Ps12
if N == 3:
    Sx = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmax()))
    Sx = U.T.conjugate() @ Sx @ U
    Sy = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmay()))
    Sy = U.T.conjugate() @ Sy @ U
    Sz = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmaz()))
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

T = 20
dt = 0.01
n = round(T / dt)
te = np.arange(0, T, dt)
C_1 = [None] * n
C_2 = [None] * n
C_a1za1z = [None] * n
C_b1zb1z = [None] * n
C_a1za2z = [None] * n
C_a1zb2z = [None] * n
C_b1zb2z = [None] * n
# ----------------------Magnetic field----------------------#
omega_0 = 10
# H = omega_e * (ax-bx)              #一个原子
# H = omega_0 * (a1x + a2x - b1x - b2x)  # 两个原子
H = omega_0 * (a1x + a2x + a3x - b1x - b2x - b3x)  # 三个原子

q, v = np.linalg.eig(H)
evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)

# ----------------------Hyperfine interaction--------------------#
Rho_atom = Rho_atomi
hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
if N == 2:
    hyperfine = np.kron(hyperfine, hyperfine)  # 两个原子
if N == 3:
    hyperfine = np.kron(hyperfine, np.kron(hyperfine, hyperfine))  # 三个原子

for t in np.arange(0, n, 1):
    C_1[t] = np.trace(Rho_atom @ Fz @ Fz) - np.trace(Rho_atom @ Fz) ** 2
    C_a1za1z[t] = np.trace(Rho_atom @ a1z @ a1z) - np.trace(Rho_atom @ a1z) * np.trace(Rho_atom @ a2z)
    C_b1zb1z[t] = np.trace(Rho_atom @ b1z @ b1z) - np.trace(Rho_atom @ b1z) * np.trace(Rho_atom @ b2z)
    C_a1za2z[t] = np.trace(Rho_atom @ a1z @ a2z) - np.trace(Rho_atom @ a1z) * np.trace(Rho_atom @ a2z)
    C_a1zb2z[t] = np.trace(Rho_atom @ a1z @ b2z) - np.trace(Rho_atom @ a1z) * np.trace(Rho_atom @ b2z)
    C_b1zb2z[t] = np.trace(Rho_atom @ b1z @ b2z) - np.trace(Rho_atom @ b1z) * np.trace(Rho_atom @ b2z)
    r = np.random.uniform()
    if r - 0.3 < 0.0001:
        phi = np.random.normal(np.pi / 2, 2)
        sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe12
    elif r - 0.6 < 0.0001:
        phi = np.random.normal(np.pi / 2, 2)
        sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe13
    else:
        phi = np.random.normal(np.pi / 2, 2)
        sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe23

    Rho_atom = sec @ Rho_atom @ sec.T.conjugate()
    # C_1z2z[t] = np.trace(ini_Rho_atom @ a1z @ a2z)
    Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    Rho_atom = hyperfine * Rho_atom
Rho_atom = Rho_atomi
for t in np.arange(0, n, 1):
    C_2[t] = np.trace(Rho_atom @ Fz @ Fz) - np.trace(Rho_atom @ Fz) ** 2
    # r = np.random.uniform()
    # if r - 0.3 < 0.0001:
    #     phi = np.random.normal(np.pi / 2, 2)
    #     sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe12
    # elif r - 0.6 < 0.0001:
    #     phi = np.random.normal(np.pi / 2, 2)
    #     sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe13
    # else:
    #     phi = np.random.normal(np.pi / 2, 2)
    #     sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe23
    #
    # Rho_atom = sec @ Rho_atom @ sec.T.conjugate()
    # C_1z2z[t] = np.trace(ini_Rho_atom @ a1z @ a2z)
    Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    Rho_atom = hyperfine * Rho_atom

tt = np.arange(0, n, 1)
with plt.style.context(['science']):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    p1, = ax1.plot(tt, C_1, color='red')
    p2, = ax1.plot(tt, C_2, color='purple')
    ax1.legend([p1, p2],
               ["With SEC", "Without SEC"]
               , loc='upper right', prop={'size': 8})
    ax1.set_xlabel('SEC times', fontsize=10)
    ax1.set_ylabel('Variance', fontsize=10)
    plt.xlim(0, 250)

    plt.savefig('graph1.png', dpi=600)

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    p3, = ax2.plot(tt, C_a1za1z, color='brown')
    p5, = ax2.plot(tt, C_a1za2z, color='olive')
    ax2.legend([p3, p5],
               ["Cov$(a_{1z}a_{1z})$", "Cov$(a_{1z}a_{2z})$"]
               , loc='upper right', prop={'size': 8})
    ax2.set_ylabel('Correlations', fontsize=10)
    ax2.set_xlabel('SEC times', fontsize=10)
    plt.xlim(0, 250)
    plt.savefig('graph2.png', dpi=600)


    fig = plt.figure()
    ax3 = fig.add_subplot(1, 1, 1)
    p4, = ax3.plot(tt, C_b1zb1z)
    p6, = ax3.plot(tt, C_a1zb2z)
    p7, = ax3.plot(tt, C_b1zb2z)
    ax3.set_xlabel('SEC times', fontsize=10)
    ax3.set_ylabel('Correlations', fontsize=10)
    ax3.legend([p4, p6, p7],
               ["Cov$(b_{1z}b_{1z})$", "Cov$(a_{1z}b_{2z})$",
                "Cov$(b_{1z}b_{2z})$", ]
               , loc='upper right', prop={'size': 8})
    plt.xlim(0, 250)

    plt.savefig('graph3.png', dpi=600)

    # plt.ylim(-0.5, 5.2)
