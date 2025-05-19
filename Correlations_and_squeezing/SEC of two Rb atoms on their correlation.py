# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月20日
"""
import sys
sys.path.append(r"/Users/liyang/Documents/GitHub/Optically_polarized_atoms/my_functions")
import numpy as np
import matplotlib.pyplot as plt
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import matplotlib.pyplot as plt
import scienceplots
from scipy import stats


N = 2
I = 3 / 2
a = round(I + 1 / 2)
b = round(I - 1 / 2)

U = alkali_atom_uncoupled_to_coupled(round(2 * I))

a1x, a2x, a1y, a2y, a1z, a2z, b1x, b2x, b1y, b2y, b1z, b2z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(N, I)

corrxx = Fx @ Fx
corrxy = Fx @ Fy
corrxz = Fx @ Fz

corryx = Fy @ Fx
corryy = Fy @ Fy
corryz = Fy @ Fz

corrzx = Fz @ Fx
corrzy = Fz @ Fy
corrzz = Fz @ Fz

# ----------------------electron spin----------------------#
Sx = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmax().full()))
Sx = U.T.conjugate() @ Sx @ U
Sy = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmay().full()))
Sy = U.T.conjugate() @ Sy @ U
Sz = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmaz().full()))
Sz = U.T.conjugate() @ Sz @ U

S1x = np.kron(Sx, np.eye(2 * (a + b + 1)))
S2x = np.kron(np.eye(2 * (a + b + 1)), Sx)
S1y = np.kron(Sy, np.eye(2 * (a + b + 1)))
S2y = np.kron(np.eye(2 * (a + b + 1)), Sy)
S1z = np.kron(Sz, np.eye(2 * (a + b + 1)))
S2z = np.kron(np.eye(2 * (a + b + 1)), Sz)
Ps = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** 2)) - (S1x @ S2x + S1y @ S2y + S1z @ S2z)
Pt = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** 2)) + (S1x @ S2x + S1y @ S2y + S1z @ S2z)
Pe = Pt - Ps
# ----------------------state--------------------#

Xi_ini = 1 / np.sqrt(2) * (
        np.kron([0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]) + np.kron([1, 0, 0, 0, 0, 0, 0, 0],
                                                                              [1, 0, 0, 0, 0, 0, 0, 0]))
Rho_ini = np.outer(Xi_ini, Xi_ini)
Rhot = Rho_ini
dt = 0.01
T = 20
t = np.arange(0, round(T / dt), 1)
# ----------------------correlations--------------------#

C_1 = [None] * round(T / dt)
C_2 = [None] * round(T / dt)
C_3 = [None] * round(T / dt)
C_4 = [None] * round(T / dt)
C_5 = [None] * round(T / dt)
C_6 = [None] * round(T / dt)
C_7 = [None] * round(T / dt)
# C_8 = [None] * round(T / dt)
# ----------------------Hyperfine interaction--------------------#
hyperfine = block_diag(np.ones((5, 5)), np.ones((3, 3)))  # 一个原子
hyperfine = np.kron(hyperfine, hyperfine)  # 两个原子

# ----------------------With magnetic field--------------------#
omega_0 = 0
# H = omega_0 * (S1z + S2z)  # 非投影定理
H = omega_0 * (a1z + a2z - b1z - b2z)  # 投影定理


q, v = np.linalg.eig(H)
evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)

for i in t:
    # if i > int(T / 2 / dt):
    # Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect

    # Rhot = hyperfine * Rhot  # Hyperfine effect
    if i % 100 == 0:
        phi = stats.cauchy.rvs(loc=0, scale=10, size=1) 
        sec = np.cos(phi) * np.eye(round((2 * (2 * I + 1)) ** 2)) - 1j * np.sin(phi) * Pe
        Rhot = sec @ Rhot @ sec.T.conjugate()  # spin exchange collision

    C_1[i] = np.trace(Rhot @ a1x @ a1x)
    C_2[i] = np.trace(Rhot @ b1x @ b1x)
    C_3[i] = np.trace(Rhot @ a1x @ a2x)
    C_4[i] = np.trace(Rhot @ b1x @ b2x)
    C_5[i] = np.trace(Rhot @ a1x @ b2x)
    C_6[i] = np.trace(Rhot @ Fx @ Fx )
#   C_6[i] = np.trace(Rhot @ b1x @ b2x)
#   C_7[i] = np.trace(Rhot @ Fx @ Fx)

plt.style.use(['science'])
with plt.style.context(['science']):
    fig = plt.figure()
    p1, = plt.plot(t*dt, C_1)
    p2, = plt.plot(t*dt, C_2)
    p3, = plt.plot(t*dt, C_3)
    p4, = plt.plot(t*dt, C_4)
    p5, = plt.plot(t*dt, C_5)
    p6, = plt.plot(t*dt, C_6)

    # p5, = plt.plot(t * dt, C_5)
    # p6, = plt.plot(t * dt, C_6)
    # p7, = plt.plot(t * dt, C_7)
    # p5, = plt.plot(t * dt, C_5)
    plt.legend([p1, p2, p3, p4, p5,p6],
               ["$\langle f^a_{1x} f^a_{1x} \\rangle$", "$\langle f^b_{1x} f^b_{1x} \\rangle$", "$\langle f^a_{1x} f^a_{2x} \\rangle$", "$\langle f^b_{1x} f^b_{2x} \\rangle$", "$\langle f^a_{1x} f^b_{2x}\\rangle$", "$\langle \mathcal F_{x} \mathcal F_{x} \\rangle$"]
               ,ncol = 2, loc='center right',bbox_to_anchor=(0.95, 0.7))

    plt.xlabel('Spin-exchange collision number')
    plt.ylabel('Correlations')

    # plt.ylim(-0.5, 5.2)
    plt.savefig('spin exchange.png', dpi=600)
    plt.show()
