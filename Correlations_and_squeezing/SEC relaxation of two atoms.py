# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月20日
"""
import sys
sys.path.append(r"D:\python\pythonProject\Optically_pumped_atoms\my_functions")

import numpy as np
import matplotlib.pyplot as plt
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from  alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
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
# Xi_ini = 1 / np.sqrt(3) * (
#         np.kron([0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]) + np.kron([0, 0, 1, 0, 0, 0, 0, 0],
#                                                                               [0, 0, 0, 0, 0, 0, 1, 0]) + np.kron(
#     [0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1]))
Xi_ini= 1 / np.sqrt(2  ) * (
        np.kron([0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0])  + np.kron(
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0]))

# Xi_ini =np.kron([0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0])
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
C_8 = [None] * round(T / dt)
C_9 = [None] * round(T / dt)
C_12 = [None] * round(T / dt)
C_22 = [None] * round(T / dt)
C_32 = [None] * round(T / dt)
C_42 = [None] * round(T / dt)
C_52 = [None] * round(T / dt)
C_62 = [None] * round(T / dt)
C_72 = [None] * round(T / dt)
C_82 = [None] * round(T / dt)
C_92 = [None] * round(T / dt)
# C_8 = [None] * round(T / dt)
# ----------------------Hyperfine interaction--------------------#
hyperfine = block_diag(np.ones((5, 5)), np.ones((3, 3)))  # 一个原子
hyperfine = np.kron(hyperfine, hyperfine) # 两个原子

# ----------------------With magnetic field--------------------#
omega_0 = 1
# H = omega_0 * (S1z + S2z)  # 非投影定理
H = omega_0 * (a1z + a2z  - b1z - b2z)  # 投影定理

q, v = np.linalg.eig(H)
evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)

for i in t:
    Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect

    Rhot = hyperfine * Rhot  # Hyperfine effect
    hh = np.random.uniform()
    if hh < 0.01:
        phi = stats.cauchy.rvs(loc=0, scale=10, size=1) 
        sec = np.cos(phi) * np.eye(round((2 * (2 * I + 1)) ** 2)) - 1j * np.sin(phi) * Pe
        Rhot = sec @ Rhot @ sec.T.conjugate()  # spin exchange collision


    C_1[i] = np.trace(Rhot @ a1x @ a1x) - np.trace(Rhot @ a1x) * np.trace(Rhot @ a1x)
    C_3[i] = np.trace(Rhot @ b1x @ b1x) - np.trace(Rhot @ b1x) * np.trace(Rhot @ b1x)

    C_5[i] = np.trace(Rhot @ a1x @ a2x) - np.trace(Rhot @ a1x) * np.trace(Rhot @ a2x)
    C_6[i] = np.trace(Rhot @ a1x @ b2x) - np.trace(Rhot @ a1x) * np.trace(Rhot @ b2x)
    C_8[i] = np.trace(Rhot @ b1x @ b2x) - np.trace(Rhot @ b1x) * np.trace(Rhot @ b2x)

    C_9[i] = np.trace(Rhot @ (a1x+a2x-b1x-b2x) @ (a1x+a2x-b1x-b2x)) - np.trace(Rhot @ (a1x+a2x-b1x-b2x)) * np.trace(Rhot @ (a1x+a2x-b1x-b2x))

Rhot2 = Rho_ini
for i in t:
    Rhot2 = evolving_B @ Rhot2 @ evolving_B.T.conjugate()  # Zeeman effect

    Rhot2 = hyperfine * Rhot2  # Hyperfine effect

    C_12[i] = np.trace(Rhot2 @ a1x @ a1x) - np.trace(Rhot2 @ a1x) * np.trace(Rhot2 @ a1x)
    C_32[i] = np.trace(Rhot2 @ b1x @ b1x) - np.trace(Rhot2 @ b1x) * np.trace(Rhot2 @ b1x)

    C_52[i] = np.trace(Rhot2 @ a1x @ a2x) - np.trace(Rhot2 @ a1x) * np.trace(Rhot2 @ a2x)
    C_62[i] = np.trace(Rhot2 @ a1x @ b2x) - np.trace(Rhot2 @ a1x) * np.trace(Rhot2 @ b2x)
    C_82[i] = np.trace(Rhot2 @ b1x @ b2x) - np.trace(Rhot2 @ b1x) * np.trace(Rhot2 @ b2x)

    C_92[i] = np.trace(Rhot2 @ (a1x+a2x-b1x-b2x) @ (a1x+a2x-b1x-b2x)) - np.trace(Rhot2 @ (a1x+a2x-b1x-b2x)) * np.trace(Rhot2 @ (a1x+a2x-b1x-b2x))

t = t 

plt.style.use(['science'])
with plt.style.context(['science']):
    fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
p1, = ax1.plot(t, C_1)
p5, = ax1.plot(t, C_5)
p6, = ax1.plot(t, C_6)
p8, = ax1.plot(t, C_8)
p3, = ax1.plot(t, C_3)
p9, = ax1.plot(t, C_9)
ax1.plot([], [])
# ax1.plot([], [])
p12, = ax1.plot(t, C_12, linestyle='dashed')
p52, = ax1.plot(t, C_52, linestyle='dashed')
p62, = ax1.plot(t, C_62, linestyle='dashed')
p82, = ax1.plot(t, C_82, linestyle='dashed')
p32, = ax1.plot(t, C_32, linestyle='dashed')
p92, = ax1.plot(t, C_92, linestyle='dashed')
ax1.legend([p1, p5, p3, p6, p8, p9],
           ["$<a_{1x} a_{1x}>$", "$<a_{1x} a_{2x}>$", "$<b_{1x} b_{1x}>$", "$<a_{1x} b_{2x}>$", "$<b_{1x} b_{2x}>$",
            "$<\mathcal F_x \mathcal F_x>$"]
           , bbox_to_anchor=(1, 1),ncol=1)
ax1.set_ylabel('Correlations')
ax1.set_xlabel('time (T$_{\mathrm{se}}$)')
ax1.tick_params(axis='x')
ax1.tick_params(axis='y')
plt.savefig('spin exchange relaxation of correlations_increase.png', dpi=600)
plt.show()
