# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月20日
"""
import numpy as np
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import matplotlib.pyplot as plt
import scienceplots

N = 3
I = 3 / 2
a = round(I + 1 / 2)
b = round(I - 1 / 2)

U = alkali_atom_uncoupled_to_coupled(round(2 * I))

a1x, a2x, a3x, a1y, a2y, a3y, a1z, a2z, a3z, b1x, b2x, b3x, b1y, b2y, b3y, b1z, b2z, b3z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(
    3, I)

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
# ----------------------state--------------------#


xi = np.array([0, 1, 0, 0, 0, 0, 0, 0])
Xi_ini =  np.kron(np.kron(xi
                                   , xi), xi)
Rho_ini = np.outer(Xi_ini, Xi_ini)
Rhot = Rho_ini
dt = 0.01
T = 5
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
hyperfine = np.kron(np.kron(hyperfine, hyperfine), hyperfine)  # 两个原子

# ----------------------With magnetic field--------------------#
omega_0 = 10
# H = omega_0 * (S1z + S2z)  # 非投影定理
H = omega_0 * (a1z + a2z + a3z - b1z - b3z - b2z)  # 投影定理

q, v = np.linalg.eig(H)
evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)

for i in t:
    Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect

    Rhot = hyperfine * Rhot  # Hyperfine effect
    hh = np.random.uniform()
    if hh < 1:
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
        Rhot = sec @ Rhot @ sec.T.conjugate()  # spin exchange collision

    C_1[i] = np.trace(Rhot @ a1x @ a1x) - np.trace(Rhot @ a1x) * np.trace(Rhot @ a1x)
    C_3[i] = np.trace(Rhot @ b1x @ b1x) - np.trace(Rhot @ b1x) * np.trace(Rhot @ b1x)

    C_5[i] = np.trace(Rhot @ a1x @ a2x) - np.trace(Rhot @ a1x) * np.trace(Rhot @ a2x)
    C_6[i] = np.trace(Rhot @ a1x @ b2x) - np.trace(Rhot @ a1x) * np.trace(Rhot @ b2x)
    C_8[i] = np.trace(Rhot @ b1x @ b2x) - np.trace(Rhot @ b1x) * np.trace(Rhot @ b2x)

    C_9[i] = np.trace(Rhot @ Fx @ Fx) - np.trace(Rhot @ Fx) * np.trace(Rhot @ Fx)
Rhot = Rho_ini
for i in t:
    Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect

    Rhot = hyperfine * Rhot  # Hyperfine effect

    C_12[i] = np.trace(Rhot @ a1x @ a1x) - np.trace(Rhot @ a1x) * np.trace(Rhot @ a1x)
    C_32[i] = np.trace(Rhot @ b1x @ b1x) - np.trace(Rhot @ b1x) * np.trace(Rhot @ b1x)

    C_52[i] = np.trace(Rhot @ a1x @ a2x) - np.trace(Rhot @ a1x) * np.trace(Rhot @ a2x)
    C_62[i] = np.trace(Rhot @ a1x @ b2x) - np.trace(Rhot @ a1x) * np.trace(Rhot @ b2x)
    C_82[i] = np.trace(Rhot @ b1x @ b2x) - np.trace(Rhot @ b1x) * np.trace(Rhot @ b2x)

    C_92[i] = np.trace(Rhot @ Fx @ Fx) - np.trace(Rhot @ Fx) * np.trace(Rhot @ Fx)

t = t / 100
plt.style.use(['science'])
with plt.style.context(['science']):
    fig1 = plt.figure(dpi=600)
ax1 = fig1.add_subplot(1, 1, 1)
# p1, = ax1.plot(t, C_1, color='red')
# p5, = ax1.plot(t, C_5, color='olive')
# p6, = ax1.plot(t, C_6, color='dodgerblue')
# p8, = ax1.plot(t, C_8, color='darkgoldenrod')
# p3, = ax1.plot(t, C_3, color='pink')
p9, = ax1.plot(t, C_9)
ax1.plot([],[])
ax1.plot([],[])
ax1.plot([],[])
ax1.plot([],[])
ax1.plot([],[])
ax1.plot([],[])
# p12, = ax1.plot(t, C_12, color='red', linestyle='dashed')
# p52, = ax1.plot(t, C_52, color='olive', linestyle='dashed')
# p62, = ax1.plot(t, C_62, color='dodgerblue', linestyle='dashed')
# p82, = ax1.plot(t, C_82, color='darkgoldenrod', linestyle='dashed')
# p32, = ax1.plot(t, C_32, color='pink', linestyle='dashed')
p92, = ax1.plot(t, C_92, linestyle='dashed')
# ax1.legend([p1, p5, p3, p6, p8, p9],
#            ["$<a_{1x} a_{1x}>$", "$<a_{1x} a_{2x}>$", "$<b_{1x} b_{1x}>$", "$<a_{1x} b_{2x}>$", "$<b_{1x} b_{2x}>$",
#             "$<F_xF_x>$"]
#            , loc='upper right', prop={'size': 10})
ax1.legend([ p9]
           ["$<F_xF_x>$"]
           , loc='upper right', prop={'size': 9})
ax1.set_ylabel('Correlations', fontsize=10)
ax1.set_xlabel('SEC times', fontsize=10)
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='y', labelsize=10)
plt.savefig('spin exchange relaxation of correlations_increase1.png', dpi=1000)
