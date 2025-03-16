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
from ptr import ptr

N = 2
I = 3 / 2
T_sq = 2
a = round(I + 1 / 2)
b = round(I - 1 / 2)
s = 5
alpha = 0.2
dt = 0.02

U = alkali_atom_uncoupled_to_coupled(round(2 * I))

a1x, a2x, a1y, a2y, a1z, a2z, b1x, b2x, b1y, b2y, b1z, b2z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(N, I)

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
    H = alpha * np.kron(Fx, sx.full())

    XiF_ini= 1 / np.sqrt(2) * (
        np.kron([0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0])  + np.kron(
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0]))

    ini_Rho_atom = np.outer(XiF_ini, XiF_ini)
    Xis_ini = np.array(spin_coherent(s, 0, 0).full())
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

# Xi_ini= 1 / np.sqrt(2) * (
#         np.kron([0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0])  + np.kron(
#     [1, 0, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 0, 0, 0]))

# Xi_ini =np.kron([0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0])
ini_Rho_atom, Rho_atom = Generate_a_squeezed_state_by_QND(2, I, T_sq, s, alpha, dt)
dt = 0.01
T = 6
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
C_10 = [None] * round(T / dt)
C_11 = [None] * round(T / dt)
C_12 = [None] * round(T / dt)
C_13 = [None] * round(T / dt)
C_14 = [None] * round(T / dt)
C_15 = [None] * round(T / dt)
C_16 = [None] * round(T / dt)
C_17 = [None] * round(T / dt)
C_18 = [None] * round(T / dt)
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

Rhot2 = Rho_atom
for i in t:
    Rhot2 = evolving_B @ Rhot2 @ evolving_B.T.conjugate()  # Zeeman effect
    Rhot2 = hyperfine * Rhot2  # Hyperfine effect
    C_1[i] = np.trace(Rhot2 @ a1x @ a1x) - np.trace(Rhot2 @ a1x) * np.trace(Rhot2 @ a1x)
    C_2[i] = np.trace(Rhot2 @ a1y @ a1y) - np.trace(Rhot2 @ a1y) * np.trace(Rhot2 @ a1y)
    C_3[i] = np.trace(Rhot2 @ b1x @ b1x) - np.trace(Rhot2 @ b1x) * np.trace(Rhot2 @ b1x)
    C_4[i] = np.trace(Rhot2 @ b1y @ b1y) - np.trace(Rhot2 @ b1y) * np.trace(Rhot2 @ b1y)
    C_5[i] = np.trace(Rhot2 @ a1x @ a2x) - np.trace(Rhot2 @ a1x) * np.trace(Rhot2 @ a2x)
    C_6[i] = np.trace(Rhot2 @ a1y @ a2y) - np.trace(Rhot2 @ a1y) * np.trace(Rhot2 @ a2y)
    C_7[i] = np.trace(Rhot2 @ a1x @ b2x) - np.trace(Rhot2 @ a1x) * np.trace(Rhot2 @ b2x)
    C_8[i] = np.trace(Rhot2 @ a1y @ b2y) - np.trace(Rhot2 @ a1y) * np.trace(Rhot2 @ b2y)
    C_9[i] = np.trace(Rhot2 @ b1x @ b2x) - np.trace(Rhot2 @ b1x) * np.trace(Rhot2 @ b2x)
    C_10[i] = np.trace(Rhot2 @ b1y @ b2y) - np.trace(Rhot2 @ b1y) * np.trace(Rhot2 @ b2y)
    C_11[i] = np.trace(Rhot2 @ (a1x+a2x-b1x-b2x) @ (a1x+a2x-b1x-b2x) )  - np.trace(Rhot2 @ (a1x+a2x-b1x-b2x) ) * np.trace(Rhot2 @ (a1x+a2x-b1x-b2x) ) 
    C_12[i] = np.trace(Rhot2 @ (a1y+a2y-b1y-b2y)  @  (a1y+a2y-b1y-b2y))  - np.trace(Rhot2 @  (a1y+a2y-b1y-b2y)) * np.trace(Rhot2 @  (a1y+a2y-b1y-b2y)) 
t=t*dt

plt.style.use(['science'])
with plt.style.context(['science']):
    fig1 = plt.figure(figsize=(6.4,8))
    ax1 = fig1.add_subplot(3, 2, 1)
    p1, = ax1.plot(t, C_1)
    p2, = ax1.plot(t, C_2)
    C12=np.array(C_1)+np.array(C_2)
    p12, = ax1.plot(t,C12 )
    ax1.legend([p1, p2, p12], ["$<a_{1x} a_{1x}>$", "$<a_{1y} a_{1y}>$", "sum"],loc='upper center',ncol=2)

    ax2 = fig1.add_subplot(3, 2, 2)
    ax2.plot([], [])
    ax2.plot([], [])
    ax2.plot([], [])
    p3, = ax2.plot(t, C_3)
    p4, = ax2.plot(t, C_4)
    C34=np.array(C_3)+np.array(C_4)
    p34, = ax2.plot(t, C34)
    ax2.legend([p3, p4, p34], ["$<b_{1x} b_{1x}>$", "$<b_{1y} b_{1y}>$", "sum"],loc='upper center',ncol=2)

    ax3 = fig1.add_subplot(3, 2, 3)
    ax3.plot([], [])
    ax3.plot([], [])
    ax3.plot([], [])
    ax3.plot([], [])   
    ax3.plot([], [])
    ax3.plot([], [])
    p5, = ax3.plot(t, C_5)
    p6, = ax3.plot(t, C_6,color='indigo')
    C56=np.array(C_5)+np.array(C_6)
    p56,=ax3.plot(t, C56,color='darkgoldenrod')
    ax3.set_ylim([-0.2,0.7])
    ax3.legend([p5, p6, p56], ["$<a_{1x} a_{2x}>$", "$<a_{1y} a_{2y}>$", "sum"],loc='upper center',ncol=2)


    ax4 = fig1.add_subplot(3, 2, 4)
    ax4.plot([], [])
    ax4.plot([], [])
    ax4.plot([], [])
    ax4.plot([], []) 
    ax4.plot([], [])
    ax4.plot([], []) 
    p7, = ax4.plot(t, C_7,color='brown')
    p8, = ax4.plot(t, C_8,color='olive')
    C78=np.array(C_7)+np.array(C_8)
    p78, = ax4.plot(t,C78,color='salmon')
    ax4.set_ylim([-0.2,0.3])
    ax4.legend([p7,p8,p78], ["$<a_{1x} b_{2x}>$", "$<a_{1y} b_{2y}>$", "sum"],loc='upper center',ncol=2)


    ax5 = fig1.add_subplot(3, 2, 5)
    p9, = ax5.plot(t, C_9,color='black')
    p10, = ax5.plot(t, C_10,color='navy')
    C910=np.array(C_9)+np.array(C_10)
    p910,=ax5.plot(t, C910,color='darkolivegreen')
    ax5.set_ylim([-0.1,0.25])
    ax5.legend([p9,p10,p910], ["$<b_{1x} b_{2x}>$", "$<b_{1y} b_{2y}>$", "sum"],loc='upper center',ncol=2)



    ax6 = fig1.add_subplot(3, 2, 6)
    p11, = ax6.plot(t, C_11,color='blue')
    p12, = ax6.plot(t, C_12,color='darkgreen')
    C1112=np.array(C_11)+np.array(C_12)
    p1112, = ax6.plot(t, C1112,color='crimson')
    ax6.set_ylim([0,6.6])
    ax6.legend([p11,p12,p1112], ["$<\mathcal F_{x} \mathcal F_{x}>$", "$<\mathcal F_{y} \mathcal F_{y} >$", "sum"],loc='upper center',ncol=2)

    fig1.text(0.055, 0.5, ' Correlations', va='center', rotation='vertical',fontsize='12')
    fig1.text(0.5, 0.071, ' time (s)', va='center', rotation='horizontal',fontsize='12')

# ax1.tick_params(axis='x')
# ax1.tick_params(axis='y')
plt.savefig('Correlations_under_magneticfield.png', dpi=600)
plt.show()
