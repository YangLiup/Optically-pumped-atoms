# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月17日
"""
import sys
# sys.path.append(r"/Users/liyang/Documents/GitHub/Optically_polarized_atoms/my_functions")
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
from matplotlib import rc
import scienceplots
from tqdm import trange
# ax, ay, az, bx, by, bz, a1x, a2x, a1y, a2y, a1z, a2z, Fx, Fy, Fz

# ----------------------Squeezing----------------------------#
# N is the number of atoms, T is the squeezing time, F is the spin of atom, s is the spin of light and alpha is the coupling constant
N = 2
I = 3 / 2
T_sq = 3.5
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
omega_0=10
Ahf=1000
if N == 2:
    """
      角动量算符
    """
    a1x, a2x, a1y, a2y, a1z, a2z, b1x, b2x, b1y, b2y, b1z, b2z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(
        2, I)
    mathcal_F=a1z+a2z+b1z+b2z
    mathcal_S=(a1z+a2z-b1z-b2z)/4
    mathcal_F=mathcal_S
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

    Ps12 = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) - (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    Pt12 = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) + (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    Pe12 = Pt12 - Ps12
    
    """
    Zeeman effect
    """
    H_z = omega_0 * (a1x + a2x - b1x - b2x)  # 两个原子

    """
      Hyperfine interaction
    """
    # 等效法
    # hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    # hyperfine = np.kron(hyperfine, hyperfine)  # 两个原子
    # 第一性原理
    H_h = Ahf *( (a1x+b1x)@(a1x+b1x)+(a1y+b1y)@(a1y+b1y)+(a1z+b1z)@(a1z+b1z)+(a2x+b2x)@(a2x+b2x)+(a2y+b2y)@(a2y+b2y)+(a2z+b2z)@(a2z+b2z) -9) # 两个原子

if N == 3:
    a1x, a2x, a3x, a1y, a2y, a3y, a1z, a2z, a3z, b1x, b2x, b3x, b1y, b2y, b3y, b1z, b2z, b3z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(
        3, I)
    H = omega_0 * (a1x + a2x + a3x - b1x - b2x - b3x)  # 三个原子
    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    hyperfine = np.kron(hyperfine, np.kron(hyperfine, hyperfine))  # 三个原子
    mathcal_F=a1z+a2z+a3z+b1z+b2z+b3z
    mathcal_S=(a1z+a2z+a3z-b1z-b2z-b3z)/4



    Fx = a1x + a2x + a3x + b1x + b2x + b3x
    Fz = a1z + a2z + a3z + b1z + b2z + b3z
    Sx = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmax().full()))
    Sx = U.T.conjugate() @ Sx @ U
    Sy = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmay().full()))
    Sy = U.T.conjugate() @ Sy @ U
    Sz = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmaz().full()))
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
# ----------------------squeezing----------------------#
ini_Rho_atom, Rho_atomi = Generate_a_squeezed_state_by_QND(2, I, T_sq, s, alpha, dt)


# if N == 3:
#     Rho_atom3 = np.zeros(2 * (a + b + 1))
#     # --------------------------------Define the initial state-----------------------------------#
#     theta = np.pi / 2
#     phi = 0
#     a_theta = spin_Jx(a) * np.sin(theta) * np.cos(phi) + spin_Jy(a) * np.sin(theta) * np.sin(phi) + spin_Jz(a) * np.cos(
#         theta)
#     b_theta = spin_Jx(b) * np.sin(theta) * np.cos(phi) + spin_Jy(b) * np.sin(theta) * np.sin(phi) + spin_Jz(b) * np.cos(
#         theta)
#     qa, va = np.linalg.eig(np.array(a_theta.full()))
#     qb, vb = np.linalg.eig(np.array(b_theta.full()))
#     vs = block_diag(va, vb)
#     qs = np.hstack((qa, qb))
#     # # -----------------spin temperature state-----------------#
#     P = 0.9999999999999
#     beta = np.log((1 + P) / (1 - P))
#     for i in np.arange(0, 2 * (a + b + 1), 1):
#         Rho_atom3 =Rho_atom3 + np.exp(beta * qs[i]) * vs[:, [i]] * vs[:, [i]].T.conjugate()
#     Rho_atom3 = Rho_atom3 / np.trace(Rho_atom3)

#     # Rho_atom3 = np.eye(8)/8
#     Rho_atomi = np.kron(Rho_atom, Rho_atom3)
# ----------------------Evolution of the spin under magnetic field and hyperfine interaction----------------------#
T = 40
dt = 1e-4
n = round(T / dt)
te = np.arange(0, T, dt)
C_1 = [None] * n
C_2 = [None] * n
C_3 = [None] * n
C_4 = [None] * n

qz, vz = np.linalg.eig(H_z)
qh, vh = np.linalg.eig(H_h)
evolving_B = vz @ np.diag(np.exp(-1j * qz * dt)) @ np.linalg.inv(vz)
evolving_h = vh @ np.diag(np.exp(-1j * qh * dt)) @ np.linalg.inv(vh)

Rho_atom = Rho_atomi
for t in trange(0, n, 1):
    C_1[t] = np.trace(Rho_atom @ mathcal_F@ mathcal_F) - np.trace(Rho_atom @mathcal_F) ** 2
    Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    # Rho_atom = hyperfine * Rho_atom
    Rho_atom = evolving_h @ Rho_atom @ evolving_h.T.conjugate()
Rho_atom = Rho_atomi
for t in np.arange(0, n, 1):
    C_4[t] =np.trace(Rho_atom @ mathcal_F@ mathcal_F) - np.trace(Rho_atom @mathcal_F) ** 2
    # Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    hh=np.random.uniform()
    if hh<0.0009:
        r = np.random.uniform()
        if r  < 0.33:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe12
        # elif r < 0.66:
        #     phi = np.random.normal(np.pi / 2, 2)
        #     sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe13
        # else:
        #     phi = np.random.normal(np.pi / 2, 2)
        #     sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe23
            Rho_atom = sec @ Rho_atom @ sec.T.conjugate()
    # Rho_atom = hyperfine * Rho_atom
    Rho_atom = evolving_h @ Rho_atom @ evolving_h.T.conjugate()

Rho_atom = Rho_atomi
for t in trange(0, n, 1):
    C_2[t] = np.trace(Rho_atom @ mathcal_F@ mathcal_F) - np.trace(Rho_atom @mathcal_F) ** 2
    hh=np.random.uniform()
    if hh<0.0009:
        r = np.random.uniform()
        if r  < 0.33:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe12
        # elif r < 0.66:
        #     phi = np.random.normal(np.pi / 2, 2)
        #     sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe13
        # else:
        #     phi = np.random.normal(np.pi / 2, 2)
        #     sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe23
            Rho_atom = sec @ Rho_atom @ sec.T.conjugate()
    # C_1z2z[t] = np.trace(ini_Rho_atom @ a1z @ a2z)
    Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    # Rho_atom = hyperfine * Rho_atom
    Rho_atom = evolving_h @ Rho_atom @ evolving_h.T.conjugate()

# ----------------------Magnetic field----------------------#
omega_0 = 1.5
# H = omega_e * (ax-bx)              #一个原子
# H = omega_0 * (a1x + a2x - b1x - b2x)  # 两个原子
H = omega_0 * (a1x + a2x  - b1x - b2x)  # 三个原子
q, v = np.linalg.eig(H)
evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
Rho_atom = Rho_atomi  
for t in trange(0, n, 1):
    C_3[t] = np.trace(Rho_atom @ mathcal_F@ mathcal_F) - np.trace(Rho_atom @mathcal_F) ** 2
    # C_a1za1z[t] = np.trace(Rho_atom @ a1z @ a1z) - np.trace(Rho_atom @ a1z) * np.trace(Rho_atom @ a1z)
    # C_b1zb1z[t] = np.trace(Rho_atom @ b1z @ b1z) - np.trace(Rho_atom @ b1z) * np.trace(Rho_atom @ b1z)
    # C_a1za2z[t] = np.trace(Rho_atom @ a1z @ a2z) - np.trace(Rho_atom @ a1z) * np.trace(Rho_atom @ a2z)
    # C_a1zb2z[t] = np.trace(Rho_atom @ a1z @ b2z) - np.trace(Rho_atom @ a1z) * np.trace(Rho_atom @ b2z)
    # C_b1zb2z[t] = np.trace(Rho_atom @ b1z @ b2z) - np.trace(Rho_atom @ b1z) * np.trace(Rho_atom @ b2z)
    hh=np.random.uniform()
    if hh<0.0009:
        r = np.random.uniform()
        if r  < 0.33:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe12
        # elif r < 0.66:
        #     phi = np.random.normal(np.pi / 2, 2)
        #     sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe13
        # else:
        #     phi = np.random.normal(np.pi / 2, 2)
        #     sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe23
            Rho_atom = sec @ Rho_atom @ sec.T.conjugate()
    # C_1z2z[t] = np.trace(ini_Rho_atom @ a1z @ a2z)
    Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    # Rho_atom = hyperfine * Rho_atom
    Rho_atom = evolving_h @ Rho_atom @ evolving_h.T.conjugate()
C_1=np.array(C_1)
C_2=np.array(C_2)
C_3=np.array(C_3)
C_4=np.array(C_4)


CSS=np.array([1,0,0,0,0,0,0,0])
CSS3=np.kron(CSS,CSS)
Rho_CSS=np.outer(CSS3,CSS3)
mathcal_S=(a1x+a2x-b1x-b2x)/4
mathcal_F=(a1x+a2x+b1x+b2x)
mathcal_F=mathcal_S
Var=np.trace(Rho_CSS@mathcal_F@mathcal_F)-np.trace(Rho_CSS@mathcal_F)**2

from scipy.signal import butter, filtfilt

# 假设你已有的数据：
fs = round(1/dt)         # 采样率 (Hz)
cutoff = 3       # 截止频率 (Hz)
order = 5           # 滤波器阶数（4 是常用选择）

# 创建滤波器
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs                 # 奈奎斯特频率
    normal_cutoff = cutoff / nyq  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# 应用滤波器
def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 滤波
C_1prime = lowpass_filter(C_1, cutoff, fs, order)
C_2prime = lowpass_filter(C_2, cutoff, fs, order)
C_3prime = lowpass_filter(C_3, cutoff, fs, order)
C_4prime = lowpass_filter(C_4, cutoff, fs, order)

tt = np.arange(0, n, 1)*dt
with plt.style.context(['science']):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    p1, = ax1.plot(tt, C_1prime)
    p2, = ax1.plot(tt, C_2prime)
    p3, = ax1.plot(tt, C_3prime)
    p4, = ax1.plot(tt, C_4prime)
    p5, = ax1.plot(tt, np.ones(len(tt))*Var)
    ax1.legend([p1,p2,p3,p4,p5],
               ["$R_{\\text{se}}=0$ Hz,$\omega_0=10$ rad/s","$R_{\\text{se}}=30$ Hz, $\omega_0=10$ rad/s","$R_{\\text{se}}=30$ Hz, $\omega_0=1.5$ rad/s","$R_{\\text{se}}=30$ Hz, $\omega_0=0$ rad/s","CSS"],bbox_to_anchor=(0.8, -0.2),ncol=1)
    ax1.set_xlabel('$t$ (s)')
    ax1.set_ylabel('Var $( \mathcal S_{x})$')
    # plt.xlim(0, 10)
    # plt.ylim(0.1,0.55)
    plt.savefig('desqueezing.png', dpi=600)
