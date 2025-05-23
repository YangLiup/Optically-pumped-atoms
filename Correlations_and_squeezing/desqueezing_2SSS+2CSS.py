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
N = 4
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
# ----------------------Hyperfine interaction--------------------#
hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子

if N == 2:
    a1x, a2x, a1y, a2y, a1z, a2z, b1x, b2x, b1y, b2y, b1z, b2z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(
        2, I)
    H = omega_0 * (a1x + a2x - b1x - b2x)  # 两个原子
    hyperfine = np.kron(hyperfine, hyperfine)  # 两个原子
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
if N == 3:
    a1x, a2x, a3x, a1y, a2y, a3y, a1z, a2z, a3z, b1x, b2x, b3x, b1y, b2y, b3y, b1z, b2z, b3z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(
        3, I)
    H = omega_0 * (a1x + a2x + a3x - b1x - b2x - b3x)  # 三个原子
    hyperfine = np.kron(hyperfine, np.kron(hyperfine, hyperfine))  # 三个原子

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
if N == 4:
    a1x, a2x, a3x, a4x, a1y, a2y, a3y, a4y, a1z, a2z, a3z, a4z,b1x, b2x, b3x, b4x,b1y, b2y, b3y,b4y, b1z, b2z, b3z,b4z, Fx, Fy, Fz = spin_operators_of_2or1_alkali_metal_atoms(4, I)
    H = omega_0 * (a1x + a2x + a3x+a4x - b1x - b2x - b3x-b4x)  # 三个原子
    hyperfine = np.kron(np.kron(hyperfine, np.kron(hyperfine, hyperfine)),hyperfine)  # 三个原子

    Sx = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmax().full()))
    Sx = U.T.conjugate() @ Sx @ U
    Sy = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmay().full()))
    Sy = U.T.conjugate() @ Sy @ U
    Sz = np.kron(np.eye(round(2 * I + 1)), np.array(1 / 2 * sigmaz().full()))
    Sz = U.T.conjugate() @ Sz @ U

    S1x = np.kron(np.kron(np.kron(Sx, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    S2x = np.kron(np.kron(np.kron(np.eye(2 * (a + b + 1)), Sx), np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    S3x = np.kron(np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), Sx), np.eye(2 * (a + b + 1)))
    S4x = np.kron(np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))),np.eye(2 * (a + b + 1)) ), Sx)

    S1y = np.kron(np.kron(np.kron(Sy, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    S2y = np.kron(np.kron(np.kron(np.eye(2 * (a + b + 1)), Sy), np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    S3y = np.kron(np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), Sy), np.eye(2 * (a + b + 1)))
    S4y = np.kron(np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))),np.eye(2 * (a + b + 1)) ), Sy)

    S1z = np.kron(np.kron(np.kron(Sz, np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    S2z = np.kron(np.kron(np.kron(np.eye(2 * (a + b + 1)), Sz), np.eye(2 * (a + b + 1))), np.eye(2 * (a + b + 1)))
    S3z = np.kron(np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))), Sz), np.eye(2 * (a + b + 1)))
    S4z = np.kron(np.kron(np.kron(np.eye(2 * (a + b + 1)), np.eye(2 * (a + b + 1))),np.eye(2 * (a + b + 1)) ), Sz)

    Ps13 = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) - (S1x @ S3x + S1y @ S3y + S1z @ S3z)
    Pt13 = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) + (S1x @ S3x + S1y @ S3y + S1z @ S3z)
    Pe13 = Pt13 - Ps13

    Ps12 = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) - (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    Pt12 = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) + (S1x @ S2x + S1y @ S2y + S1z @ S2z)
    Pe12 = Pt12 - Ps12

    Ps14 = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) - (S1x @ S4x + S1y @ S4y + S1z @ S4z)
    Pt14 = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) + (S1x @ S4x + S1y @ S4y + S1z @ S4z)
    Pe14 = Pt14 - Ps14

    Ps23 = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) - (S2x @ S3x + S2y @ S3y + S2z @ S3z)
    Pt23 = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) + (S2x @ S3x + S2y @ S3y + S2z @ S3z)
    Pe23 = Pt23 - Ps23

    Ps24 = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) - (S2x @ S4x + S2y @ S4y + S2z @ S4z)
    Pt24 = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) + (S2x @ S4x + S2y @ S4y + S2z @ S4z)
    Pe24 = Pt24 - Ps24

    Ps34 = 1 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) - (S3x @ S4x + S3y @ S4y + S3z @ S4z)
    Pt34 = 3 / 4 * np.eye(round((2 * (2 * I + 1)) ** N)) + (S3x @ S4x + S3y @ S4y + S3z @ S4z)
    Pe34 = Pt34 - Ps34
# ----------------------squeezing----------------------#
ini_Rho_atom, Rho_atom = Generate_a_squeezed_state_by_QND(2, I, T_sq, s, alpha, dt)

Rho_atom3 = np.zeros(2 * (a + b + 1))
# --------------------------------Define the initial state-----------------------------------#
theta = np.pi / 2
phi = 0
a_theta = spin_Jx(a) * np.sin(theta) * np.cos(phi) + spin_Jy(a) * np.sin(theta) * np.sin(phi) + spin_Jz(a) * np.cos(
    theta)
b_theta = spin_Jx(b) * np.sin(theta) * np.cos(phi) + spin_Jy(b) * np.sin(theta) * np.sin(phi) + spin_Jz(b) * np.cos(
    theta)
qa, va = np.linalg.eig(np.array(a_theta.full()))
qb, vb = np.linalg.eig(np.array(b_theta.full()))
vs = block_diag(va, vb)
qs = np.hstack((qa, qb))
# # -----------------spin temperature state-----------------#
P = 0.5
beta = np.log((1 + P) / (1 - P))
for i in np.arange(0, 2 * (a + b + 1), 1):
    Rho_atom3 =Rho_atom3 + np.exp(beta * qs[i]) * vs[:, [i]] * vs[:, [i]].T.conjugate()
Rho_atom3 = Rho_atom3 / np.trace(Rho_atom3)

# Rho_atom3 = np.eye(8)/8
Rho_atomi =np.kron(np.kron(Rho_atom, Rho_atom3),Rho_atom3)
# ----------------------Evolution of the spin under magnetic field and hyperfine interaction----------------------#
T = 2
dt = 0.01
n = round(T / dt)
te = np.arange(0, T, dt)
C_1F = [None] * n
C_2F = [None] * n
C_3F = [None] * n
C_4F = [None] * n
C_1S = [None] * n
C_2S = [None] * n
C_3S = [None] * n
C_4S = [None] * n



q, v = np.linalg.eig(H)
evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
mathcal_F=a1z+a2z+a3z+a4z+b1z+b2z+b3z+b4z
mathcal_S=(a1z+a2z+a3z+a4z-b1z-b2z-b3z-b4z)/4
Rho_atom = Rho_atomi
for t in trange(0, n, 1):
    C_1F[t] = np.trace(Rho_atom @ mathcal_F@ mathcal_F) - np.trace(Rho_atom @mathcal_F) ** 2
    C_1S[t] = np.trace(Rho_atom @ mathcal_S@ mathcal_S) - np.trace(Rho_atom @mathcal_S) ** 2
    Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    Rho_atom = hyperfine * Rho_atom

Rho_atom = Rho_atomi
for t in  trange(0, n, 1):
    C_2F[t] = np.trace(Rho_atom @ mathcal_F@ mathcal_F) - np.trace(Rho_atom @mathcal_F) ** 2
    C_2S[t] = np.trace(Rho_atom @ mathcal_S@ mathcal_S) - np.trace(Rho_atom @mathcal_S) ** 2
    # Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    hh=np.random.uniform()
    if hh<0.9:
        r = np.random.uniform()
        if r  < 0.16:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe12
        elif r < 0.32:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe13
        elif r<0.48 :
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe14
        elif r<0.64 : 
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe23
        elif r<0.8:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe24
        else:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe34
        Rho_atom = sec @ Rho_atom @ sec.T.conjugate()
    Rho_atom = hyperfine * Rho_atom

Rho_atom = Rho_atomi
for t in  trange(0, n, 1):
    C_3F[t] = np.trace(Rho_atom @ mathcal_F@ mathcal_F) - np.trace(Rho_atom @mathcal_F) ** 2
    C_3S[t] = np.trace(Rho_atom @ mathcal_S@ mathcal_S) - np.trace(Rho_atom @mathcal_S) ** 2
    hh=np.random.uniform()
    if hh<0.9:
        r = np.random.uniform()
        if r  < 0.16:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe12
        elif r < 0.32:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe13
        elif r<0.48 :
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe14
        elif r<0.64 : 
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe23
        elif r<0.8:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe24
        else:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe34
        Rho_atom = sec @ Rho_atom @ sec.T.conjugate()
    # C_1z2z[t] = np.trace(ini_Rho_atom @ a1z @ a2z)
    Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    Rho_atom = hyperfine * Rho_atom

# ----------------------Magnetic field----------------------#
omega_0 = 1.5
# H = omega_e * (ax-bx)              #一个原子
# H = omega_0 * (a1x + a2x - b1x - b2x)  # 两个原子
# H = omega_0 * (a1x + a2x + a3x- b1x - b2x - b3x)  三个原子
H = omega_0 * (a1x + a2x + a3x+a4x - b1x - b2x - b3x-b4x)  #四个原子
q, v = np.linalg.eig(H)
evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
Rho_atom = Rho_atomi  
for t in  trange(0, n, 1):
    C_4F[t] = np.trace(Rho_atom @ mathcal_F@ mathcal_F) - np.trace(Rho_atom @mathcal_F) ** 2
    C_4S[t] = np.trace(Rho_atom @ mathcal_S@ mathcal_S) - np.trace(Rho_atom @mathcal_S) ** 2
    hh=np.random.uniform()
    if hh<0.9:
        r = np.random.uniform()
        if r  < 0.16:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe12
        elif r < 0.32:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe13
        elif r<0.48 :
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe14
        elif r<0.64 : 
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe23
        elif r<0.8:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe24
        else:
            phi = np.random.normal(np.pi / 2, 2)
            sec = np.cos(phi) * np.eye((2 * (a + b + 1)) ** N) - 1j * np.sin(phi) * Pe34
        Rho_atom = sec @ Rho_atom @ sec.T.conjugate()
    # C_1z2z[t] = np.trace(ini_Rho_atom @ a1z @ a2z)
    Rho_atom = evolving_B @ Rho_atom @ evolving_B.T.conjugate()
    Rho_atom = hyperfine * Rho_atom
C_1F=np.array(C_1F)
C_1S=np.array(C_1S)
C_2F=np.array(C_2F)
C_2S=np.array(C_2S)
C_3F=np.array(C_3F)
C_3S=np.array(C_3S)
C_4F=np.array(C_4F)
C_4S=np.array(C_4S)

CSS=np.array([1,0,0,0,0,0,0,0])
CSS3=np.kron(np.kron(np.kron(CSS,CSS),CSS),CSS)
Rho_CSS=np.outer(CSS3,CSS3)
mathcal_S=(a1x+a2x+a3x-b1x-b2x-b3x)/4
mathcal_F=(a1x+a2x+a3x+b1x+b2x+b3x)

VarS=np.trace(Rho_CSS@mathcal_S@mathcal_S)-np.trace(Rho_CSS@mathcal_S)**2
VarF=np.trace(Rho_CSS@mathcal_F@mathcal_F)-np.trace(Rho_CSS@mathcal_F)**2

tt = np.arange(0, n, 1)*dt
with plt.style.context(['science']):
    fig = plt.figure(figsize=(3,5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    p1F, = ax1.plot(tt, C_1F )
    p1S, = ax2.plot(tt, C_1S )
    p2F, = ax1.plot(tt, C_2F)
    p2S, = ax2.plot(tt, C_2S)
    p3F, = ax1.plot(tt, C_3F)
    p3S, = ax2.plot(tt, C_3S)
    p4F, = ax1.plot(tt, C_4F)
    p4S, = ax2.plot(tt, C_4S)
    # p5F, = ax1.plot(tt, np.ones(len(tt))*VarF)
    # p5S, = ax2.plot(tt, np.ones(len(tt))*VarS)
    # ax1.legend([p1F,p2F,p3F,p4F,p5F],
    #            ["$R_{\\text{se}}=0$ Hz,$\omega_0=30$ rad/s","$R_{\\text{se}}=30$ Hz, $\omega_0=30$ rad/s","$R_{\\text{se}}=30$ Hz, $\omega_0=1.5$ rad/s","$R_{\\text{se}}=30$ Hz, $\omega_0=0$ rad/s","CSS"],bbox_to_anchor=(0.8, -0.2),ncol=1)
    ax2.legend([p1S,p3S,p4S,p2S],
               ["$R_{\\text{se}}=0$ Hz,$\omega_0=30$ rad/s","$R_{\\text{se}}=30$ Hz, $\omega_0=30$ rad/s","$R_{\\text{se}}=30$ Hz, $\omega_0=1.5$ rad/s","$R_{\\text{se}}=30$ Hz, $\omega_0=0$ rad/s"],bbox_to_anchor=(0.8, -0.2),ncol=1)
    ax1.set_xlabel('$t$ (s)')
    ax1.set_ylabel('Var $( \mathcal F_{x})$')
    ax2.set_xlabel('$t$ (s)')
    ax2.set_ylabel('Var $( \mathcal S_{x})$')
    # plt.xlim(0, 2)
    # plt.ylim(0.1,0.55)
    plt.savefig('desqueezing.png', dpi=600)
