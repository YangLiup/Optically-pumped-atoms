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
from tqdm import trange

I=3/2
omega_0=0.01
dt=0.01
T=50000
# cycle=round(5e5)
# omega_0=0.05
P=0.99
# --------------------------------Properties of the alkali metal atom-----------------------------------#
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

# --------------------------------Characterize interactions envolved-----------------------------------#
Rse = 1


# --------------------------------Define the initial state-----------------------------------#
theta = np.pi / 2
phi = 0
a_theta = spin_Jx(a) * np.sin(theta) * np.cos(phi) + spin_Jy(a) * np.sin(theta) * np.sin(phi) + spin_Jz(a) * np.cos(
    theta)
b_theta = spin_Jx(b) * np.sin(theta) * np.cos(phi) + spin_Jy(b) * np.sin(theta) * np.sin(phi) + spin_Jz(b) * np.cos(
    theta)
qa, va = np.linalg.eig(a_theta.full())
qb, vb = np.linalg.eig(b_theta.full())
v = block_diag(va, vb)
q = np.hstack((qa, qb))
Rho_ini = np.zeros(2 * (a + b + 1))
# # -----------------spin temperature state-----------------#
beta = np.log((1 + P) / (1 - P))
for i in np.arange(0, 2 * (a + b + 1), 1):
    Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
Rho_ini = Rho_ini / np.trace(Rho_ini)

# -----------------eigenstates-----------------#

# Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

# --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
Rhot = Rho_ini
hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
cycle=round(T/dt)
FF = np.zeros(cycle)
PP = np.zeros(cycle)
Gammam1 = np.zeros(cycle)
Gammam2 = np.zeros(cycle)
Gammam3 = np.zeros(cycle)

# MFx = np.array([])
# MFy = np.array([])
# MPx = np.array([])
# MPy = np.array([])
H = omega_0 * (az - bz)  # 投影定理
q, v = np.linalg.eig(H)
evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
for k in np.arange(10000):
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
    Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect
    Rhot = Rse * (alpha + 4 * alpha @ mSS - Rhot) * dt + Rhot
    Rhot = hyperfine * Rhot

for n in trange(cycle):
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
    Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect
    Rhot = Rse * (alpha + 4 * alpha @ mSS - Rhot) * dt  + Rhot
    Rhot = hyperfine * Rhot
    # -----------------Observables-----------------#
    P_t=np.sqrt(np.trace(Rhot@Sx)**2+np.trace(Rhot@Sy)**2)*2

    eta=(5+3*P_t**2)/(1-P_t**2)
    q=2*(3+P_t**2)/(1+P_t**2)
    max = np.trace((ax)@Rhot)
    mbx = np.trace((bx)@Rhot)
    mFx = max+mbx
    may = np.trace((ay)@Rhot)
    mby = np.trace((by)@Rhot)
    mFy = may+mby
    module=np.sqrt(mFx**2+mFy**2)
    ex=mFx/module
    ey=mFy/module
    maX=max*ex+may*ey
    mbX=mbx*ex+mby*ey
    maY=np.sqrt((max**2+may**2)-(max*ex+may*ey)**2)
    mbY=-np.sqrt((mbx**2+mby**2)-(mbx*ex+mby*ey)**2)
    Gammam1[n] =maX-eta*mbX
    Gammam2[n] =maY-eta*mbY
    Gammam3[n] =maY+eta*mbY






plt.figure()
# plt.plot(PP,(fp2/DD-Gammam/fm2)/(fp2/DD+Gammam/fm2))
# plt.plot(np.arange(0,cycle,1)*dt,(Gammam2/Gammam1))
plt.plot(np.arange(0,cycle,1)*dt,(Gammam2))
plt.plot(np.arange(0,cycle,1)*dt,(Gammam3))

plt.show()
    


