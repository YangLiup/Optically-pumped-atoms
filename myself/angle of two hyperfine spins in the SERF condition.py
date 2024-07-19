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
from sympy import pi
from scipy.linalg import *
import scienceplots
from tqdm import trange
I=3/2

# --------------------------------Properties of the alkali metal atom-----------------------------------#
# I = 5 / 2
a = round(I + 1 / 2)
b = round(I - 1 / 2)

# --------------------------------Generate the angular momentum operators-----------------------------------#
U = alkali_atom_uncoupled_to_coupled(round(2 * I))
ax, ay, az, bx, by, bz = spin_operators_of_2or1_alkali_metal_atoms(1, I)
sx=np.array([[0,0.5],[0.5,0]])
sy=np.array([[0,1j],[-1j,0]])*0.5
sz=np.array([[0.5,0],[0,-0.5]])
Sx = np.kron(np.eye(round(2 * I + 1)), np.array(sx))
Sx = U.T.conjugate() @ Sx @ U
Sy = np.kron(np.eye(round(2 * I + 1)), np.array(sy))
Sy = U.T.conjugate() @ Sy @ U
Sz = np.kron(np.eye(round(2 * I + 1)), np.array(sz))
Sz = U.T.conjugate() @ Sz @ U

# --------------------------------Characterize interactions envolved-----------------------------------#
Rse = 1
omega_0 = 0.05
# --------------------------------Define the initial state-----------------------------------#
theta = np.pi/2
phi = 0
a_theta = spin_Jx(a) * np.sin(theta) * np.cos(phi) + spin_Jy(a) * np.sin(theta) * np.sin(phi) + spin_Jz(a) * np.cos(
    theta)
b_theta = spin_Jx(b) * np.sin(theta) * np.cos(phi) + spin_Jy(b) * np.sin(theta) * np.sin(phi) + spin_Jz(b) * np.cos(
    theta)
qa, va = np.linalg.eig(np.array((a_theta.full())))
qb, vb = np.linalg.eig(np.array((b_theta).full()))
v = block_diag(va, vb)
q = np.hstack((qa, qb))
P=0.9
Rho_ini = np.zeros(2 * (a + b + 1))
beta = np.log((1 + P) / (1 - P))
for i in np.arange(0, 2 * (a + b + 1), 1):
    Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] @ v[:, [i]].T.conjugate()
Rho_ini = Rho_ini / np.trace(Rho_ini)
Rhot = Rho_ini


H =omega_0 * (az - bz) # 投影定理
dt = 0.01
q, v = np.linalg.eig(H)
# # -----------------spin temperature state-----------------#


# -----------------eigenstates-----------------#

# Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

# --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#

hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    # -----------------Evolution-----------------#

cycle=round(1/dt)*5000
module=np.zeros(cycle)
PP = np.zeros(cycle)
for k in trange(0,cycle,1):

    P_ini=np.sqrt(np.trace(Rhot@Sx)**2+np.trace(Rhot@Sy)**2)*2
    PP[k]=P_ini
    eta=(5+3*P_ini**2)/(1-P_ini**2)
    evolving_B = v @ np.diag(np.exp(-1j * q *dt)) @ np.linalg.inv(v)
    Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect
    Rhot = hyperfine * Rhot
    x1 = Rhot @ Sx
    x2 = Rhot @ Sy
    x3 = Rhot @ Sz
    AS = 3 / 4 * Rhot - (Sx @ x1 + Sy @ x2 + Sz @ x3)
    alpha = Rhot - AS
    mSx = np.trace(x1)
    mSy = np.trace(x2)
    mSz = np.trace(x3)
    mSS = mSx * Sx + mSy * Sy + mSz * Sz
    Rhot = Rse * (alpha + 4 * alpha @ mSS - Rhot) * dt  + Rhot
    # module[k]=np.sqrt(np.trace((ax-eta*bx)@Rhot)**2+np.trace((ay-eta*by)@Rhot)**2)
    # module[k]=(np.trace((ax)@Rhot))
    max=np.trace((ax)@Rhot)
    mbx=np.trace((bx)@Rhot)
    may=np.trace((ay)@Rhot)
    mby=np.trace((by)@Rhot)
    mcx=np.trace((ax-bx)@Rhot)
    mcy=np.trace((ay-by)@Rhot)
    module[k]=np.arccos(((max**2+may**2)+(mbx**2+mby**2)-(mcx**2+mcy**2))/(2*np.sqrt((max**2+may**2))*np.sqrt((mbx**2+mby**2))))
# plt.plot(np.arange(0,t,1)*dt,-(np.log(module)-np.max(np.log(module)))/(np.arange(0,t,1)*dt))
plt.figure()
plt.plot(PP,module)
plt.ylim([0.12,0.14])
plt.show()

