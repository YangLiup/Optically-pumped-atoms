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



# -----------------eigenstates-----------------#

Rho_ini = (np.outer(np.array([1, 0, 0, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0, 0, 0]))+np.outer(np.array([0, 0, 0, 0, 1, 0, 0, 0]), np.array([0, 0, 0, 0, 1, 0, 0, 0])))/2

# --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
Rhot = Rho_ini
dt = 0.01
T = 200
t = np.arange(0, T, dt)
hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
clear = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.zeros((2 * b + 1, 2 * b + 1)))  # 一个原子

MSx = np.zeros(round(T / dt))
MSz = np.zeros(round(T / dt))
V = np.zeros(round(T / dt))
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
    Rhot = Rse * (alpha + 4 * alpha @ mSS - Rhot) * dt + Rhot
    for j in np.arange(0,1,1):
        Rhott=Rhot
        Rhot = clear * Rhot
        Rhot[0,0]=Rhott[5,5]*1/6+Rhot[0,0]
        Rhot[1,1]=Rhott[5,5]*1/12+Rhott[6,6]/4+Rhot[1,1]
        Rhot[2,2]=Rhott[5,5]*1/4+Rhott[7,7]/4+Rhot[2,2]
        Rhot[3,3]=Rhott[7,7]*1/12+Rhott[6,6]/4+Rhot[3,3]
        Rhot[4,4]=Rhott[7,7]*1/6+Rhot[4,4]
        Rhot[5,5]=1/4*Rhott[5,5]+1/12*Rhott[6,6]+Rhot[5,5]
        Rhot[6,6]=1/3*Rhott[6,6]+1/4*Rhott[5,5]+1/4*Rhott[7,7]+Rhot[6,6]
        Rhot[7,7]=1/12*Rhott[6,6]+1/4*Rhott[7,7]+Rhot[7,7]
    Rhot = hyperfine * Rhot
    # -----------------Observables-----------------#
    MSz[n] = mSz
    MSx[n] = mSx
    # Vx = np.trace(Rhot @ ((ax + bx) @ (ax + bx) + (ay + by) @ (ay + by) + (az + bz) @ (az + bz)))
    # V[n] = Vx
plt.style.use(['science'])
with plt.style.context(['science']):
    plt.figure()
    plt.bar(np.array([1,2,3,4,5,6,7,8]), np.diag(Rhot))
    # plt.xlim(0, 200)
    # plt.ylim(0, 18)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.legend([p1, p3, p2, p4], ["$P_x^{\mathrm{DM}}$", "$P_x^{\mathrm{NB}}$", "$P_z^{\mathrm{DM}}$", "$P_z^{\mathrm{NB}}$"], loc='upper right',
    #            prop={'size': 10})

    plt.show()