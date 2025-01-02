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




# --------------------------------Define the initial state-----------------------------------#
theta = 0
phi = 0
PP=np.arange(1e-5,0.9999,0.01)
mT20=np.zeros(len(PP))
j=0
for P in PP: 
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
    T20=np.sqrt(30/((2*a+3)*(2*a+1)*a*(2*a-1)*(a+1)))/np.sqrt(6)*(3*az@az-(az@az+ax@ax+ay@ay))
    mT20[j]=np.trace(T20@Rho_ini)
    j=j+1

plt.figure()
plt.plot(PP,mT20)
plt.xlabel('P')
plt.ylabel('$\langle T_{20}(aa) \\rangle$')
plt.show()