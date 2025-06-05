# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月24日
"""
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import scienceplots
from tqdm import trange

I=3/2
omega_0=1
dt=0.001
T=50
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
hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
H = omega_0 * (az - bz)  # 投影定理
q, v = np.linalg.eig(H)
evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)

theta = np.pi / 2
phi = 0
a_theta = spin_Jx(a) * np.sin(theta) * np.cos(phi) + spin_Jy(a) * np.sin(theta) * np.sin(phi) + spin_Jz(a) * np.cos(
    theta)
b_theta = spin_Jx(b) * np.sin(theta) * np.cos(phi) + spin_Jy(b) * np.sin(theta) * np.sin(phi) + spin_Jz(b) * np.cos(
    theta)
qa, va = np.linalg.eig(a_theta.full())
qb, vb = np.linalg.eig(b_theta.full())
v = block_diag(va, vb)
eigenvalues = np.hstack((qa, qb))

cycle=round(T/dt)

# --------------------------------Define the initial state-----------------------------------#
#  -----------------state1-----------------#
Rho_ini=np.diag([1/4,1/4,0,0,0,1/4,1/4,0])
Rhot = Rho_ini
maz1 = np.zeros(cycle)
for n in trange(0,cycle,1):
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
    maz1[n]=np.trace(Rhot@az)

#  -----------------state2-----------------#
ket=np.array([1,1j,0,0,0,1,-1j,0])/2
Rho_ini=np.outer(ket,ket.conjugate())
Rhot = Rho_ini
maz2 = np.zeros(cycle)
for n in trange(0,cycle,1):
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
    maz2[n]=np.trace(Rhot@az)



# -----------------state2 spin temperature state-----------------#
# P=0.99
# Rho_ini = np.zeros(2 * (a + b + 1))
# beta = np.log((1 + P) / (1 - P))
# for i in np.arange(0, 2 * (a + b + 1), 1):
#     Rho_ini = Rho_ini + np.exp(beta * eigenvalues[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
# Rho_ini = Rho_ini / np.trace(Rho_ini)




t=np.arange(0,cycle,1)*dt
plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig = plt.figure()
    plt.rc('font',family='Times New Roman')
    ax = fig.add_subplot(111)
    p1=ax.plot(t,maz1)
    p2=ax.plot(t,maz2)
    ax.set_xlabel("t (1/Rse)")
    ax.set_ylabel("az")
    ax.legend(["without coherence", "with coherence"], loc='lower right')
plt.savefig('coupling.png', dpi=1000)
plt.show()