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
omega_0=0.05
dt=0.01
T=200
# cycle=round(5e5)
# omega_0=0.05
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
hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
H = omega_0 * (az - bz)  # 投影定理
q, v = np.linalg.eig(H)
evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)

theta = np.pi / 180
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
module = np.zeros(cycle)
PP = np.zeros(cycle)
Gammam1 = np.zeros(cycle)
Gammam2 = np.zeros(cycle)

P=0.5
Rho_ini = np.zeros(2 * (a + b + 1))
# # -----------------spin temperature state-----------------#
beta = np.log((1 + P) / (1 - P))
for i in np.arange(0, 2 * (a + b + 1), 1):
    Rho_ini = Rho_ini + np.exp(beta * eigenvalues[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
Rho_ini = Rho_ini / np.trace(Rho_ini)
Rhot = Rho_ini

for n in np.arange(0,cycle,1):
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

    if a==2:
        eta=(5+3*P_t**2)/(1-P_t**2)
        q=2*(3+P_t**2)/(1+P_t**2)
    if a==3:
        q=2*(19+26*P_t**2+3*P_t**4)/(3+10*P_t**2+3*P_t**4)
        eta=(q+6)/(q-6)
    if a==4:
        q=2*(11+35*P_t**2+17*P_t**4+P_t**6)/(1+7*P_t**2+7*P_t**4+P_t**6)
        eta=(q+8)/(q-8)

    max=np.trace((ax)@Rhot)
    mbx=np.trace((bx)@Rhot)
    may=np.trace((ay)@Rhot)
    mby=np.trace((by)@Rhot)
    mcx=np.trace((ax-bx)@Rhot)
    mcy=np.trace((ay-by)@Rhot)
    module[n]=np.arccos(((max**2+may**2)+(mbx**2+mby**2)-(mcx**2+mcy**2))/(2*np.sqrt((max**2+may**2))*np.sqrt((mbx**2+mby**2))))
    PP[n]=max


    # Vx = np.trace(Rhot @ ((ax + bx) @ (ax + bx) + (ay + by) @ (ay + by) + (az + bz) @ (az + bz)))
    # V[n] = V

plt.style.use(['science','nature'])
with plt.style.context(['science','nature']):
    fig = plt.figure()
    plt.rc('font',family='Times New Roman')
    ax3 = fig.add_subplot(111)
    ax3.plot(np.arange(0,cycle,1)*dt,module)
    # ax3.plot(np.arange(0,cycle,1)*dt,PP)
    # ax3.plot(PP, Gammam1)
    # ax3.plot(PP, Gammam2)

    # ax3.set_xlabel('$P$', fontsize=10)
    # ax3.set_ylabel('$\Gamma_t^- \\approx \Gamma_z^-$ ($R_{\\rm{se}}$) ', fontsize=10)
    ax3.tick_params(axis='x', labelsize='10')
    ax3.tick_params(axis='y', labelsize='10')
    # ax3.set_xlim([0,1])
plt.savefig('ndd',dpi=1000)
plt.show()