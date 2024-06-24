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
def masterequation(I):
    # --------------------------------Properties of the alkali metal atom-----------------------------------#
    T=5000
    dt=0.01
    omega_0=0.05
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
    Rop = 0
    Rsd = 0
    sx=np.sqrt(1)/(2)
    sz=np.sqrt(1)/(2)
    # --------------------------------Define the initial state-----------------------------------#
    theta = np.pi / 2
    phi = 0
    a_theta = spin_Jx(a) * np.sin(theta) * np.cos(phi) + spin_Jy(a) * np.sin(theta) * np.sin(phi) + spin_Jz(a) * np.cos(
        theta)
    b_theta = spin_Jx(b) * np.sin(theta) * np.cos(phi) + spin_Jy(b) * np.sin(theta) * np.sin(phi) + spin_Jz(b) * np.cos(
        theta)
    qa, va = np.linalg.eig(a_theta)
    qb, vb = np.linalg.eig(b_theta)
    v = block_diag(va, vb)
    q = np.hstack((qa, qb))
    Rho_ini = np.zeros(2 * (a + b + 1))
    # # -----------------spin temperature state-----------------#
    P = 0.99
    beta = np.log((1 + P) / (1 - P))
    for i in np.arange(0, 2 * (a + b + 1), 1):
        Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
    Rho_ini = Rho_ini / np.trace(Rho_ini)

    # -----------------eigenstates-----------------#

    # Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

    # --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
    Rhot = Rho_ini
    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    MFx = np.zeros(round(T / dt))
    MFy = np.zeros(round(T / dt))
    MPx = np.zeros(round(T / dt))
    MPy = np.zeros(round(T / dt))
    H = omega_0 * (az - bz)  # 投影定理
    q, v = np.linalg.eig(H)
    evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
    for n in np.arange(0, round(T / dt), 1):
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
        # -----------------Observables-----------------#
        MFx[n] = np.trace((ax+bx)@Rhot)
        MFy[n] = np.trace((ay+by)@Rhot)
        MPx[n] = np.trace((Sx)@Rhot)*2
        MPy[n] = np.trace((Sy)@Rhot)*2
        # Vx = np.trace(Rhot @ ((ax + bx) @ (ax + bx) + (ay + by) @ (ay + by) + (az + bz) @ (az + bz)))
        # V[n] = Vx
    FF=np.sqrt(MFy**2+MFx**2)
    PP=np.sqrt(MPy**2+MPx**2)
    D = np.zeros(round(T / dt))
    for n in np.arange(0, round(T / dt)-1, 1):
        D[n+1]=(FF[n+1]-FF[n])/dt
    DD=-D/FF/omega_0**2/(2*I+1)**2 

    P=np.arange(0,1,0.01)
    if a==2:
        q = 2 * (3 + P ** 2) / (1 + P ** 2)
        Gamma =  (-4 + q) * (4 + q)  / 12 / q ** 2 
    if a==3:
        q = 2 * (19 + 26 * P ** 2 + 3 * P ** 4) / (3 + 10 * P ** 2 + 3 * P ** 4)
        Gamma =(-6 + q) * (6 + q)  / 2 / q ** 2 /(2*19/3)
    if a==4:
        q = 2 * (11 + 35 * P ** 2 + 17 * P ** 4 + P ** 6) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6)
        Gamma =(-8 + q) * (8 + q)  / 2 / q ** 2 /(22)
    
    return P, Gamma, PP, DD
    
    # plt.style.use(['science','nature'])
    # with plt.style.context(['science','nature']):
    #     plt.rc('font',family='Times New Roman')
    #     plt.figure()
    #     p1, = plt.plot(PP,-D/FF/omega_0**2)
    #     p2, = plt.plot(P,Gamma)

    #     # plt.xlim(0, 200)
    #     # plt.ylim(0, 18)
    #     plt.xticks(fontsize=10)
    #     plt.yticks(fontsize=10)
    #     plt.legend([p1, p2], ["Density matrix", "NB"], loc='upper right',
    #             prop={'size': 9})

    #     plt.xlabel('$P$', fontsize=10)
    #     plt.ylabel('$\Gamma_t^+$', fontsize=10)
    #     plt.savefig('imag/Gamma_t+.png', dpi=1000)
    # plt.show()

