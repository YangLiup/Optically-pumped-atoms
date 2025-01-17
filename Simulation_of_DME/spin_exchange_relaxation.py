# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年12月24日
"""

import sys
sys.path.append(r"D:\python\pythonProject\Optically_pumped_atoms\my_functions")
import numpy as np
import matplotlib.pyplot as plt
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import scienceplots
from tqdm import trange
P_range=np.arange(0.0001,0.99,0.01)
def rate(I):
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

    H = (az - bz) # 投影定理
    dt = 0.01
    qH, vH = np.linalg.eig(H)
    #-----------------spin temperature state-----------------#


    # -----------------eigenstates-----------------#

    # Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

    # --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#

    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    evolving_B = vH @ np.diag(np.exp(-1j * qH *(np.pi/40))) @ np.linalg.inv(vH)



    P_range=np.arange(0.0001,0.99,0.01)
    Gamma=np.zeros(len(P_range))
    Fm0_=np.zeros(len(P_range))

    j=0
    for P in P_range:
        Rho_ini = np.zeros(2 * (a + b + 1))
        beta = np.log((1 + P) / (1 - P))
        for i in np.arange(0, 2 * (a + b + 1), 1):
            Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] @ v[:, [i]].T.conjugate()
        Rho_ini = Rho_ini / np.trace(Rho_ini)
        Rhot = Rho_ini


        # ------------------------------------------转开一个小角度------------------------------------------------#
        Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect
        Rhot = hyperfine * Rhot


        if a==2:
            eta=(5+3*P**2)/(1-P**2)

        if a==3:
            q2 = 2 * (19 + 26 * P ** 2 + 3 * P ** 4) / (3 + 10 * P ** 2 + 3 * P ** 4)
            eta=(q2+6)/(q2-6)
       
        if a==4:
            q3 = 2 * (11 + 35 * P ** 2 + 17 * P ** 4 + P ** 6) / (1 + 7 * P ** 2 + 7 * P ** 4 + P ** 6)
            eta=(q3+8)/(q3-8)
        
        Fm0=np.sqrt(np.trace((ax-eta*bx)@Rhot)**2+np.trace((ay-eta*by)@Rhot)**2)

        for k in np.arange(0,2*dt,dt):
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
        Fm=np.sqrt(np.trace((ax-eta*bx)@Rhot)**2+np.trace((ay-eta*by)@Rhot)**2)
        Gamma[j]=-(Fm-Fm0)/(2*dt)/Fm0
        Fm0_[j]=Fm0
        j=j+1
    return Gamma,Fm0_
Gamma1,Fm01=rate(3/2)
Gamma2,Fm02=rate(5/2)
Gamma3,Fm03=rate(7/2)

plt.figure()
# p1,=plt.plot(P_range,Gamma1)
# p2,=plt.plot(P_range,Gamma2)
# p3,=plt.plot(P_range,Gamma3)
p1,=plt.plot(P_range,Fm01)
p2,=plt.plot(P_range,Fm02)
p3,=plt.plot(P_range,Fm03)
plt.legend([p1,p2,p3],['I=3/2','I=5/2','I=7/2'])
plt.xlabel('P')
plt.ylabel('${[a-\eta b]}_0$')
# plt.ylim([0.12,0.14])
plt.show()

