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


def master_equation(T, dt, tp,omega_0): 
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
    Rop = 0.1
    Rsd = 0.0005
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
    P = 0.000001
    beta = np.log((1 + P) / (1 - P))
    for i in np.arange(0, 2 * (a + b + 1), 1):
        Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
    Rho_ini = Rho_ini / np.trace(Rho_ini)

    # -----------------eigenstates-----------------#

    # Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

    # --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
    Rhot = Rho_ini
    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    MSx = np.zeros(round(T / dt))
    MSz = np.zeros(round(T / dt))

    H = omega_0 * (az - bz)  # 投影定理
    q, v = np.linalg.eig(H)
    evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
    for n in np.arange(0, round(T / dt), 1):
        # -----------------Evolution-----------------#
        if n==round(T / dt)/tp:
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
        ER = -Rsd * AS
        OP = Rop * (2 * alpha @ (sx*Sx+sz*Sz) - AS)
        Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect
        Rhot = Rse * (alpha + 4 * alpha @ mSS - Rhot) * dt + (
                ER + OP) * dt + Rhot
        Rhot = hyperfine * Rhot
        # -----------------Observables-----------------#
        MSz[n] = mSz
        MSx[n] = mSx
        # Vx = np.trace(Rhot @ ((ax + bx) @ (ax + bx) + (ay + by) @ (ay + by) + (az + bz) @ (az + bz)))
        # V[n] = Vx

    # ---------------------------Bloch Equation-----------------------------------
    Rop = 0.1
    transverse = np.zeros(round(T / dt))
    longitude = np.zeros(round(T / dt))
    Px = P * np.sin(theta)
    Pz = P * np.cos(theta)
    Py = 0
    for n in np.arange(0, round(T / dt), 1):
        if n==round(T / dt)/tp:
            Rop = 0.0
        transverse[n] = Px
        longitude[n] = Pz
        qnm = 2 * (3 + P ** 2) / (1 + P ** 2)
        Qnm = 2 * (3 + P ** 4) / ((1 + P ** 2) ** 2)
        Gamma = 8 * (-4 + qnm) * (4 + qnm) * omega_0 ** 2 / qnm ** 3 * qnm / Qnm

        # qnm = 2*(19+26*P**2+3*P**4)/(3+10*P**2+3*P**4)
        # Qnm = 2 * (57 +44* P ** 2+134*P**4+12*P**6+9*P**8) / ((3 +10* P ** 2+3*P**4 )**2)
        # Gamma = 18 * (-6 + qnm) * (6 + qnm) * omega_0 ** 2 / qnm ** 3 * qnm / Qnm

        # qnm = 2 * (11 +35* P ** 2+17*P**4+P**6) / (1 +7* P ** 2+7*P**4+P**6) 
        # Qnm = 2 * (11 +28* P ** 2+99*P**4+64*P**6+49*P**8+4*P**10+P**12) / ((1 +7* P ** 2+7*P**4+P**6)**2 )
        # Gamma = 32 * (-8 + qnm) * (8 + qnm) * omega_0 ** 2 / qnm ** 3 * qnm / Qnm

    
    
        T2 = (1 - ((qnm - Qnm) / qnm) * Pz ** 2 / P ** 2) * Gamma
        T1 = T2-Gamma/qnm*Qnm
        # Pz = Pz + (-1 / Qnm * (Rsd + Rop) * Pz + Rop * (
        #         1 / Qnm * Pz ** 2 / P ** 2 + 1 / qnm * (1 - Pz ** 2 / P ** 2)) - Pz * T1) * dt
        # Px = Px + (-1 / Qnm * (Rsd + Rop) * Px + Rop * (
        #         1 / Qnm - 1 / qnm) * Pz / P ** 2 * Px - 1 / qnm * omega_0 * 4 * Py - Px * T2) * dt
        # Py = Py + (-1 / Qnm * (Rsd + Rop) * Py + Rop * (
        #         1 / Qnm - 1 / qnm) * Pz / P ** 2 * Py + 1 / qnm * omega_0 * 4 * Px - Py * T2) * dt
        Pz = Pz + (-1 / Qnm * (Rsd + Rop) * Pz + Rop *
                ((1 / Qnm - 1 / qnm) * (Px *sx+Pz*sz)*(Pz-Px *sx*sz-Pz*sz*sz) / P ** 2)+Rop *sz*
                ((1 / Qnm  * (Px *sx+Pz*sz)**2/ P ** 2)+(1 / qnm  * (1-(Px *sx+Pz*sz)**2/ P ** 2)))
                - Pz * T1) * dt
        Px = Px + (-1 / Qnm * (Rsd + Rop) * Px +Rop *
                ((1 / Qnm - 1 / qnm) * (Px *sx+Pz*sz)*(Px-Px *sx*sx-Pz*sz*sx) / P ** 2)+Rop *sx*
                ((1 / Qnm  * (Px *sx+Pz*sz)**2/ P ** 2)+(1 / qnm  * (1-(Px *sx+Pz*sz)**2/ P ** 2)))
                - 1 / qnm * omega_0 * 4 * Py - Px * T2) * dt
        Py = Py + (-1 / Qnm * (Rsd + Rop) * Py + Rop * (
                1 / Qnm - 1 / qnm) * (Px*sx+Pz*sz) / P ** 2 * Py + 1 / qnm * omega_0 * 4 * Px - Py * T2) * dt
        P = np.sqrt(Px ** 2 + Py ** 2 + Pz ** 2)
    return transverse, longitude, 2*MSz, 2*MSx