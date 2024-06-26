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
def gammam(I,bound): 

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
    H = (az - bz) # 投影定理
    dt = 0.0001
    q, v = np.linalg.eig(H)
    evolving_B = v @ np.diag(np.exp(-1j * q *0.01)) @ np.linalg.inv(v)
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
    # # -----------------spin temperature state-----------------#


    # -----------------eigenstates-----------------#

    # Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

    # --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#

    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    Fmmt = np.zeros(bound)
    PP = np.zeros(bound)

    for n in np.arange(0,bound, 1):
        # -----------------Evolution-----------------#
        Rho_ini = np.zeros(2 * (a + b + 1))
        P = n/1000
        beta = np.log((1 + P) / (1 - P))
        for i in np.arange(0, 2 * (a + b + 1), 1):
            Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] @ v[:, [i]].T.conjugate()
        Rho_ini = Rho_ini / np.trace(Rho_ini)
        Rho_ini = evolving_B @Rho_ini @ evolving_B.T.conjugate()  # Zeeman effect
        Rhot = Rho_ini
        P_ini=np.sqrt(np.trace(Rhot@Sx)**2+np.trace(Rhot@Sy)**2)*2
        PP[n]=P_ini
        if a==2:
            eta=(5+3*P_ini**2)/(1-P_ini**2)
        if a==3:
            qq=2*(19+26*P_ini**2+3*P_ini**4)/(3+10*P_ini**2+3*P_ini**4)
            eta=(qq+6)/(qq-6)
        if a==4:
            qq=2*(11+35*P_ini**2+17*P_ini**4+P_ini**6)/(1+7*P_ini**2+7*P_ini**4+P_ini**6)
            eta=(qq+8)/(qq-8)
        Fxm0 = np.trace((ax-eta*bx)@Rhot)
        Fym0 = np.trace((ay-eta*by)@Rhot)
        Fm0=np.sqrt(Fxm0**2+Fym0**2)
        for k in np.arange(0,2,1):
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
        Fxm = np.trace((ax-eta*bx)@Rhot)
        Fym = np.trace((ay-eta*by)@Rhot)
        Fm=np.sqrt(Fxm**2+Fym**2)
        

        Fmmt[n]=(Fm-Fm0)/(2*dt)/Fm0

    return PP,Fmmt
    
