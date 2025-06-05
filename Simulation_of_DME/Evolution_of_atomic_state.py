import sys
sys.path.append(r"D:\Optically-pumped-atoms\my_functions")

import matplotlib.pyplot as plt
from spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import numpy as np
from matplotlib import ticker
from tqdm import trange
import scienceplots


def master_equation(I,Rse,omega_0,T,N):
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
    # omega_0 = 0.01
    Rop =0.5 #0.1
    Rsd =0.002 #50/1e4
    sx=0
    sz=1
    # --------------------------------Define the initial state-----------------------------------#
    theta = np.pi / 4
    phi = np.pi / 4
    a_theta = spin_Jx(a) * np.sin(theta) * np.cos(phi) + spin_Jy(a) * np.sin(theta) * np.sin(phi) + spin_Jz(a) * np.cos(
        theta)
    b_theta = spin_Jx(b) * np.sin(theta) * np.cos(phi) + spin_Jy(b) * np.sin(theta) * np.sin(phi) + spin_Jz(b) * np.cos(
        theta)
    qa, va = np.linalg.eig(np.array(a_theta.full()))
    qb, vb = np.linalg.eig(np.array(b_theta.full()))
    v = block_diag(va, vb)
    q = np.hstack((qa, qb))
    Rho_ini = np.zeros(2 * (a + b + 1))

    # # -----------------spin temperature state-----------------#
    P = 1e-10
    beta = np.log((1 + P) / (1 - P))
    for i in np.arange(0, 2 * (a + b + 1), 1):
        Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
    Rho_ini = Rho_ini / np.trace(Rho_ini)

    # -----------------eigenstates-----------------#

    # Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0])


    # --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
    Rhot = Rho_ini
    dt = 0.01
    t = np.arange(0, T, dt)
    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    MSx = np.zeros(round(T / dt))
    MSz = np.zeros(round(T / dt))


    theta_probe=np.pi/4
    a_theta=(ay*np.sin(theta_probe)+az*np.cos(theta_probe))
    b_theta=(by*np.sin(theta_probe)+bz*np.cos(theta_probe))
    b1=0.001
    b2=-0.0001
    H = omega_0 * (az - bz) +b1*(3*a_theta@a_theta-6)+b2*(3*b_theta@b_theta-2)    
    q, v = np.linalg.eig(H)
    evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
    for n in trange(0, round(T / dt), 1):
        # -----------------Evolution-----------------#
        if n==round(T / dt)/N:
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
        MSx[n] = np.sqrt(mSx**2+mSy**2)
  
    Px=2*MSx
    Pz=2*MSz
    return Px, Pz

T=5000
dt=0.01
tt=np.arange(0,T,dt)
Px,Pz=master_equation(3/2,1,0.01,T,10)
op=np.zeros(round(T/dt))
x=int(round(T / dt)/10)
for i in range(x):
    op[i] = 0.5


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(tt, Px)
ax1.plot(tt, op,linestyle='dashed')
ax1.set_ylabel('Polarization', fontsize=8)
ax1.tick_params(axis='both', which='major', labelsize=8)
ax1.tick_params(axis='both', which='minor', labelsize=8)

ax1.legend(["$P_{x}$","$R_{\\text{op}}$"])
plt.savefig('Evolution.png', dpi=1000)
