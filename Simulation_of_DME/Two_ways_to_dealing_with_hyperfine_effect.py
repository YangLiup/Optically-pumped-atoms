import sys
sys.path.append("..")
import matplotlib.pyplot as plt
from my_functions.spin_operators_of_2or1_alkali_metal_atoms import spin_operators_of_2or1_alkali_metal_atoms
from my_functions.alkali_atom_uncoupled_to_coupled import alkali_atom_uncoupled_to_coupled
from qutip import *
from scipy.linalg import *
import numpy as np
from tqdm import trange
import scienceplots

def master_equation1(I,Rse,omega_0,T,dt):
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

    Ix = np.kron(spin_Jx(I).full(), np.eye(round(2)))
    Ix = U.T.conjugate() @ Ix @ U
    Iy = np.kron(spin_Jy(I).full(), np.eye(round(2)))
    Iy = U.T.conjugate() @ Iy @ U
    Iz = np.kron(spin_Jz(I).full(), np.eye(round(2)))
    Iz = U.T.conjugate() @ Iz @ U

    # --------------------------------Characterize interactions envolved-----------------------------------#
    # omega_0 = 0.01
    Rsd = 0.1
    # sx=1  #/np.sqrt(2)
    # sz=0    #1/np.sqrt(2)
    # --------------------------------Define the initial state-----------------------------------#
    theta = np.pi/2
    phi = 0
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
    P = 0.99
    beta = np.log((1 + P) / (1 - P))
    for i in np.arange(0, 2 * (a + b + 1), 1):
        Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
    Rho_ini = Rho_ini / np.trace(Rho_ini)

    # -----------------eigenstates-----------------#

    # Rho_ini = np.outer(np.array([0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0, 0, 0, 0]))

    # --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
    # omega_rev=1/100
    Rhot = Rho_ini
    hyperfine = block_diag(np.ones((2 * a + 1, 2 * a + 1)), np.ones((2 * b + 1, 2 * b + 1)))  # 一个原子
    H = omega_0 * (az-bz) # 投影定理
    q, v = np.linalg.eig(H)
    evolving_B = v @ np.diag(np.exp(-1j * q * dt)) @ np.linalg.inv(v)
    MSx = np.zeros(round(T / dt))
    for n in trange(0, round(T / dt), 1):
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
        ER = -Rsd * AS
        # OP = Rop * (2 * alpha @ (sx*Sx+sz*Sz) - AS)
        # Rhot =Rhot+ (H@Rhot-Rhot@H)/1j*dt # Zeeman effect

        
        # Rhot=-1j*(H@Rhot-Rhot@H)*dt1+Rhot
        Rhot = Rse * (alpha + 4 * alpha @ mSS - Rhot) * dt + (
                ER ) * dt + Rhot
        Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect
        Rhot = hyperfine * Rhot
        # -----------------Observables-----------------#
        MSx[n] = mSx

    return MSx

def master_equation2(I,Rse,omega_0,T,dt):
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

    Ix = np.kron(spin_Jx(I).full(), np.eye(round(2)))
    Ix = U.T.conjugate() @ Ix @ U
    Iy = np.kron(spin_Jy(I).full(), np.eye(round(2)))
    Iy = U.T.conjugate() @ Iy @ U
    Iz = np.kron(spin_Jz(I).full(), np.eye(round(2)))
    Iz = U.T.conjugate() @ Iz @ U

    # --------------------------------Characterize interactions envolved-----------------------------------#
    # omega_0 = 0.01
    Rsd = 0.1
    # sx=1  #/np.sqrt(2)
    # sz=0    #1/np.sqrt(2)
    # --------------------------------Define the initial state-----------------------------------#
    theta = np.pi/2
    phi = 0
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
    P = 0.99
    beta = np.log((1 + P) / (1 - P))
    for i in np.arange(0, 2 * (a + b + 1), 1):
        Rho_ini = Rho_ini + np.exp(beta * q[i]) * v[:, [i]] * v[:, [i]].T.conjugate()
    Rho_ini = Rho_ini / np.trace(Rho_ini)

    # --------------------------------------Evolution under hyperfine effect, etc.--------------------------------#
    Rhot = Rho_ini
    HB = omega_0 *4* Sz # 投影定理
    Hh = 200* (Sx@Ix+Sy@Iy+Sz@Iz) # 投影定理
    qB, vB = np.linalg.eig(HB)
    evolving_B = vB @ np.diag(np.exp(-1j * qB * dt)) @ np.linalg.inv(vB)
    qh, vh = np.linalg.eig(Hh)
    evolving_h = vh @ np.diag(np.exp(-1j * qh * dt)) @ np.linalg.inv(vh)
    MSx = np.zeros(round(T / dt))
    for n in trange(0, round(T / dt), 1):
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
        ER = -Rsd * AS

        
        Rhot = Rse * (alpha + 4 * alpha @ mSS - Rhot) * dt + (
                ER ) * dt + Rhot
        Rhot=-1j*((HB+Hh)@Rhot-Rhot@(HB+Hh))*dt2+Rhot
        # Rhot = evolving_h @ Rhot @ evolving_h.T.conjugate()  # hyperfine effect
        # Rhot = evolving_B @ Rhot @ evolving_B.T.conjugate()  # Zeeman effect

        # -----------------Observables-----------------#
        MSx[n] = mSx

    return MSx

T=100
dt1=1e-2
dt2=5e-6

Msy1=master_equation1(3/2,1,1,T,dt1)
Msy2=master_equation2(3/2,1,1,T,dt2)


tt1=np.arange(0,T,dt1)
tt2=np.arange(0,T,dt2)

plt.style.use(['science'])
with plt.style.context(['science']):
    fig = plt.figure()
    plt.rc('font',family='Times New Roman')
    ax1 = fig.add_subplot(111)
    p1,=ax1.plot(tt1, Msy1)
    p2,=ax1.plot(tt2, Msy2)
    ax1.set_ylabel('$\langle Sx \\rangle$')
    ax1.set_xlabel('$t $')
    ax1.legend([p1,p2], ["Effective","First principle"],ncol=1)
    ax1.set_title('dt=5e-6, $\omega_0=1$,$R_{\\text{se}}=1$,$A_{\\text{hf}}=200$')
    ax1.set_xlim([0,50])
plt.savefig('FID_.png', dpi=1000)
plt.show()
